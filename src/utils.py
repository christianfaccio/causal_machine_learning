import torch
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import itertools
import pandas as pd
import pyro
import pyro.distributions as dist
from pyro.infer import SVI, Trace_ELBO, autoguide, Predictive
from pyro.optim import Adam
import matplotlib.pyplot as plt
import seaborn as sns
from pyro.contrib.cevae import CEVAE
import os


# ------------------------ classes for causal inference --------------------- #

class LinearModel:
    def __init__(self, **kwargs):
        self.model = LinearRegression(**kwargs)

    def fit(self, x, t, y, **fit_kwargs):
        # Convert to numpy if input is torch.Tensor
        if isinstance(x, torch.Tensor):
            x = x.detach().cpu().numpy()
        if isinstance(t, torch.Tensor):
            t = t.detach().cpu().numpy()
        if isinstance(y, torch.Tensor):
            y = y.detach().cpu().numpy()

        X = np.concatenate([t[:, None], x], axis=1)
        self.model.fit(X, y, **fit_kwargs)
        y_pred = self.model.predict(X)
        mse = mean_squared_error(y, y_pred)
        return [mse]  # Fake ELBO for compatibility

    def ite(self, x):
        if isinstance(x, torch.Tensor):
            x_np = x.detach().cpu().numpy()
        else:
            x_np = x

        X1 = np.concatenate([np.ones((x_np.shape[0], 1)), x_np], axis=1)
        X0 = np.concatenate([np.zeros((x_np.shape[0], 1)), x_np], axis=1)
        y1 = self.model.predict(X1)
        y0 = self.model.predict(X0)
        diff = y1 - y0
        return torch.tensor(diff, dtype=torch.float32).reshape(-1, 1)
    
# ---------------------------- experiments function -------------------------- #
def run_experiment(
    param_grid: dict,
    data_fn: callable,
    model_cls: type,
    metrics_fns: dict,
    data_kwargs: dict,
    model_kwargs: dict,
    fit_kwargs: dict,
    test_size: float = 0.2,
    random_state: int = 0,
):
    """
    Sweeps over one or more parameters (given in param_grid) and returns a DataFrame
    of results.

    param_grid: mapping from "stage__param_name" to list of values.
       e.g. {"data__shuffle_pct": np.linspace(0,1,11),
              "model__latent_dim": [1,2,5]}

    data_fn:        function(**data_kwargs) -> dict with keys "x","t","y","ite"
    
    model_cls:      class of your model; instantiated as model_cls(**model_kwargs)
    
    fit_kwargs:     kwargs passed to model.fit(...)
    
    metrics_fns:    mapping from metric_name -> function(model, x_te, ite_te) -> float

    Returns a pandas DataFrame with one row per combination, columns for each
    swept param plus each metric.
    """
    # 1) build list of all combinations
    keys, values = zip(*param_grid.items())
    combos = [dict(zip(keys, combo)) for combo in itertools.product(*values)]

    records = []
    for combo in combos:
        # prepare stage-specific kwargs
        dk = data_kwargs.copy()
        mk = model_kwargs.copy()
        fk = fit_kwargs.copy()

        # dispatch combo entries into dk/mk/fk
        for full_key, val in combo.items():
            stage, param = full_key.split("__", 1)
            if stage == "data":
                dk[param] = val
            elif stage == "model":
                mk[param] = val
            elif stage == "fit":
                fk[param] = val
            else:
                raise ValueError(f"Unknown stage “{stage}”")

        if model_cls is CEVAE and "num_proxies" in dk:
            mk["feature_dim"] = int(dk["num_proxies"])

        # run one trial
        data = data_fn(**dk)
        x, t, y, ite = data["x"], data["t"], data["y"], data["ite"]
        x_tr, x_te, t_tr, t_te, y_tr, y_te, ite_tr, ite_te = train_test_split(
            x, t, y, ite, test_size=test_size, random_state=random_state
        )

        pyro.clear_param_store()
        model = model_cls(**mk)
        losses = model.fit(x_tr, t_tr, y_tr, **fk)

        # 4) build result row
        row = dict(combo)
        row["final_elbo"] = losses[-1] 
        for mname, mfn in metrics_fns.items():
            row[mname] = mfn(model, x_te, ite_te)
        records.append(row)

    return pd.DataFrame.from_records(records)

# ---------------------------- metrics definition ---------------------------- #

def abs_ate_error(model, x_te, ite_te):
    est = model.ite(x_te).mean().detach().cpu().numpy()  # Detach before converting to NumPy
    true = ite_te.mean().detach().cpu().numpy()          # Detach before converting to NumPy
    return abs(est - true)

def rel_ate_error(model, x_te, ite_te):
    est = model.ite(x_te).mean().detach().cpu().numpy()  # Detach before converting to NumPy
    true = ite_te.mean().detach().cpu().numpy()          # Detach before converting to NumPy
    return abs(est - true) / abs(true)

def rmse_ite(model, x_te, ite_te):
    pred = model.ite(x_te).detach().cpu().numpy()
    true = ite_te.cpu().numpy()
    rmse = np.sqrt(np.mean((pred - true)**2))
    return rmse

def synthetic_dataset_non_linear(
    n: int = 2000,
    beta: float = 2.0,
    num_proxies: int = 10,
    shuffle_pct: float = 0.0,
    prior_type: str = "gaussian",
    sigma_x: float = 1.0,
    sigma_y: float = 1.0,
    rho: float = 0.0,   
    seed: int | None = None,
):
    """
    Data generator for a simple causal model with proxies and latent confounders.
    Args:
      n: int               (number of samples)
      beta: float          (treatment effect strength)
      num_proxies: int     (number of proxy covariates)
      shuffle_pct: float   (fraction of proxies to shuffle,i.e., uninformative proxies, 0.0 to 1.0)
      prior_type: str      ('gaussian' or 'bimodal' for latent confounder z)
      sigma_x: float       (noise level for proxy covariates)
      sigma_y: float       (noise level for outcome y)
      seed: int | None     (random seed for reproducibility)
    Returns:
      x: [n, num_proxies]   (proxy covariates)
      t: [n]               (binary treatment)
      y: [n]               (continuous outcome)
      z: [n]               (latent confounder, continuous or bimodal)
      ite: [n]             (true individual treatment effect = 1 + 0.5*z, s.t. the true ATE is 1)
    """

    # --------------------------------- set seed --------------------------------- #
    if seed is not None:
        pyro.set_rng_seed(seed)
        torch.manual_seed(seed)

    # ----------------------------- Latent Confounder ---------------------------- #

    if prior_type == "gaussian":
        # z ~ Normal(0, 1)
        z = dist.Normal(0.0, 1.0).sample([n])
    elif prior_type == "bimodal":
        # mixture: component ~ Bernoulli(0.5), then Normal(center, 1)
        comp = dist.Bernoulli(0.5).sample([n])  # 0 or 1
        # define centers: 1 to +2, 0 to -2
        centers = torch.where(comp == 1, torch.full([n], -5.0), torch.full([n], -2.0))
        z = centers + dist.Normal(0.0, 1.0).sample([n])
    elif prior_type == "lognormal":
        # z ~ LogNormal(0, 1)
        z = dist.LogNormal(0.0, 1.0).sample([n])
    else:
        raise ValueError("prior_type must be 'gaussian' or 'bimodal'")

    # ------------------------------ Proxy variables ----------------------------- #

    # a_j ~ Uniform(-1.5, 1.5) for each proxy dimension j
    a = dist.Uniform(-10, 10).sample([num_proxies])

    # --------------------------- correlated noise ------------------------- #
    if not (0.0 <= rho < 1.0):
        raise ValueError("rho must satisfy 0 ≤ rho < 1")

    if rho == 0.0:
        # independent noise  ——  identical to the original implementation
        eps = dist.Normal(0.0, sigma_x).sample([n, num_proxies])
    else:
        # build block-constant covariance Σ = σ²[(1-ρ)I + ρ11ᵀ]
        eye   = torch.eye(num_proxies)
        ones  = torch.ones(num_proxies, num_proxies)
        sigma = sigma_x ** 2 * ((1.0 - rho) * eye + rho * ones)
        mvn   = dist.MultivariateNormal(loc=torch.zeros(num_proxies), covariance_matrix=sigma)
        eps   = mvn.sample([n])  

    # x_ij = tanh(z_i) * a_j + eps_ij
    x = torch.tanh(z).unsqueeze(1) * a.unsqueeze(0) + eps     

    # Optionally shuffle a fraction of proxy columns to break their link to z
    # This simulates uninformative proxies that do not correlate with the latent confounder
    if shuffle_pct > 0.0:
        k = int(num_proxies * shuffle_pct)
        cols_to_shuffle = torch.randperm(num_proxies)[:k]
        for c in cols_to_shuffle:
            x[:, c] = x[torch.randperm(n), c]

    # --------------------------------- Treatment -------------------------------- #

    # p(t=1 | z) = sigmoid(beta * z)
    p_t = torch.sigmoid(beta * z)
    t = dist.Bernoulli(probs=p_t).sample()

    # ---------------------------------- Outcome --------------------------------- #

    # use sin(z) to create a non-linear relationship
    # and add a linear term with z to ensure non-periodicity
    # g(z) = sin(z) + 0.5 * z
    g = torch.sin(z) + 0.5 * z
    # tau(z) = 1 + 0.5 * z   (true ITE)
    # This ensures that the true average treatment effect (ATE) is 1
    tau = 1.0 + 0.5 * z
    # y_mean_i = g(z_i) + tau(z_i)*t_i
    y_mean = g + tau * t
    # add noise: y_i ~ Normal(y_mean_i, sigma_y)
    y = dist.Normal(y_mean, sigma_y).sample()

    return {
        "x": x,                  # [n, num_proxies]
        "t": t,                  # [n]
        "y": y,                  # [n]
        "z": z,                  # [n]
        "ite": tau               # [n]  (since ITE = 1 + 0.5*z exactly)
    }


def synthetic_dataset_linear(
    n: int = 2000,
    beta: float = 1.0,
    num_proxies: int = 10,
    shuffle_pct: float = 0.0,
    sigma_x: float = 1.0,
    sigma_y: float = 1.0,
    seed: int | None = None,
    rho: float = 0.0,
    prior_type: str = "gaussian"  # 'gaussian' or 'bimodal'
):
    """
    Linear-Gaussian (See NIMPS 2021).

    Model (all relationships are linear, noises are Gaussian):
        z           ~  N(0, 1)                                (latent confounder)
        eps_x_ij    ~  N(0, sigma_x)
        x_ij        =  a_j * z_i + eps_x_ij                   (proxies)

        t_i         ~  Bernoulli( sigmoid(beta * z_i) )       (treatment)

        eps_y_i     ~  N(0, sigma_y)
        y_i         =  gamma * z_i + tau * t_i + eps_y_i      (outcome)
                        where gamma = 1.0 and tau = 1.0       (→ true ATE = 1)

    Args
    ----
    n            : number of samples
    beta         : strength of z → t link (log-odds scale)
    num_proxies  : how many proxy covariates
    shuffle_pct  : fraction of proxy columns to shuffle (breaks z-x link)
    sigma_x      : std-dev of proxy noise
    sigma_y      : std-dev of outcome noise
    seed         : random seed for reproducibility

    Returns
    -------
    dict with keys
        x   : [n, num_proxies]  proxy covariates
        t   : [n]               binary treatment
        y   : [n]               continuous outcome
        z   : [n]               latent confounder
        ite : [n]               true individual treatment effect (all = 1)
    """
    # --------------------------- reproducibility --------------------------- #
    if seed is not None:
        pyro.set_rng_seed(seed)
        torch.manual_seed(seed)

    # --------------------------- latent confounder ------------------------- #
    if prior_type == "gaussian":
        # z ~ Normal(0, 1)
        z = dist.Normal(0.0, 1.0).sample([n])
    elif prior_type == "bimodal":
        # mixture: component ~ Bernoulli(0.5), then Normal(center, 1)
        comp = dist.Bernoulli(0.5).sample([n])  # 0 or 1
        # define centers: 1 to +2, 0 to -2
        centers = torch.where(comp == 1, torch.full([n], -5.0), torch.full([n], -2.0))
        z = centers + dist.Normal(0.0, 1.0).sample([n])
    elif prior_type == "lognormal":
        # z ~ LogNormal(0, 1)
        z = dist.LogNormal(0.0, 1.0).sample([n])
    else:
        raise ValueError("prior_type must be 'gaussian' or 'bimodal'")
              # [n]

    # --------------------------- linear proxies ---------------------------- #
    a = dist.Uniform(-10, 10).sample([num_proxies])
    
    # eps = dist.Normal(0.0, sigma_x).sample([n, num_proxies])
    if rho == 0.0:
        # independent noise  ——  identical to the original implementation
        eps = dist.Normal(0.0, sigma_x).sample([n, num_proxies])
    else:
        # build block-constant covariance Σ = σ²[(1-ρ)I + ρ11ᵀ]
        eye   = torch.eye(num_proxies)
        ones  = torch.ones(num_proxies, num_proxies)
        sigma = sigma_x ** 2 * ((1.0 - rho) * eye + rho * ones)
        mvn   = dist.MultivariateNormal(loc=torch.zeros(num_proxies), covariance_matrix=sigma)
        eps   = mvn.sample([n]) 
    
    x = z.unsqueeze(1) * a + eps                       # [n, p]

    # optionally destroy information in a subset of proxies
    if shuffle_pct > 0.0:
        k = int(num_proxies * shuffle_pct)
        cols_to_shuffle = torch.randperm(num_proxies)[:k]
        for c in cols_to_shuffle:
            x[:, c] = x[torch.randperm(n), c]

    # ---------------------------- treatment -------------------------------- #
    p_t = torch.sigmoid(beta * z)
    t = dist.Bernoulli(probs=p_t).sample()               # [n]

    # ----------------------------- outcome --------------------------------- #
    gamma = 1.0                                          # z → y coefficient
    tau   = 1.0                                          # constant ITE (true ATE = 1)
    eps_y = dist.Normal(0.0, sigma_y).sample([n])
    y = gamma * z + tau * t + eps_y

    return {
        "x":   x,
        "t":   t,
        "y":   y,
        "z":   z,
        "ite": torch.full_like(z, tau)                   # every unit has ITE = 1
    }

def compare_all(data_sets, model_defs, param_grid, metrics):
    results = {}
    for ds_name, (data_fn, data_kwargs) in data_sets.items():
        for m_name, (ModelCls, model_kwargs, fit_kwargs) in model_defs.items():
            pyro.clear_param_store()
            key = f"{ds_name}_{m_name}"
            results[key] = run_experiment(
                param_grid=param_grid,
                data_fn=data_fn,
                data_kwargs=data_kwargs,
                model_cls=ModelCls,
                model_kwargs=model_kwargs,
                fit_kwargs=fit_kwargs,
                metrics_fns=metrics,
            )
    return results

def _shade(color, factor):
    c = np.array(color)
    if factor >= 1:
        return tuple(c + (1 - c) * (factor - 1))
    else:
        return tuple(c * factor)

def plot_datasets(
    linear_res=None,
    nonlin_res=None,
    x_param="data__n",
    palette_name="Set2",
    light_factor=1.3,
    dark_factor=0.7,
    save_dir: str = None,
    file_name_template: str = "{metric}_vs_{x_param}.pdf",
    show: bool = True,
):
    """
    Plot curves for one or both dataset types, and optionally save as PDFs
    with custom filenames.

    Args:
      linear_res / nonlin_res: dicts model→DataFrame (or None)
      x_param: column to use for x axis
      palette_name, light_factor, dark_factor: styling params
      save_dir: if not None, directory to save PDFs (created if needed)
      file_name_template: Python format string for the filenames,
          with {metric} and {x_param} available.
          e.g. "myplot_{metric}.pdf" or "{x_param}--{metric}.pdf"
      show: whether to call plt.show() (otherwise closes figure)
    """
    if linear_res is None and nonlin_res is None:
        raise ValueError("Must pass at least one of linear_res or nonlin_res")
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    models = list(linear_res or nonlin_res)
    base_colors = sns.color_palette(palette_name, n_colors=len(models))

    sample_df = (next(iter(linear_res.values())) 
                 if linear_res is not None 
                 else next(iter(nonlin_res.values())))
    metrics = [c for c in sample_df.columns if c not in (x_param, 'final_elbo')]

    sns.set_style("whitegrid")
    sns.set_context("talk")

    for metric in metrics:
        plt.figure(figsize=(10, 6))
        for idx, model in enumerate(models):
            base = base_colors[idx]
            lin_col = (_shade(base, light_factor) 
                       if linear_res and nonlin_res else base)
            non_col = (_shade(base, dark_factor) 
                       if linear_res and nonlin_res else base)

            if linear_res:
                df_lin = linear_res[model]
                plt.plot(df_lin[x_param], df_lin[metric],
                         marker="o",
                         label=(f"{model.upper()} (linear)"
                                if nonlin_res else model.upper()),
                         color=lin_col)
            if nonlin_res:
                df_non = nonlin_res[model]
                plt.plot(df_non[x_param], df_non[metric],
                         marker="s",
                         label=(f"{model.upper()} (non-linear)"
                                if linear_res else model.upper()),
                         color=non_col)

        plt.xlabel(x_param, fontsize=12)
        plt.ylabel(metric.replace('_', ' ').title(), fontsize=12)
        plt.title(f"{metric.replace('_',' ').title()} vs {x_param}", fontsize=14)
        plt.legend(frameon=True, fontsize=10)
        plt.tight_layout()

        if save_dir:
            fname = file_name_template.format(metric=metric, x_param=x_param)
            path = os.path.join(save_dir, fname)
            plt.savefig(path)
        if show:
            plt.show()
        else:
            plt.close()


def auto_plot_datasets(
    all_results,
    x_param,
    plot_fn=plot_datasets,
    models=('cevae', 'lr', 'pgm'),
    ds_names=('linear', 'non_linear'),
    save_dir: str = None,
    file_name_template: str = "{metric}_vs_{x_param}.pdf",
    show: bool = True,
    **plot_kwargs
):
    """
    Extracts linear/non-linear runs from all_results and calls plot_datasets,
    passing through save_dir, file_name_template, show, and any other styling kwargs.
    """
    lin_key, nonlin_key = ds_names

    linear_res = {
        m: all_results[f"{lin_key}_{m}"]
        for m in models
        if f"{lin_key}_{m}" in all_results
    }
    nonlin_res = {
        m: all_results[f"{nonlin_key}_{m}"]
        for m in models
        if f"{nonlin_key}_{m}" in all_results
    }

    plot_fn(
        linear_res=linear_res or None,
        nonlin_res=nonlin_res or None,
        x_param=x_param,
        save_dir=save_dir,
        file_name_template=file_name_template,
        show=show,
        **plot_kwargs
    )


# ---------------------------------------------------------------------------- #
#                                 NEW PGM CLASS                                #
# ---------------------------------------------------------------------------- #


class PyroLinearProxyModel:
    def __init__(self,
                 init_scale: float    = 0.1,
                 learning_rate: float = 1e-2,
                 weight_decay: float  = 0.0,
                 local_steps: int     = 50,
                 latent_dim: int      = 1):
        """
        A simple latent‐dimensional SVI model whose ATE is the learned scalar 'e'.
        """
        self.init_scale     = init_scale
        self.learning_rate  = learning_rate
        self.weight_decay   = weight_decay
        self.local_steps    = local_steps
        self.latent_dim     = latent_dim   

    def model(self, x, t, y):
        N, D = x.shape
        L    = self.latent_dim

        # now a is L×D, c and f are length‐L
        a      = pyro.param("a",      torch.zeros(L, D))
        b      = pyro.param("b",      torch.zeros(D))
        c      = pyro.param("c",      torch.zeros(L))
        e      = pyro.param("e",      torch.tensor(0.))    # ATE scalar
        f      = pyro.param("f",      torch.zeros(L))
        sigmax = pyro.param("sigmax", torch.ones(D))
        sigmay = pyro.param("sigmay", torch.tensor(1.))

        with pyro.plate("data", N):
            # z is now a vector of length L
            z = pyro.sample("z",
                            dist.Normal(0., 1.)
                                .expand([L])
                                .to_event(1))           # z: [N, L]

            # x_obs: project z [N,L] → [N,D] via matrix multiply
            loc_x = z.matmul(a) + b             # (N,L) @ (L,D) → (N,D)
            pyro.sample("x_obs",
                        dist.Normal(loc_x, sigmax)
                            .to_event(1),
                        obs=x)

            # t_obs: logistic on a linear combination of z
            logits_t = (z * c).sum(-1)          # dot‐product → (N,)
            pyro.sample("t_obs",
                        dist.Bernoulli(logits=logits_t),
                        obs=t)

            # y_obs: ATE e*t plus projection f·z
            mean_y = e * t + (z * f).sum(-1)
            pyro.sample("y_obs",
                        dist.Normal(mean_y, sigmay),
                        obs=y)

    def model_xt(self, x, t):
        """
        Same as `model` but without observing y, so we can infer z|x,t at test time.
        """
        N, D = x.shape
        L    = self.latent_dim

        a      = pyro.param("a")
        b      = pyro.param("b")
        c      = pyro.param("c")
        e      = pyro.param("e")
        f      = pyro.param("f")
        sigmax = pyro.param("sigmax")
        # sigmay is unused here

        with pyro.plate("data", N):
            z = pyro.sample("z",
                            dist.Normal(0., 1.)
                                .expand([L])
                                .to_event(1))
            loc_x = z.matmul(a) + b
            pyro.sample("x_obs",
                        dist.Normal(loc_x, sigmax)
                            .to_event(1),
                        obs=x)
            logits_t = (z * c).sum(-1)
            pyro.sample("t_obs",
                        dist.Bernoulli(logits=logits_t),
                        obs=t)

    def fit(self,
            x, t, y,
            num_epochs: int = 1000,
            posterior_epochs: int = 1000,
            batch_size: int = 100,       # ignored here, we do full-batch
            learning_rate: float = None,
            weight_decay: float    = None,
            log_every: int        = 0
           ):
        """
        Runs SVI on (x, t, y), returns a list of ELBO losses.
        """
        # override defaults if provided
        lr = learning_rate or self.learning_rate
        wd = weight_decay  or self.weight_decay

        pyro.clear_param_store()
        guide = autoguide.AutoDiagonalNormal(self.model, init_scale=self.init_scale)
        optim = Adam({"lr": lr, "weight_decay": wd})
        svi   = SVI(self.model, guide, optim, loss=Trace_ELBO())

        losses = []
        for epoch in range(1, num_epochs+1):
            loss = svi.step(x, t, y)
            losses.append(loss)
            if log_every and epoch % log_every == 0:
                print(f"[SVI] epoch {epoch:>4} ELBO = {loss:.2f}")
        
        # # snapshot & freeze
        # raw = {name: val.detach().clone()
        #        for name, val in pyro.get_param_store().items()}
        # pyro.clear_param_store()
        # for name, val in raw.items():
        #     p = pyro.param(name, val)
        #     p.requires_grad_(False)

        # snapshot p-params (and only those) AFTER clearing guide1
        # i.e. clear then re-register only your p_params, then snapshot the names
        raw_p = {name: val.detach().clone()
                for name, val in pyro.get_param_store().items()
                if not name.startswith("AutoDiagonalNormal")}  # or simply list your p-param names
        pyro.clear_param_store()
        for name, val in raw_p.items():
            p = pyro.param(name, val)
            p.requires_grad_(False)

        # NOW snapshot the store keys – these are *only* your p-params*
        after1_names = set(pyro.get_param_store().keys())
        self.p_params = {name: pyro.param(name).detach().clone()
                        for name in after1_names}


        # now set up & train guide_xt on (x,t) only
        self.guide_xt = autoguide.AutoDiagonalNormal(
            self.model_xt, init_scale=self.init_scale
        )
        optim2 = Adam({
            "lr": self.learning_rate,
            "weight_decay": self.weight_decay
        })
        losses_xt = []
        svi2 = SVI(self.model_xt, self.guide_xt, optim2, loss=Trace_ELBO())
        
        for pos_epoch in range(1, posterior_epochs):
            loss_xt = svi2.step(x, t)    # note: no y here, model_xt only uses x,t
            losses_xt.append(loss_xt)
            if log_every and pos_epoch % log_every == 0:
                print(f"[SVI-xt] epoch {pos_epoch+1:>4} ELBO = {loss_xt:.2f}")
        
        after2_names = set(pyro.get_param_store().keys())

        # The difference now *is* just the new q-params
        q_only_names = after2_names - after1_names
        self.q_params = {
            name: pyro.param(name).detach().clone()
            for name in q_only_names
        }

        # build the predictive object _after_ training guide_xt
        self.predictive_xt = Predictive(
            self.model_xt,
            guide=self.guide_xt,
            num_samples=100,
            return_sites=["z"]
        )
        
        return losses

    def ite(self, x_new):
        """
        For a new batch x_new (shape [N_new,D]), compute ITE by:
         1) locally re-inferring q(z|x_new,t=0)
         2) locally re-inferring q(z|x_new,t=1)
         3) forming ITE_i = [e + f z_i^(1)] - [f z_i^(0)]
        """

        N_new = x_new.size(0)
        device = x_new.device

        # helper: given a fixed treatment vector t_vec of length N_new,
        # re-run a fresh AutoDiagonalNormal guide on (x_new, t_vec)
        # so that its latent size = N_new
        def infer_loc(t_vec):
            # 1) clear out any old guide params, but keep your frozen p-params
            pyro.clear_param_store()
            for name, val in self.p_params.items():
                p = pyro.param(name, val.to(device))
                p.requires_grad_(False)

            # 2) instantiate a brand-new guide that will now see plate("data",N_new)
            guide_local = autoguide.AutoDiagonalNormal(
                self.model_xt, init_scale=self.init_scale
            )
            svi_local = SVI(self.model_xt,
                            guide_local,
                            Adam({"lr": self.learning_rate,
                                  "weight_decay": self.weight_decay}),
                            loss=Trace_ELBO())

            # 3) do a few local SVI steps so that guide_local learns loc/scale of length N_new
            for _ in range(self.local_steps):
                svi_local.step(x_new, t_vec)

            raw_loc = pyro.param("AutoDiagonalNormal.loc")               # shape: [N_new * L]
            return raw_loc.view(N_new, self.latent_dim)                  # shape: [N_new, L]

        # infer for controls (t=0) and treatment (t=1)
        t0 = torch.zeros(N_new, device=device)
        t1 = torch.ones(N_new,  device=device)

        z0_loc = infer_loc(t0)   # shape [N_new]
        z1_loc = infer_loc(t1)   # shape [N_new]

        # read off your generative params
        e = pyro.param("e")
        f = pyro.param("f")
        # return (e + f * z1_loc) - (f * z0_loc)
        # compute ITE_i = [e + f * z1_i] - [f * z0_i]
        proj1 = (z1_loc * f).sum(dim=-1)
        proj0 = (z0_loc * f).sum(dim=-1)
        # now ITE is a length-N vector again
        return (e + proj1) - proj0

    def ite_train(self, x_train):
        """
        Calcola l'ITE per i dati di training (x_train, t_train) usando
        il guide_xt già addestrato e il Predictive obj.
        Restituisce un tensor [N_train] con:
            ITE_i = [e + f * E[z_i | x_train, t_i=1]] - [f * E[z_i | x_train, t_i=0]]
        """
        # costruiamo i vettori di intervento 0/1
        t0 = torch.zeros(len(x_train), dtype=torch.float32)
        t1 = torch.ones(len(x_train),  dtype=torch.float32)

        # campioniamo la posteriore di z per t=0 e per t=1
        post0 = self.predictive_xt(x_train, t0)["z"]  # shape [S, N_train]
        post1 = self.predictive_xt(x_train, t1)["z"]  # shape [S, N_train]

        # prendiamo la media su S campioni
        z0 = post0.mean(0)  # [N_train]
        z1 = post1.mean(0)  # [N_train]

        # leggiamo i parametri generativi
        e = pyro.param("e")
        f = pyro.param("f")

        # compute ite for each unit
        proj1 = (z1 * f).sum(dim=-1)
        proj0 = (z0 * f).sum(dim=-1)
        ite   = (e + proj1) - proj0
        return ite

    def estimate_ate(self, x=None, t=None, y=None):
        """
        The ATE is exactly the learned 'e' scalar.
        """
        return pyro.param("e").item()