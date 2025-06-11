import torch
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import itertools
import pandas as pd
import pyro
import pyro.distributions as dist
from pyro.infer import SVI, Trace_ELBO, autoguide
from pyro.optim import Adam
from pyro.infer import Predictive
import matplotlib.pyplot as plt
import seaborn as sns

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

class PyroLinearProxyModel:
    def __init__(self, init_scale=0.1, learning_rate=1e-2, weight_decay=0.0,local_steps=50):
        """
        A simple 1-dim latent SVI model whose estimated ATE is the learned scalar 'e'.
        """
        self.init_scale     = init_scale
        self.learning_rate  = learning_rate
        self.weight_decay   = weight_decay
        self.local_steps    = local_steps  # for ITE estimation  

    def model(self, x, t, y):
        N, D = x.shape
        a      = pyro.param("a",      torch.zeros(D))
        b      = pyro.param("b",      torch.zeros(D))
        c      = pyro.param("c",      torch.tensor(0.))
        e      = pyro.param("e",      torch.tensor(0.))   # <-- ATE
        f      = pyro.param("f",      torch.tensor(0.))
        sigmax = pyro.param("sigmax", torch.ones(D))
        sigmay = pyro.param("sigmay", torch.tensor(1.))

        with pyro.plate("data", N):
            z = pyro.sample("z", dist.Normal(0., 1.))
            loc_x = a * z.unsqueeze(-1) + b
            pyro.sample("x_obs", dist.Normal(loc_x, sigmax).to_event(1), obs=x)
            pyro.sample("t_obs", dist.Bernoulli(logits=c * z),        obs=t)
            pyro.sample("y_obs", dist.Normal(e * t + f * z, sigmay),  obs=y)

    def model_xt(self, x, t):
        """
        Same generative model but only observing x and t, so we can
        infer z|x,t at test time (for potential‐outcome sampling).
        """
        N, D = x.shape
        a      = pyro.param("a")
        b      = pyro.param("b")
        c      = pyro.param("c")
        e      = pyro.param("e")
        f      = pyro.param("f")
        sigmax = pyro.param("sigmax")
        sigmay = pyro.param("sigmay")  # not used as obs here

        with pyro.plate("data", N):
            z = pyro.sample("z", dist.Normal(0., 1.))
            loc_x = a * z.unsqueeze(-1) + b
            pyro.sample("x_obs", dist.Normal(loc_x, sigmax).to_event(1), obs=x)
            pyro.sample("t_obs", dist.Bernoulli(logits=c * z),        obs=t)
            # we could sample y_latent = Normal(e*t + f*z, sigmay) but we don't need it

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

    # def ite(self, x):
    #     """
    #     Predict individual treatment effects for each row of x:
    #       ITE_i = E[z|x,t=1]*f + e  -  E[z|x,t=0]*f
    #     """
    #     N = x.shape[0]
    #     t0 = torch.zeros(N, dtype=torch.float32)
    #     t1 = torch.ones(N,  dtype=torch.float32)

    #     post0 = self.predictive_xt(x, t0)["z"]  # shape [S, N]
    #     post1 = self.predictive_xt(x, t1)["z"]  # shape [S, N]

    #     z0 = post0.mean(0)  # [N]
    #     z1 = post1.mean(0)  # [N]

    #     e = pyro.param("e")
    #     f = pyro.param("f")

    #     ite = (e + f * z1) - (f * z0)
    #     return ite

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

            # 4) grab the freshly learned posterior means
            return pyro.param("AutoDiagonalNormal.loc")

        # infer for controls (t=0) and treatment (t=1)
        t0 = torch.zeros(N_new, device=device)
        t1 = torch.ones(N_new,  device=device)

        z0_loc = infer_loc(t0)   # shape [N_new]
        z1_loc = infer_loc(t1)   # shape [N_new]

        # read off your generative params
        e = pyro.param("e")
        f = pyro.param("f")

        # compute ITE_i = [e + f * z1_i] - [f * z0_i]
        return (e + f * z1_loc) - (f * z0_loc)

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

        # calcoliamo l’ITE per ogni unità
        ite = (e + f * z1) - (f * z0)
        return ite

    def estimate_ate(self, x=None, t=None, y=None):
        """
        The ATE is exactly the learned 'e' scalar.
        """
        return pyro.param("e").item()

    def get_all_params(self):
        """
        Returns a dict mapping *every* pyro.param name
        to its final tensor value.
        """
        return self.all_params
    
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

# ---------------------------- plot function --------------------------------- #

def plot_three_experiment_results(ceave_res, linear, pgm, x_param="data__shuffle_pct", palette_name="Set2"):
    """
    Plot results and linear_results overlapping for each metric on the same plot,
    using a nicer seaborn color palette.
    """
    # set up seaborn styling
    sns.set_style("whitegrid")
    sns.set_context("talk")  # makes labels/text a bit larger

    # find all metric columns
    metrics = [c for c in ceave_res.columns if c not in (x_param, "final_elbo")]

    # grab exactly 3 colors from the specified palette
    colors = sns.color_palette(palette_name, n_colors=3)

    for metric in metrics:
        plt.figure(figsize=(10, 6))
        plt.plot(ceave_res[x_param], ceave_res[metric],
                 marker="o", label="CEVAE", color=colors[0])
        plt.plot(linear[x_param], linear[metric],
                 marker="x", label="Linear Regression", color=colors[1])
        plt.plot(pgm[x_param], pgm[metric],
                 marker="^", label="PGM", color=colors[2])

        plt.xlabel(x_param, fontsize=12)
        plt.ylabel(metric.replace('_', ' ').title(), fontsize=12)
        plt.title(f"{metric.replace('_',' ').title()} vs {x_param}", fontsize=14)
        plt.legend(frameon=True)
        plt.tight_layout()
        plt.show()


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
    
    else:
        raise ValueError("prior_type must be 'gaussian' or 'bimodal'")

    # ------------------------------ Proxy variables ----------------------------- #

    # a_j ~ Uniform(-1.5, 1.5) for each proxy dimension j
    a = dist.Uniform(-1.5, 1.5).sample([num_proxies])

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
    z = dist.Normal(0.0, 1.0).sample([n])                # [n]

    # --------------------------- linear proxies ---------------------------- #
    a = dist.Uniform(-10, 10).sample([num_proxies])
    
    eps = dist.Normal(0.0, sigma_x).sample([n, num_proxies])

    
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