import torch
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import pyro
import pyro.distributions as dist
from pyro.infer import SVI, Trace_ELBO, autoguide, Predictive
from pyro.optim import Adam
from pyro import poutine


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
        sigmax = pyro.param("sigmax", torch.tensor(1.))
        sigmay = pyro.param("sigmay", torch.tensor(1.))

        with pyro.plate("data", N):
            # z is now a vector of length [1,L]
            z = pyro.sample("z",
                            dist.Normal(0., 1.)
                                .expand([L])
                                .to_event(1))           

            # x_obs: project z [1,L] → [1,D] via matrix multiply
            loc_x = z.matmul(a) + b             # (1,L) @ (L,D) → (1,D)
            pyro.sample("x_obs",
                        dist.MultivariateNormal(loc_x, sigmax*torch.eye(D)),
                        obs=x)

            # t_obs: logistic on a linear combination of z
            logits_t = (z * c).sum(-1)          # dot‐product → (1,)
            t = pyro.sample("t_obs",
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