import torch
import pyro
import pyro.distributions as dist

def synthetic_dataset_pyro(
    n: int = 2000,
    beta: float = 2.0,
    num_proxies: int = 10,
    shuffle_pct: float = 0.0,
    prior_type: str = "gaussian",
    sigma_x: float = 1.0,
    sigma_y: float = 1.0,
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
        centers = torch.where(comp == 1, torch.full([n], 2.0), torch.full([n], -2.0))
        z = centers + dist.Normal(0.0, 1.0).sample([n])
    else:
        raise ValueError("prior_type must be 'gaussian' or 'bimodal'")

    # ------------------------------ Proxy variables ----------------------------- #

    # a_j ~ Uniform(-1.5, 1.5) for each proxy dimension j
    a = dist.Uniform(-1.5, 1.5).sample([num_proxies])
    # noise eps_ij ~ Normal(0, sigma_x)
    eps = dist.Normal(0.0, sigma_x).sample([n, num_proxies])
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

import torch
import pyro
import pyro.distributions as dist


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
    a = dist.Normal(0.0, 1.0).sample([num_proxies])      # coefficients a_j
    eps_x = dist.Normal(0.0, sigma_x).sample([n, num_proxies])
    x = z.unsqueeze(1) * a + eps_x                       # [n, p]

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
