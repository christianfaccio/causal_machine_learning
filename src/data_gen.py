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

    # -------------- 1. Set seed for reproducibility -------------------
    if seed is not None:
        pyro.set_rng_seed(seed)
        torch.manual_seed(seed)

    # -------------- 2. Sample latent confounder z ---------------------
    if prior_type == "gaussian":
        # z ~ Normal(0, 1)
        z = dist.Normal(0.0, 1.0).sample([n])
    elif prior_type == "bimodal":
        # mixture: component ~ Bernoulli(0.5), then Normal(center, 1)
        comp = dist.Bernoulli(0.5).sample([n])  # 0 or 1
        # define centers: 1 → +2, 0 → -2
        centers = torch.where(comp == 1, torch.full([n], 2.0), torch.full([n], -2.0))
        z = centers + dist.Normal(0.0, 1.0).sample([n])
    else:
        raise ValueError("prior_type must be 'gaussian' or 'bimodal'")

    # -------------- 3. Generate proxy covariates x -------------------
    # a_j ~ Uniform(-1.5, 1.5) for each proxy dimension j
    a = dist.Uniform(-1.5, 1.5).sample([num_proxies])           # shape: [num_proxies]
    # noise ε_ij ~ Normal(0, sigma_x)
    eps = dist.Normal(0.0, sigma_x).sample([n, num_proxies])   # shape: [n, num_proxies]
    # x_ij = tanh(z_i) * a_j + ε_ij
    x = torch.tanh(z).unsqueeze(1) * a.unsqueeze(0) + eps      # shape: [n, num_proxies]

    # Optionally shuffle a fraction of proxy columns to break their link to z
    if shuffle_pct > 0.0:
        k = int(num_proxies * shuffle_pct)
        cols_to_shuffle = torch.randperm(num_proxies)[:k]
        for c in cols_to_shuffle:
            x[:, c] = x[torch.randperm(n), c]

    # -------------- 4. Sample treatment assignment t ------------------
    # p(t=1 | z) = sigmoid(beta * z)
    p_t = torch.sigmoid(beta * z)
    t = dist.Bernoulli(probs=p_t).sample()                     # shape: [n]

    # -------------- 5. Generate continuous outcome y -----------------
    # g(z) = sin(z) + 0.5 * z
    g = torch.sin(z) + 0.5 * z
    # τ(z) = 1 + 0.5 * z   (true ITE)
    tau = 1.0 + 0.5 * z
    # y_mean_i = g(z_i) + τ(z_i)*t_i
    y_mean = g + tau * t
    # add noise: y_i ~ Normal(y_mean_i, sigma_y)
    y = dist.Normal(y_mean, sigma_y).sample()                   # shape: [n]

    # -------------- 6. Return everything as a dict --------------------
    return {
        "x": x,                  # [n, num_proxies]
        "t": t,                  # [n]
        "y": y,                  # [n]
        "z": z,                  # [n]
        "ite": tau               # [n]  (since ITE = 1 + 0.5*z exactly)
    }

# Example usage:
data = synthetic_dataset_pyro(n=1000, beta=2.0, num_proxies=5, shuffle_pct=0.2, seed=42)
print({k: v.shape for k, v in data.items()})