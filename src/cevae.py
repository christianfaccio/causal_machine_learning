from data_gen import synthetic_dataset_pyro
import dowhy
from dowhy import CausalModel
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
import pyro
from pyro.contrib.cevae import CEVAE
import torch
import platform


### Generating data
data = synthetic_dataset_pyro(n=2000, beta=2.0, num_proxies=10,prior_type = "gaussian", seed=42)


### Transforming data in tensors of data
x = data["x"].detach().clone().float() if torch.is_tensor(data["x"]) else torch.tensor(data["x"], dtype=torch.float32)
t = data["t"].detach().clone().float() if torch.is_tensor(data["t"]) else torch.tensor(data["t"], dtype=torch.float32)
y = data["y"].detach().clone().float() if torch.is_tensor(data["y"]) else torch.tensor(data["y"], dtype=torch.float32)

### Estimating the ATE

y_1 = y[t==1]  # Outcomes for treated group
y_0 = y[t==0]  # Outcomes for control group

print("Unadjusted ATE", y_1.mean() - y_0.mean())


### Using CEVAE
pyro.set_rng_seed(42)
pyro.clear_param_store()

cevae = CEVAE(
    feature_dim=len(x[0]),
    latent_dim=1,
    outcome_dist='normal'
)

cevae.fit(x, t, y, num_epochs=500)

# Computing ITEs and ATE
ite = cevae.ite(x, num_samples=100)
ate = ite.mean().item()

# Compare
print(f"Estimated ATE: {ate:.4f}")
print(f"True ATE: {data['ite'].mean().item():.4f}")