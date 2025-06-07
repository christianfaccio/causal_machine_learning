from pyro.contrib.cevae import CEVAE
from data_gen import synthetic_dataset_pyro
import matplotlib.pyplot as plt
import torch

# ------------------------------ data generation ----------------------------- #
data      = synthetic_dataset_pyro(n=2000, 
                                   beta=2.0, 
                                   num_proxies=10,
                                   prior_type="gaussian", 
                                   seed=42)
length_x  = data["x"].shape[1]
x, t, y   = (data["x"],
             data["t"],
             data["y"])

# -------------------------------- model specs ------------------------------- #
cevae = CEVAE(feature_dim=length_x,
              latent_dim=1,
              outcome_dist="normal",
              num_layers=5,
              hidden_dim=300)

# ----------------------------------- train ---------------------------------- #
losses = cevae.fit(x, t, y,
                   num_epochs=500,
                   batch_size=128,
                   learning_rate=5e-4,
                   learning_rate_decay=0.1,
                   weight_decay=1e-5)

# --------------------------------- evaluate --------------------------------- #
ate_est  = cevae.ite(x, num_samples=100).mean().item()
ate_true = data["ite"].mean().item()
print(f"Estimated ATE: {ate_est:.4f}")
print(f"True ATE:      {ate_true:.4f}")

# plt.plot(losses)
# plt.xlabel('Epoch')
# plt.ylabel('Loss')
# plt.title('CEVAE Training Loss')
# plt.show()
# ----------------------------- save model params ---------------------------- #
torch.save(cevae.state_dict(), "cevae_state_dict.pt")
