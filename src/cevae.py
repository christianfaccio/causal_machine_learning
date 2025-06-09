from original_cevae import CEVAE
from data_gen import synthetic_dataset_pyro
import matplotlib.pyplot as plt
import torch
from scipy import stats
import numpy as np

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
info = cevae.fit(x, t, y,
                   num_epochs=500,
                   batch_size=128,
                   learning_rate=5e-4,
                   learning_rate_decay=0.1,
                   weight_decay=1e-5,
                   track_params=True,
                   track_every=10)

losses = info['losses']
decoder_params = info['decoder_params']

# --------------------------------- evaluate --------------------------------- #
ate_est  = cevae.ite(x, num_samples=100).mean().item()
ate_true = data["ite"].mean().item()
print(f"Estimated ATE: {ate_est:.4f}")
print(f"True ATE:      {ate_true:.4f}")

# --------------------------------- plotting true p(x) vs approximated p(x|z) -------------------------------- #

plt.figure(figsize=(15, 10))

# Plot 1: Evolution of decoder PDFs with shadow effect
plt.subplot(2, 2, 1)
feature_dim = 0  # First feature

# Plot true data distribution (marginal p(x))
data_feature = x[:, feature_dim].numpy()
kde = stats.gaussian_kde(data_feature)
x_range_true = np.linspace(data_feature.min(), data_feature.max(), 1000)
plt.plot(x_range_true, kde(x_range_true), 'red', linewidth=3, label='True p(x) - KDE', alpha=0.8)

# Plot decoder approximations p(x|z) with increasing opacity
num_params = len(decoder_params)
for i, params in enumerate(decoder_params):
    # Fixed key names based on the modified CEVAE code
    mean_x = params["x_mean"][feature_dim].item()
    var_x = params["x_var"][feature_dim].item()
    std_x = np.sqrt(var_x)
    
    # Create alpha that increases over training (shadow effect)
    alpha = 0.1 + 0.7 * (i / (num_params - 1))
    
    # Define range for this specific distribution
    x_range = np.linspace(mean_x - 4 * std_x, mean_x + 4 * std_x, 1000)
    pdf_x = stats.norm.pdf(x_range, loc=mean_x, scale=std_x)
    
    plt.plot(x_range, pdf_x, 'blue', alpha=alpha, linewidth=1.5)

plt.xlabel('Feature Value')
plt.ylabel('Probability Density')
plt.title(f'True vs Decoder PDFs - Feature {feature_dim + 1}\n(Light→Dark: Early→Late Training)')
plt.legend()
plt.grid(True, alpha=0.3)

# Plot 2: Parameter evolution over training
plt.subplot(2, 2, 2)
steps = [params["step"] for params in decoder_params]
means = [params["x_mean"][feature_dim].item() for params in decoder_params]
stds = [np.sqrt(params["x_var"][feature_dim].item()) for params in decoder_params]

plt.plot(steps, means, 'b-', marker='o', markersize=4, label='Mean')
plt.plot(steps, stds, 'r-', marker='s', markersize=4, label='Std Dev')
plt.xlabel('Training Step')
plt.ylabel('Parameter Value')
plt.title(f'Decoder Parameters Evolution - Feature {feature_dim + 1}')
plt.legend()
plt.grid(True, alpha=0.3)

# Plot 3: Comparison at final step
plt.subplot(2, 2, 3)
final_params = decoder_params[-1]
final_mean = final_params["x_mean"][feature_dim].item()
final_var = final_params["x_var"][feature_dim].item()
final_std = np.sqrt(final_var)

# True distribution
plt.plot(x_range_true, kde(x_range_true), 'red', linewidth=3, label='True p(x)', alpha=0.8)

# Final decoder distribution
x_range_final = np.linspace(final_mean - 4 * final_std, final_mean + 4 * final_std, 1000)
pdf_final = stats.norm.pdf(x_range_final, loc=final_mean, scale=final_std)
plt.plot(x_range_final, pdf_final, 'blue', linewidth=3, label=f'Final Decoder p(x|z)', alpha=0.8)

plt.xlabel('Feature Value')
plt.ylabel('Probability Density')
plt.title(f'Final Comparison - Feature {feature_dim + 1}')
plt.legend()
plt.grid(True, alpha=0.3)

# Add text with statistics
textstr = f'True: μ={data_feature.mean():.3f}, σ={data_feature.std():.3f}\nDecoder: μ={final_mean:.3f}, σ={final_std:.3f}'
plt.text(0.02, 0.98, textstr, transform=plt.gca().transAxes, fontsize=10,
         verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

# Plot 4: Training loss
plt.subplot(2, 2, 4)
plt.plot(losses)
plt.xlabel('Training Step')
plt.ylabel('Loss')
plt.title('CEVAE Training Loss')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.suptitle('CEVAE Decoder Analysis', fontsize=16, fontweight='bold', y=0.98)
plt.show()

# --------------------------------- Additional Analysis --------------------------------- #

def plot_multiple_features(decoder_params, x, max_features=4):
    """Plot comparison for multiple features"""
    num_features = min(max_features, x.shape[1])
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.flatten()
    
    for feat_idx in range(num_features):
        ax = axes[feat_idx]
        
        # True distribution
        data_feature = x[:, feat_idx].numpy()
        kde = stats.gaussian_kde(data_feature)
        x_range_true = np.linspace(data_feature.min(), data_feature.max(), 1000)
        ax.plot(x_range_true, kde(x_range_true), 'red', linewidth=3, label='True p(x)', alpha=0.8)
        
        # Decoder evolution
        num_params = len(decoder_params)
        for i, params in enumerate(decoder_params):
            mean_x = params["x_mean"][feat_idx].item()
            var_x = params["x_var"][feat_idx].item()
            std_x = np.sqrt(var_x)
            
            alpha = 0.1 + 0.7 * (i / (num_params - 1))
            x_range = np.linspace(mean_x - 3 * std_x, mean_x + 3 * std_x, 500)
            pdf_x = stats.norm.pdf(x_range, loc=mean_x, scale=std_x)
            ax.plot(x_range, pdf_x, 'blue', alpha=alpha, linewidth=1)
        
        ax.set_title(f'Feature {feat_idx + 1}')
        ax.set_xlabel('Feature Value')
        ax.set_ylabel('Probability Density')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # Hide unused subplots
    for i in range(num_features, 4):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    plt.suptitle('Multi-Feature Decoder Analysis', fontsize=16, fontweight='bold', y=0.98)
    plt.show()

# Run additional analysis
plot_multiple_features(decoder_params, x, max_features=4)

print(f"\nAnalysis Summary:")
print(f"- Number of tracking points: {len(decoder_params)}")
print(f"- Training steps: {decoder_params[0]['step']} to {decoder_params[-1]['step']}")
print(f"- Feature dimensions: {x.shape[1]}")