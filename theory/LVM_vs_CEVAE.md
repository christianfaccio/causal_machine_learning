# LVM vs CEVAE Comparison

Consider the dataset of i.i.d. observations $\{(x_i,t_i,y_i)\}_{i=1}^N$ and let $z_i$ be the latent variable relative to **each** observation.

### Generative Model

We assume the following factorization 

$$
p_\theta(z,x,t,y)=\prod_{i=1}^N p_\theta(z_i,x_i,t_i,y_i)\\
p_\theta(z_i,x_i,t_i,y_i) = p(z_i)p_\theta(x_i|z_i)p_\theta(t_i|z_i)p_\theta(y_i|z_i, t_i)
$$

| $\text{Probability Distribution}$ | $\text{LVM}$ | $\text{DLVM}$ |
|---|---|---|
| $p(z_i)$ | $\mathcal{N}(z_i\|0,1)$ | $\mathcal{N}(z_i\|0,1)$ |
| $p_\theta(x_i\|z_i)$|  $\mathcal{N}(x_i\|az_i+b,\sigma_X^2)$ | $\mathcal{N}(x_i\|g_\theta^X(z_i))$ |
| $p_\theta(t_i\|z_i)$ | $\text{Bernoulli}(t_i\mid\sigma(cz_i))$   | $\text{Bernoulli}(t_i\mid g_\theta^T(z_i))$  |
| $p_\theta(y_i\|z_i,t_i)$|  $\mathcal{N}(y_i\|et_i+fz_i,\sigma_Y^2)$ | $\mathcal{N}(y_i\|g_\theta^Y(z_i,t_i))$ |

where $g_\theta^X(z_i)$ is the Multi-Layer Perceptron (MLP) relative to the varible $X$ with weights $\theta$. 
- **LVM:** $\theta=(a,b,c,e,f,\sigma_X,\sigma_Y)$; 
- **DLVM:** $\theta$ stores all the weights of the neural network.

The generative model in the DLVM is basically the **decoder**.

### True Posterior

$$
p_\theta(z_i|x_i,t_i,y_i)=\frac{p_\theta(z_i,x_i,t_i,y_i)}{p_\theta(x_i,t_i,y_i)}
$$

**Remark:** because of the bernoulli factor, which contains a non-linearity, even the simple posterior of the LVM is not analitically tractable, therefore we'll use SVI.

### Variational Distribution

$$
q_\phi(z|x,t,y)=\prod_{i=1}^Nq_\phi(z_i|x_i,t_i,y_i)\\
q_{\phi_{i}}(z_i|x_i,t_i,y_i)=\mathcal{N}(z_i|\mu_{\phi_i}(x_i,t_i,y_i),\sigma_{\phi_i}^2(x_i,t_i,y_i))
$$

- **LVM:** we have one $q_{\phi_{i}}(z_i|x_i,t_i,y_i)$ distribution for each datapoint. Every $z_i$ generate "only" $x_i,t_i,y_i$, so the true posterior for each observation depends only on those variables. During the optimization we want to find $N$ **local parameters** $(\mu_{\phi_i},\sigma_{\phi_i}^2)$. The problem of this approach is that for each new data point $(x_i,t_i,y_i)$ we need to optimize again to find its (approximate) posterior.
- **DLVM:** here $(\mu_{\phi}(x,t,y),\sigma_{\phi}^2(x,t,y))=g_\phi^Z(x,t,y)$ is again a MLP and $\phi$ (nn wheights) is a **global parameter**. Here we have the same flexibility of $N$ posterior distributions but through a funcion. This function is exactly the **encoder**.

### Evidence Lower Bound

$
\mathcal{L}_i(\theta,\phi)=\mathbb{E}_{q_{\phi_{i}}}[\log p_\theta(z_i,x_i,t_i,y_i)] - \mathbb{E}_{q_{\phi_{i}}}[\log q_{\phi_{i}}(z_i|x_i,t_i,y_i)] \\

\mathcal{L}=\sum_{i=1}^N\mathcal{L}_i
$

#### Estimating the Gradient

- Generative parameters: 
$$
\nabla_\theta\mathcal{L}=\sum_{i\in B}\nabla_\theta[\log p_\theta(z_i,x_i,t_i,y_i)]
$$
- Guide parameters:
$$
\nabla_\phi\mathcal{L}=\sum_{i\in B}\nabla_{z_i}[\log p_\theta(z_i,x_i,t_i,y_i)-\log q_{\phi_{i}}(z_i|x_i,t_i,y_i)] \nabla_\phi z_i
$$

where $z_i=\mu_i+\sigma_i\cdot\epsilon_i\,,\quad \epsilon_i\sim\mathcal{N}(0,1)$, using the reparametrization trick the gradient over $\phi$ can be computed.

During every optimization step and for each mini-batch, through `svi.step` the idea is to 
- sample $\epsilon_i$ and compute $z_i$
- compute the montecarlo estimate of the gradients above
- perform an adam step over $\theta,\phi$

### Inference for an unseen data point $x^*$

As mentioned before, for the LVM in order to perform inference starting from a new data point $x^*$ we have to train a new small guide. The model is the same generative one mentioned before but without the factor relative to $y$ (that we want to predict) and such that $\theta$ is fixed and is the one found during the training phase. The new guide of such model aim to find a new $q(z^*|x^*,t^*)$ (remember that in our problem $t\in\{0,1\}$ so we can compute the posterior on $z$ for both values of $t$). In particular, the goal of this shorter (for compute-cost efficiency) train is to find the posterior parameters $(\mu_*,\sigma_*^2)$.

If we use the DLVM the value of $(\mu_*,\sigma_*^2)$ is straightforward to obtain, given the relative encoder network already trained, the posterior parameters are given "for free" as outcome of such NN. Basically the **encoder network** subtitutes the $2N$ local parameters $(\mu_i,\sigma_i^2)$ created by the `AutoDiagonalNormal` in the linear LVM.

It is important to note that the simple MLP encoder $(\mu_{\phi}(x,t,y),\sigma_{\phi}^2(x,t,y))=g_\phi^Z(x,t,y)$ does not allow to perform inference starting from a single new data point $x^*$, because such encoder requires three inputs. In order to address this problem the CEVAE model contains some refinements that allow inferences to be made starting from $x^*$ alone, *i.e.*, more MLPs.