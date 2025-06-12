<div align="center">
    <h1>From Latent to Deep Latent</h1>
    <h2>Causal Inference with CEVAE<h2>
    <h3>Authors: Valeria De Stasio, Christian Faccio, Giovanni Lucarelli</h3>
    <h6>This project explores the problem of causal inference, where we want to make inference on the hidden confounder Z, which is represented by some proxies X and influences both the treatment T and the outcome Y of an experiment. Starting from a basic approach where we did not consider hidden confounders at all, we delve deeper in the latent variables model with SVI optimization and in the CEVAE model, experimenting with the underlying architecture and comparing the results.</h6>
</div>

--- 

This repository is organized in the following way:

- `./References` contains the papers we built our project on
- `./src` contains the Python code used for the project
    
    - `./results` contains all the plots of the results of the experiments
    - `causal_effect_inference.ipynb` is the main notebook containing all the procedures done in this project
    - `data_generation.py` contains the data generating functions
    - `experiments.py` contains the functions to run the experiments
    - `models.py` contains the causal inference model classes
    - `synthetic_dataset.ipynb` contains graphs of the synthetic generated data
    

- `./slides` contains the slides for the presentation
- `./theory` contains the underlying theory of this project

    - `causal_ml_theory.md` covers the main theory about causal machine learning
    - `cevae_theory.md` explains the CEVAE architecture
    - `TARNet.md` covers the neural network architecture underlying the CEVAE


---

References:

- Louizos, Christos, et al. "Causal effect inference with deep latent-variable models." Advances in neural information processing systems 30 (2017).
- Hoffman, Matthew D., et al. "Stochastic variational inference." the Journal of machine Learning research 14.1 (2013): 1303-1347.
- Dang, Khue-Dung, and Luca Maestrini. "Fitting structural equation models via variational approximations." Structural Equation Modeling: A Multidisciplinary Journal 29.6 (2022): 839-853.
- Rissanen, Severi, and Pekka Marttinen. "A critical look at the consistency of causal estimation with deep latent variable models." Advances in Neural Information Processing Systems 34 (2021): 4207-4217.



