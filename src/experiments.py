import torch
import numpy as np
from sklearn.model_selection import train_test_split
import itertools
import pandas as pd
import pyro
import matplotlib.pyplot as plt
import seaborn as sns
from pyro.contrib.cevae import CEVAE
import os
from pyro import poutine

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
    # build list of all combinations
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
                raise ValueError(f"Unknown stage {stage}")

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

        # build result row
        row = dict(combo)
        row["final_elbo"] = losses[-1] 
        for mname, mfn in metrics_fns.items():
            row[mname] = mfn(model, x_te, ite_te)
        records.append(row)

    return pd.DataFrame.from_records(records)


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


# ---------------------------- plotting functions ---------------------------- #

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
    models=('cevae', 'lr', 'lvm'),
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


# @torch.no_grad()
# def plot_predicted_vs_true(
#     cevae,
#     x_test,
#     true_g_fn,
#     true_tau_fn,
#     num_samples: int = 100,
#     figsize: tuple = (15,6),
#     z_dim: int = 0,
#     invert = False
# ):
#     # whiten and sample
#     x = cevae.whiten(x_test)
#     with pyro.plate("num_particles", num_samples, dim=-2):
#         with poutine.trace() as tr, poutine.block(hide=["y", "t"]):
#             cevae.guide(x)
#         z = tr.trace.nodes["z"]["value"]
#         with poutine.do(data={"t": torch.zeros(())}):
#             y0_pred = poutine.replay(cevae.model.y_mean, tr.trace)(x)
#         with poutine.do(data={"t": torch.ones(())}):
#             y1_pred = poutine.replay(cevae.model.y_mean, tr.trace)(x)

#     # flatten & sort
#     z_flat  = z.reshape(-1, z.shape[-1])
#     y0_flat = y0_pred.reshape(-1)
#     y1_flat = y1_pred.reshape(-1)
#     z_plot  = z_flat[:, z_dim]
#     idx     = torch.argsort(z_plot)
#     z_s     = z_plot[idx]
#     if invert:
#         z_s = -z_s  # invert z if needed
#     y0_s    = y0_flat[idx]
#     y1_s    = y1_flat[idx]

#     # true curves
#     y0_true = true_g_fn(z_s)
#     tau_true = true_tau_fn(z_s)
#     y1_true = y0_true + tau_true

#     # Compute common y-limits
#     all_y = torch.cat([y0_s, y1_s, y0_true, y1_true]).cpu()
#     y_min, y_max = all_y.min().item(), all_y.max().item()

#     # plot with shared y-axis
#     fig, (ax0, ax1) = plt.subplots(
#         1, 2,
#         figsize=figsize,
#         sharey=True  
#     )

#     # Control
#     ax0.scatter(z_s.cpu(), y0_s.cpu(), alpha=0.4, s=10, label="Predicted")
#     ax0.plot   (z_s.cpu(), y0_true.cpu(),  '-', linewidth=3, label="True g(z)")
#     ax0.set(xlabel=f"Z[{z_dim}]", ylabel="Outcome", title="Control")
#     ax0.set_ylim(y_min, y_max)  # enforce identical limits
#     ax0.legend(); ax0.grid(True, alpha=0.3)

#     # Treatment
#     ax1.scatter(z_s.cpu(), y1_s.cpu(), alpha=0.4, s=10, label="Predicted")
#     ax1.plot   (z_s.cpu(), y1_true.cpu(),   '-', linewidth=3, label="True g(z)+τ(z)")
#     ax1.set(xlabel=f"Z[{z_dim}]", title="Treatment")
#     ax1.set_ylim(y_min, y_max)  # same limits as ax0
#     ax1.legend(); ax1.grid(True, alpha=0.3)

#     plt.tight_layout()
#     plt.show()
#     return fig, (ax0, ax1)

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import torch, pyro
from pyro import poutine

def _lighter(color, alpha=.3):
    """Return RGBA tuple of `color` with reduced opacity (looks lighter)."""
    rgba = list(mcolors.to_rgba(color))
    rgba[-1] = alpha              # only change the alpha channel
    return tuple(rgba)

@torch.no_grad()
def plot_predicted_vs_true(
    cevae,
    x_test,
    true_g_fn,
    true_tau_fn,
    num_samples: int = 100,
    figsize: tuple = (15, 6),
    z_dim: int = 0,
    invert: bool = False,

    colors: tuple = ("tab:blue", "tab:orange"),  # control, treatment
):

    x = cevae.whiten(x_test)
    with pyro.plate("num_particles", num_samples, dim=-2):
        with poutine.trace() as tr, poutine.block(hide=["y", "t"]):
            cevae.guide(x)
        z = tr.trace.nodes["z"]["value"]
        with poutine.do(data={"t": torch.zeros(())}):
            y0_pred = poutine.replay(cevae.model.y_mean, tr.trace)(x)
        with poutine.do(data={"t": torch.ones(())}):
            y1_pred = poutine.replay(cevae.model.y_mean, tr.trace)(x)


    z_flat, y0_flat, y1_flat = z.reshape(-1, z.shape[-1]), y0_pred.reshape(-1), y1_pred.reshape(-1)
    z_plot = z_flat[:, z_dim]
    idx = torch.argsort(z_plot)
    z_s = -z_plot[idx] if invert else z_plot[idx]
    y0_s, y1_s = y0_flat[idx], y1_flat[idx]


    y0_true = true_g_fn(z_s)
    tau_true = true_tau_fn(z_s)
    y1_true = y0_true + tau_true

    # identical y-limits
    y_min, y_max = torch.cat([y0_s, y1_s, y0_true, y1_true]).min().item(), torch.cat([y0_s, y1_s, y0_true, y1_true]).max().item()

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=figsize, sharey=True)

    # Control (scatter lighter, line darker)
    light_c0 = _lighter(colors[0])
    ax0.scatter(z_s.cpu(), y0_s.cpu(), s=10, color=light_c0, label="Predicted")
    ax0.plot   (z_s.cpu(), y0_true.cpu(), color=colors[0], linewidth=3, label="True g(z)")
    ax0.set(xlabel=f"Z[{z_dim}]", ylabel="Outcome", title="Control", ylim=(y_min, y_max))
    ax0.legend(); ax0.grid(True, alpha=.3)

    # Treatment
    light_c1 = _lighter(colors[1])
    ax1.scatter(z_s.cpu(), y1_s.cpu(), s=10, color=light_c1, label="Predicted")
    ax1.plot   (z_s.cpu(), y1_true.cpu(), color=colors[1], linewidth=3, label="True g(z)+τ(z)")
    ax1.set(xlabel=f"Z[{z_dim}]", title="Treatment", ylim=(y_min, y_max))
    ax1.legend(); ax1.grid(True, alpha=.3)

    plt.tight_layout()
    plt.show()
    return fig, (ax0, ax1)
