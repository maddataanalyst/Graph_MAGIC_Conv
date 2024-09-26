"""This module contains some helper functions used to build summaries
and visualizations of the achieved results
"""

from copy import deepcopy
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import optuna as opt
import pandas as pd
import pingouin as pg
import plotly as py
import seaborn as sns
from IPython.display import Markdown, display

import aml_magic.src.consts as cc

IMAGES_DIR = Path(".", "importance_images")
IMAGES_DIR.mkdir(exist_ok=True, parents=True)


def load_study(ds_name: str):
    return joblib.load(Path(cc.DATA_DIR, "hp_results", f"{ds_name}_study.pkl"))


def get_study_trials(study: opt.study.Study, sort_by: tuple = (1, 2), ascending: tuple = (False, True)):
    return study.trials_dataframe().sort_values(by=[f"values_{i}" for i in sort_by], ascending=ascending)

def get_feature_importance(study, target: int = 1):
    optuna_importance = opt.importance.PedAnovaImportanceEvaluator(
        baseline_quantile=0.1
    )
    importances = optuna_importance.evaluate(study=study, target=lambda t: t.values[1])
    return importances


def study_fig_path(ds_name: str, fig_name: str):
    ds_name_dir_friendly = ds_name.replace("/", "")
    ds_path = IMAGES_DIR / ds_name_dir_friendly
    ds_path.mkdir(exist_ok=True, parents=True)
    return ds_path / f"{ds_name_dir_friendly}_{fig_name}.pdf"


def anova_check_importance(
    study,
    target_col: str = "values_1",
    param_cols: tuple[str, ...] = (
        "Conv. Dim.",
        "No. Conv. Layers",
        "Aggr.",
        "Embed Reduction",
    ),
    alpha: float = 0.05,
):
    trials_df = study.trials_dataframe()
    trials_df_human_friendly = trials_df.rename(
        columns={
            "params_gnn_sizes": "Conv. Dim.",
            "params_n_convs": "No. Conv. Layers",
            "params_aggr": "Aggr.",
            "params_embed_reduction_mode": "Embed Reduction",
        }
    )
    anova_res = pg.anova(
        data=trials_df_human_friendly, dv=target_col, between=list(param_cols)
    )
    anova_res["Significance"] = anova_res["p-unc"].apply(
        lambda x: "Yes" if x < alpha else "No"
    )
    # for col in ["SS", "MS", "F"]:
    #     anova_res[col] = anova_res[col].apply(
    #         lambda x: f"{x:.1e}" if pd.notnull(x) else x
    #     )
    return anova_res


def _plot_and_save_figure(
    fig: plt.Figure | py.graph_objects.Figure,
    dataset_name: str,
    fig_name: str,
    title: str = None,
):
    if title:
        fig.update_layout(title=title)
    fig.show()
    fig_path = study_fig_path(dataset_name, fig_name)
    if isinstance(fig, plt.Figure):
        fig.savefig(fig_path)
    else:
        fig.write_image(fig_path)


def plot_importances(
    study,
    dataset_name: str,
    target: int = 1,
    quantile_val: float = 0.1,
    target_name: str = "Illicit F1",
):
    fig = opt.visualization.plot_param_importances(
        study,
        target=lambda t: t.values[target],
        evaluator=opt.importance.PedAnovaImportanceEvaluator(
            baseline_quantile=quantile_val
        ),
        target_name=target_name,
    )
    _plot_and_save_figure(
        fig,
        dataset_name,
        "hp_importance",
        f"Hyperparameter Importance for {dataset_name}",
    )


def plot_slice(
    study, dataset_name: str, target: int = 1, target_name: str = "Illicit F1"
):
    fig = opt.visualization.plot_slice(
        study,
        target=lambda t: t.values[target],
        target_name=target_name,
        params=["aggr", "n_convs", "gnn_sizes"],
    )
    _plot_and_save_figure(
        fig, dataset_name, "hp_slice", f"Hyperparameters Slice Plot for {dataset_name}"
    )


def plot_gnn_size_contours(
    study, dataset_name: str, target: int = 1, target_name: str = "Illicit F1"
):
    fig = opt.visualization.plot_contour(
        study,
        target=lambda t: t.values[target],
        target_name=target_name,
        params=["n_convs", "gnn_sizes"],
    )
    _plot_and_save_figure(
        fig,
        dataset_name,
        "hp_contour",
        f"GNN depth and size contour for {dataset_name}",
    )


def plot_param_interactions(
    study, dataset_name: str, target: int = 1, target_name: str = "Illicit F1"
):
    fig = opt.visualization.plot_contour(
        study,
        target=lambda t: t.values[target],
        target_name=target_name,
        params=["n_convs", "gnn_sizes", "aggr"],
    )
    _plot_and_save_figure(
        fig,
        dataset_name,
        "hp_interaction",
        f"Hyperparameters interaction for {dataset_name}",
    )


def plot_pareto_front(
    study, dataset_name: str, target: int = 1, target_name: str = "Illicit F1"
):
    fig = opt.visualization.plot_pareto_front(
        study,
        target_names=["complexity", target_name],
        targets=lambda t: (t.values[-1], t.values[target]),
    )
    _plot_and_save_figure(
        fig, dataset_name, "hp_pareto", f"Pareto Front for {dataset_name}"
    )


def plot_illicit_f1_vs_complexity(
    study,
    dataset_name,
    fig=None,
    ax=None,
    show: bool = True,
    save: bool = True,
    figsize=(10, 6),
):
    df = deepcopy(study.trials_dataframe())

    # Calculate complexity
    df["Complexity"] = df["params_n_convs"] * df["params_gnn_sizes"]

    # Normalize the metric for color mapping
    norm = plt.Normalize(df["values_1"].min(), df["values_1"].max())
    sm = plt.cm.ScalarMappable(cmap="RdYlGn", norm=norm)
    sm.set_array([])

    # Plot Illicit F1 vs Complexity
    if fig is None:
        fig, ax = plt.subplots(figsize=figsize)
    scatter = ax.scatter(
        df["Complexity"], df["values_1"], c=df["values_1"], cmap="RdYlGn", s=100
    )
    fig.colorbar(sm, ax=ax, label="Illicit F1")

    ax.set_title(f"Illicit F1 vs Complexity for {dataset_name}")
    ax.set_xlabel("Complexity (params_n_convs * params_gnn_sizes)")
    ax.set_ylabel("Illicit F1")
    ax.grid(True)
    plt.tight_layout()
    if show:
        plt.show()
    if save:
        _plot_and_save_figure(fig, dataset_name, "hp_complexity_vs_quality")


def plot_gnn_sizes_vs_n_convs(
    study,
    dataset_name,
    fig=None,
    ax=None,
    show: bool = True,
    save: bool = True,
    figsize=(10, 6),
):
    df = deepcopy(study.trials_dataframe())
    if fig is None:
        fig, ax = plt.subplots(figsize=figsize)
    sns.scatterplot(
        data=df,
        x="params_gnn_sizes",
        y="params_n_convs",
        hue="values_1",
        palette="viridis_r",
        ax=ax,
    )
    ax.set_xlabel("GNN Sizes")
    ax.set_ylabel("Number of Convolution Layers")
    ax.set_title(f"GNN Sizes vs No. of Conv. Layers for {dataset_name}")
    ax.legend(loc="center left", bbox_to_anchor=(1, 0.5), title="Illicit F1")
    ax.grid()
    plt.tight_layout()
    if show:
        plt.show()
    if save:
        _plot_and_save_figure(fig, dataset_name, "hp_nconv_vs_convsize_f1")


def plot_importance_barplot(
    studies_with_names: dict[str, opt.study.Study],
    figsize=(10, 6),
):
    # Get a consistent list of features
    all_features = set()
    for study in studies_with_names.values():
        importance_dict = get_feature_importance(study)
        all_features.update(importance_dict.keys())

    all_features = sorted(all_features)

    fig, axs = plt.subplots(2, 2, figsize=figsize)
    axs = axs.flatten()

    for i, (name, study) in enumerate(studies_with_names.items()):
        importance_dict = get_feature_importance(study)
        importance_df = pd.DataFrame(
            list(importance_dict.items()), columns=["Feature", "Importance"]
        )

        # Ensure all features are present in the dataframe
        importance_df = (
            importance_df.set_index("Feature")
            .reindex(all_features)
            .fillna(0)
            .reset_index()
        )

        sns.barplot(x="Importance", y="Feature", data=importance_df, ax=axs[i])
        axs[i].set_title(f"{name}")
        if i == 0:
            axs[i].set_ylabel("Feature")
        else:
            axs[i].set_ylabel("'")
        if i > 1:
            axs[i].set_xlabel("Importance")
        else:
            axs[i].set_xlabel("'")

    plt.suptitle("Hyperparameter importance across datasets")
    plt.tight_layout()
    plt.show()


def plot_all_for_study(
    study, dataset_name: str, target: int = 1, target_name: str = "Illicit F1"
):
    plot_importances(study, dataset_name, target=target, target_name=target_name)
    plot_slice(study, dataset_name, target=target, target_name=target_name)
    plot_gnn_size_contours(study, dataset_name, target=target, target_name=target_name)
    plot_param_interactions(study, dataset_name, target=target, target_name=target_name)
    # plot_pareto_front(study, dataset_name, target=target, target_name=target_name)
    plot_illicit_f1_vs_complexity(study, dataset_name)
    plot_gnn_sizes_vs_n_convs(study, dataset_name)


def build_and_save_study_analysis(study, name: str):
    trials_df = study.trials_dataframe()
    display(Markdown(f"### {name} studies analysis"))
    display(trials_df.head(3))
    display(Markdown("#### Feature importances"))
    importances = get_feature_importance(study)
    display(importances)
    anova_res = anova_check_importance(study)
    display(anova_res.round(4))
    display(Markdown("#### Plots"))
    plot_all_for_study(study, name)


def make_f1_vs_complexity_joint_plot(
    studies, fig=None, ax=None, show: bool = True, save: bool = True, figsize=(10, 6)
):
    if fig is None:
        fig, ax = plt.subplots(2, 2, figsize=figsize)

    for i, (dataset_name, study) in enumerate(studies.items()):
        df = deepcopy(study.trials_dataframe())

        # Calculate complexity
        df["Complexity"] = df["params_n_convs"] * df["params_gnn_sizes"]

        # Normalize the metric for color mapping
        norm = plt.Normalize(df["values_1"].min(), df["values_1"].max())
        sm = plt.cm.ScalarMappable(cmap="RdYlGn", norm=norm)
        sm.set_array([])

        # Plot Illicit F1 vs Complexity
        scatter = ax[i // 2, i % 2].scatter(
            df["Complexity"],
            df["values_1"],
            c=df["values_1"],
            cmap="RdYlGn",
            s=100,
            label=dataset_name,
        )
        ax[i // 2, i % 2].set_title(f"Illicit F1 vs Complexity for {dataset_name}")
        ax[i // 2, i % 2].set_xlabel("Complexity (N convs. * Conv. size)")
        ax[i // 2, i % 2].set_ylabel("Illicit F1")
        ax[i // 2, i % 2].grid(True)

    # Add color legend only once
    cbar = fig.colorbar(
        sm, ax=ax.ravel().tolist(), label="Illicit F1", location="right"
    )
    plt.suptitle("Complexity vs Illicit F1 across datasets")
    plt.tight_layout()
    fig.subplots_adjust(right=0.85)
    cbar.ax.set_position([0.87, 0.15, 0.03, 0.7])

    if show:
        plt.show()
    if save:
        fig.savefig(IMAGES_DIR / "complexity_vs_f1_joint.pdf")


def clean_torch_model_summary(model_summary: str) -> str:
    """Returns markdown-friendly Torch model summary. Removes some characters
    that are not supported in markdown.

    Parameters
    ----------
    model_summary : str
        String summary of the model.

    Returns
    -------
    str
        Cleaned model summary.
    """
    return "\n".join(model_summary.split("\n")[1:-1]).replace("+", "|")
