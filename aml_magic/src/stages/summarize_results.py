"""This module contains functions for summarizing the results of the current study and comparing them with the results of the previous paper.
It produces a summary table, visualizations, and saves the results in a structured format."""

import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import autoroot  # noqa
import seaborn as sns
import numpy as np
import pandas as pd
import typer
import matplotlib.pyplot as plt
from pandas.io.formats.style import Styler

from aml_magic.src.consts import (
    PREV_PAPER_RESULTS_PATH,
    STUDY_RESULTS_PATH,
    STUDY_COMPARISON_PATH,
)


def _dataset_sort_key(ds_name: str) -> int:
    """Utility function to sort datasets by their number.

    Parameters
    ----------
    ds_name : str
        Name of the dataset eg. AMLSim 31

    Returns
    -------
    int
        Numerical part of the dataset name.
    """
    return int(ds_name.split(" ")[-1])


@dataclass
class Score:
    """A helper object for storing scores."""

    model: str
    dataset: str
    metric: str
    mu: float
    std: float
    score_hi: float
    score_lo: float


def parse_lines_prev_paper(
    results_lines: List[str], model: str, dataset: str
) -> List[Score]:
    """A utility function that parsers the results of the previous paper and returns a list of Score objects.
    Previous paper Authors reported scores in a text file, where each line is in the following format:
        `metric name: mean +- std`
    For example:
        `roc_auc: 0.9 +- 0.01`

    Parameters
    ----------
    results_lines : List[str]
        File lines containing the results of the previous paper.
    model : str
        Name of the model
    dataset : str
        Name of the dataset

    Returns
    -------
    List[Score]
        List of scores with standardized field names.
    """
    metric_vals = []
    for line in results_lines:
        metric, vals = line.split(":")
        mu, std = vals.split("+-")
        mu = float(mu)
        std = float(std)
        score_up = min(1.0, mu + 1.6 * std)
        score_down = max(0.0, mu - 1.6 * std)
        metric_vals.append(Score(model, dataset, metric, mu, std, score_up, score_down))
    return metric_vals


def gather_scores_prev_paper(prev_scores_path: Path) -> pd.DataFrame:
    """A utility function that reads and stores results from a previous paper.
    Authors reported their results as text files in a single folder.
    Each file has the following naming convention:
        `datasetname_modelname.txt`
    For example:
        `amlsim_51_gcn.txt".
    This function iterates the folder, reads each file, and parses the results into a DataFrame.

    Parameters
    ----------
    prev_scores_path : Path
        Path to prev scores.

    Returns
    -------
    pd.DataFrame
        Data frame with standardized scores in a common format.
    """
    model_scores = []
    for f in os.listdir(prev_scores_path):
        if f.endswith(".txt"):
            dataset = "_".join(f.split("_", 2)[:2])
            model = f.split("_", 2)[-1].split(".")[0]

            results_f = open(prev_scores_path / f, encoding="UTF-8").readlines()
            model_scores.extend(parse_lines_prev_paper(results_f, model, dataset))
    scores_df = pd.DataFrame(model_scores)
    return scores_df


def gather_current_study_scores_and_summary(
    study_scores_path: Path,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """A utility function that reads and stores results from the current study.
    Current study reports scores in a separate folder for each dataset, with a dataframe
    of the following structure:
     1. Header: metric names;
     2. First row: index=score, values=mean scores per each metric
     3. Second row: index=std, values=std scores per each metric.
     For example:
        ```
        element,roc_auc,pr_auc
        score,0.9,0.8
        std,0.01,0.02
        ```

    And a second DataFrame with raw scores only.

    This function iterates the folder, reads each file, and parses the results into a DataFrame with
    standardized columns per each metric: mu, std, score_hi, score_lo.

    Parameters
    ----------
    study_scores_path : Path
        Path to current study results.

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame]
        A tuple with:
        1. Data Frame with standardized scores and their respective high/low values.
        2. Data Frame with raw scores.
    """
    new_model_scores_summary = []
    new_model_raw_scores = []
    for ds in os.listdir(study_scores_path):
        for f in os.listdir(study_scores_path / ds):
            if f.endswith(".csv") and "raw" not in f:
                scores = pd.read_csv(study_scores_path / ds / f).rename(
                    columns={"Unnamed: 0": "element"}
                )
                mu_scores = scores.set_index("element").loc["score"].iloc[:-1]
                std_scores = scores.set_index("element").loc["std"].iloc[:-1]

                score_up = np.minimum(mu_scores + 1.6 * std_scores, 1.0)
                score_low = np.maximum(0.0, mu_scores - 1.6 * std_scores)

                scores_standardized = (
                    pd.DataFrame(
                        {
                            "mu": mu_scores,
                            "std": std_scores,
                            "score_hi": score_up,
                            "score_lo": score_low,
                        }
                    )
                    .reset_index(names="metric")
                    .assign(model=scores.model.unique()[0], dataset=ds)
                )
                new_model_scores_summary.append(scores_standardized)
            elif f.endswith("raw.csv"):
                raw_scores = pd.read_csv(study_scores_path / ds / f, index_col=0)
                raw_scores["dataset"] = ds
                new_model_raw_scores.append(raw_scores)
    scores_summary = pd.concat(new_model_scores_summary)
    raw_scores = pd.concat(new_model_raw_scores)

    return scores_summary, raw_scores


def __make_plus_minus_col(data: pd.DataFrame) -> pd.DataFrame:
    """A helper function that fuilds a text column with +/- signs for the mean and std scores.
    It is needed for TeX tables.

    Parameters
    ----------
    data : pd.DataFrame
        DataFrame with model scores.

    Returns
    -------
    pd.DataFrame
        DataFrame with the additional column mu +/- std
    """
    res = data.apply(lambda row: f"{row['mu']:.3f} +/- {row['std']:.3f}", axis=1)
    return data.assign(score=res)


def build_styled_summary_table(
    all_scores: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Reformats table with scores to be saved in three formats:
    1. CSV pivot
    2. Excel pivot
    3. Tex tables per dataset

    Parameters
    ----------
    all_scores : pd.DataFrame
        DataFrame with all scores.

    Returns
    -------
    Tuple[Styler, dict]
        A tuple of:
            1. Pivot table with scores;
            1. Formatted pivot table with scores (for Excel);
            2. Dict with tex table per dataset;
    """
    all_scores_for_report = all_scores.copy()
    all_scores_for_report[["scope", "metric"]] = all_scores_for_report[
        "metric"
    ].str.split(" ", n=1, expand=True)
    all_scores_for_report["model"] = all_scores_for_report["model"].str.replace(
        "_", " "
    )
    all_scores_for_report["dataset"] = (
        all_scores_for_report["dataset"]
        .str.replace("aml", "AML")
        .str.replace("sim", "Sim")
    )
    all_scores_for_report = (
        all_scores_for_report.groupby(["dataset", "model", "metric", "scope"])[
            ["mu", "std"]
        ]
        .apply(__make_plus_minus_col)
        .reset_index()
        .reset_index()
    )
    models_original_order = list(all_scores.model.unique())

    scores_pivot = all_scores_for_report.pivot_table(
        index=["model"],
        columns=["dataset", "metric", "scope"],
        values="score",
        aggfunc="first",
    )
    scores_pivot = scores_pivot.loc[models_original_order]
    scores_pivot_styled = scores_pivot.style.highlight_max(color="lightgreen", axis=0)

    tex_tables_per_dataset = {}
    for dataset in all_scores.dataset.unique():
        tex_table = scores_pivot.loc[models_original_order][
            dataset
        ].style.highlight_max(axis=0, props="font-weight: bold")
        tex_out = (
            tex_table.to_latex(convert_css=True, column_format="|c|cc|cc|cc|")
            .replace("+/-", "$\pm$")
            .replace("metric", "")
            .replace("scope", "")
            .replace("model", "")
            .replace("{r}", "{c|}")
            .replace("&  &  &  &  &  &  \\\\", "\hline &  &  &  &  &  &  \\\\")
        )
        tex_tables_per_dataset[dataset] = tex_out
    return scores_pivot, scores_pivot_styled, tex_tables_per_dataset


def combine_scores_and_build_analysis(
    prev_scores_df: pd.DataFrame,
    current_scores_summary_df: pd.DataFrame,
    current_raw_scores_df: pd.DataFrame,
    scores_comparison_path: Path,
    save_figures: bool = True,
):
    """Main function for combining the scores from the previous paper and the current study and analyzing
    them by building a summary table, visualizations, and saving the results in a structured format.

    Parameters
    ----------
    prev_scores_df : pd.DataFrame
        Scores from the previous paper.
    current_scores_df : pd.DataFrame
        Scores from the current study.
    current_raw_scores_df : pd.DataFrame
        Raw scores from the current study.
    scores_comparison_path : Path
        Path to save the results.
    save_figures: bool
        Whether to save the figures or not.
    """
    all_scores = pd.concat(
        [current_scores_summary_df, prev_scores_df], ignore_index=True
    )
    all_scores = improve_scores_readability(all_scores.copy())
    # Sort values according to the name of the dataset: 31, 51, etc.
    all_scores.sort_values(
        by="dataset", key=lambda series: series.apply(_dataset_sort_key), inplace=True
    )
    scores_comparison_path.mkdir(parents=True, exist_ok=True)
    scores_summary, styled_summary, tex_tables_per_dataset = build_styled_summary_table(
        all_scores
    )

    save_score_tables(
        scores_summary,
        styled_summary,
        tex_tables_per_dataset,
        scores_comparison_path,
    )
    build_scores_visualization(
        prev_scores_df, current_raw_scores_df, scores_comparison_path, save_figures
    )


def improve_scores_readability(all_scores: pd.DataFrame) -> pd.DataFrame:
    """Renames columns and values in the scores DataFrame for better readability.

    Parameters
    ----------
    all_scores : _type_
        DataFrame with scores.

    Returns
    -------
    pd.DataFrame
        DataFrame with scores renamed.
    """
    all_scores["dataset"] = (
        all_scores["dataset"].str.replace("aml", "AML").str.replace("_", " ")
    ).str.replace("sim", "Sim")
    all_scores["model"] = (
        all_scores["model"]
        .str.replace("_", "+")
        .str.title()
        .str.replace("Xgboost", "XGB", case=False)
        .str.replace("gcn", "GCN", case=False)
        .str.replace("Magic", "MAGIC", case=False)
    )
    return all_scores


def save_score_tables(
    scores_summary: pd.DataFrame,
    styled_summary: Styler,
    tex_tables_per_dataset: dict,
    scores_comparison_path: Path,
):
    """Saves scores in a structured format: csv, tex, and excel.
    They are useful for further analysis and reporting, as well as exporting
    the results to the paper.

    Parameters
    ----------
    scores_summary : pd.DataFrame
        Summary table with scores.
    styled_summary : Styler
        Stylized summary table, intented to be exported to latex or Excel.
    tex_tables_per_dataset : dict
        Dictionary with tex tables per dataset.
    scores_comparison_path : Path
        Path to save the results.
    """
    scores_summary.T.to_csv(scores_comparison_path / "scores_summary.csv")
    styled_summary.to_excel(scores_comparison_path / "scores_summary.xlsx")
    for dataset, tex_table in tex_tables_per_dataset.items():
        with open(scores_comparison_path / f"{dataset}_scores.tex", "w") as f:
            f.write(tex_table)


def build_scores_visualization(
    prev_scores_df: pd.DataFrame,
    current_raw_scores_df: pd.DataFrame,
    scores_comparison_path: Path,
    save_figures: bool = True,
):
    """Builds visualizations for the scores, comparing them across metrics and datasets.
    All visualizations are boxplots.

    Parameters
    ----------
    prev_scores_df : pd.DataFrame
        Scores from the previous paper
    current_raw_scores_df : pd.DataFrame
        Raw scores from the current study.
    scores_comparison_path : Path
        Path to save the results.
    """
    new_scores_melted = current_raw_scores_df.melt(id_vars=["model", "dataset"]).rename(
        columns={"variable": "metric"}
    )
    prev_scores_melted = prev_scores_df.melt(
        id_vars=["model", "dataset", "metric"],
        value_vars=["mu", "score_hi", "score_lo"],
    ).drop(columns="variable")
    all_scores_melted = pd.concat(
        [new_scores_melted, prev_scores_melted], ignore_index=True
    )
    all_scores_melted = improve_scores_readability(all_scores_melted)

    if save_figures:
        figures_out_path = scores_comparison_path / "figures"
        figures_out_path.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", font_scale=1.8)
    for idx, metric in enumerate(all_scores_melted.metric.unique()):
        if "Macro" in metric:
            legend = "brief"
            plt.rc("legend", fontsize=20, title_fontsize=20)
        else:
            legend = None
        g = sns.catplot(
            all_scores_melted.loc[all_scores_melted.metric == metric],
            x="model",
            y="value",
            col="dataset",
            hue="model",
            col_wrap=2,
            sharex=True,
            sharey=False,
            kind="box",
            legend=legend,
            legend_out=True,
            margin_titles=True,
            row_order=all_scores_melted.model.unique(),
            col_order=sorted(all_scores_melted.dataset.unique(), key=_dataset_sort_key),
        )
        g.axes[-1].set_xticklabels([])
        g.axes[-1].set_xlabel("")
        g.axes[-2].set_xticklabels([])
        g.axes[-2].set_xlabel("")
        if "Macro" in metric:
            g.axes[0].set_ylabel("")
            g.axes[2].set_ylabel("")
        g.figure.subplots_adjust(top=0.91)
        g.figure.suptitle(f"Metric {metric}")
        if save_figures:
            g.savefig(figures_out_path / f"{metric}_boxplot.pdf")
        else:
            plt.show()


def main(
    prev_scores_path: Path = PREV_PAPER_RESULTS_PATH,
    study_scores_path: Path = STUDY_RESULTS_PATH,
    comparison_path: Path = STUDY_COMPARISON_PATH,
):
    prev_scores_df = gather_scores_prev_paper(prev_scores_path)
    current_scores_summary, current_raw_scores = (
        gather_current_study_scores_and_summary(study_scores_path)
    )

    combine_scores_and_build_analysis(
        prev_scores_df, current_scores_summary, current_raw_scores, comparison_path
    )


if __name__ == "__main__":
    typer.run(main)
