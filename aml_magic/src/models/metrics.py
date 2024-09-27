"""A helper module for calculating metrics and confidence intervals during the experimentation."""

import scipy
import os
import pandas as pd
import numpy as np
import mlflow as mlf
from typing import List, Tuple, Dict
from pathlib import Path


def get_confidence_intervals(
    metric_list: List[float], n_repeats: int
) -> Tuple[float, float]:
    """Function to calculate the confidence intervals with a 95%.
    This is the original implementation, used in the Silva et. al paper:
    https://github.com/italodellagarza/SBSITests

    Parameters
    ----------
    metric_list : List[float]
        List containing the metrics obtained.
    n_repeats : int
        Number of experiment repetitions.

    Returns
    -------
    Tuple[float, float]
        (metric average, confidence interval length)
    """
    confidence = 0.95
    t_value = scipy.stats.t.ppf((1 + confidence) / 2.0, df=n_repeats - 1)
    metric_avg = np.mean(metric_list)

    se = 0.0
    for m in metric_list:
        se += (m - metric_avg) ** 2
    se = np.sqrt((1.0 / (n_repeats - 1)) * se)
    ci_length = t_value * se

    return metric_avg, ci_length


def calculate_results_ci(
    metric_scores: Dict[str, List[float]],
    model_name: str,
    n_repeats: int,
) -> Dict[str, Dict[str, float]]:
    """Function for calculating and storing the confidence intervals for the metrics.

    Parameters
    ----------
    metric_scores : Dict[str, List[float]]
        Dictionary containing the metrics and their corresponding scores obtained
        during subsequent experiments.
    model_name : str
        Name of the model in question
    n_repeats : int
        Number of experimental repeats

    Returns
    -------
    Dict[str, Dict[str, float]]
        Dictionary containing the metrics, their scores and confidence intervals.
    """
    results = {}
    for metric, scores in metric_scores.items():
        metric_avg, ci_length = get_confidence_intervals(scores, n_repeats)
        results[metric] = {
            "score": metric_avg,
            "std": ci_length,
        }
    results["model"] = model_name
    return results


def save_results_summary(
    metric_scores: dict[str, List[float]],
    experiment_name: str,
    model_name: str,
    n_repeats,
    output_path: Path,
):
    """Saves metric summaries to a file and logs them to MLFlow.

    Parameters
    ----------
    metric_scores : dict[str, List[float]]
        Dictionary containing the metrics and their corresponding scores obtained
        during subsequent experiments.
    experiment_name : str
        Name of the experiment.
    model_name : str
        Model name.
    n_repeats : _type_
        Number of experimental repeats.
    output_path : Path
        Path to the output directory.
    """
    raw_scores = pd.DataFrame(metric_scores).assign(model=model_name)
    results = calculate_results_ci(metric_scores, model_name, n_repeats)
    mlf.set_experiment(experiment_name=experiment_name)
    with mlf.start_run(run_name="CI SUMMARY ") as run:
        for metric, values in results.items():
            if metric != "model":
                mlf.log_metric(metric, values["score"])
                mlf.log_metric(f"{metric}_std", values["std"])

    out_file_path = output_path / experiment_name
    out_file_path.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(results)
    df.to_csv(
        out_file_path / f"{experiment_name}_{model_name}.csv",
        index=True,
    )
    raw_scores.to_csv(
        out_file_path / f"{experiment_name}_{model_name}_raw.csv",
        index=True,
    )


def read_results_for_dataset(dataset_name: str) -> pd.DataFrame:
    """A helper method for parsing scores from the previous study Silva et. al.
    Scores there were reported as txt files with mean +/- std. The file name
    indicates the model + dataset name.
    This function reads and checks all the files related to a particular dataset,
    so various models can be compared.

    Parameters
    ----------
    dataset_name : str
        Name of the dataset

    Returns
    -------
    pd.DataFrame
        Scores from the previous study parsed as the DataFrame.
    """
    all_model_results = []
    for root, _, files in os.walk("./results"):
        for f in files:
            if f.endswith(".txt") and dataset_name in f:
                result_file_path = os.path.join(root, f)
                all_model_results.append(
                    parse_result_file(result_file_path, dataset_name)
                )
    all_model_reuslts_df = pd.concat(all_model_results).pivot_table(
        index=["model"], columns=["metric"], values=["score", "std"], aggfunc="mean"
    )
    return all_model_reuslts_df


def parse_result_file(result_file_path: str, dataset_name: str) -> pd.DataFrame:
    """A helper method for parsing a single file from the previous study Silva et. al.
    Scores there were reported as txt files with mean +/- std.
    This function parses the file and returns the scores as a DataFrame.

    Parameters
    ----------
    result_file_path : str
        Path to the file.
    dataset_name : str
        Name of the dataset

    Returns
    -------
    pd.DataFrame
        A DataFrame parsed from the file.
    """
    lines = open(result_file_path, "r").readlines()
    file_name = os.path.basename(result_file_path).replace(".txt", "")
    model_name = file_name.replace(dataset_name, "").strip("_")

    scores = []
    for line in lines:
        metric, values = line.split(":")
        score, std = values.split("+-")
        score_val = float(score.strip())
        score_std = float(std.strip())
        scores.append(
            {
                "metric": metric,
                "score": score_val,
                "std": score_std,
                "model": model_name,
            }
        )
    scores_df = pd.DataFrame(scores)
    return scores_df


def build_results_comparison_for_dataset(dataset_name: str):
    """Builds a summary dataset and saves it into three formats: csv, xlsx, html.

    Parameters
    ----------
    dataset_name : str
        Name of the dataset.
    """
    results_df = read_results_for_dataset(dataset_name)
    results_df = results_df.reset_index()
    results_df.to_csv("./summaries/{}.csv".format(dataset_name), index=False)
    results_df.to_excel("./summaries/{}.xlsx".format(dataset_name))
    results_df.style.highlight_max(color="green", axis=0).to_html(
        "./summaries/{}.html".format(dataset_name)
    )
