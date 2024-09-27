"""A main module for performing the stage for training and testing models."""

import autoroot  # noqa
import typer
import joblib
import torch_geometric.data as pyg
import torch_geometric.loader as pyg_loader
from tqdm.auto import tqdm
from typing import Dict
from pathlib import Path
import aml_magic.src.consts as cc
import aml_magic.src.utils.configs as cfg
import aml_magic.src.models.training as training
import aml_magic.src.models.metrics as metrics


def train_and_test_repeat(
    experiment_config: cfg.ExperimentConfig,
    experiment_name: str,
    training_dataset: pyg.Data,
    test_dataset: pyg.Data,
    model_arch_cfg: cfg.GNNArchConfigs,
    model_train_cfg: cfg.DatasetTrainConfig,
    gb_train_cfg: cfg.GradientBoostingConfigs,
    train_loader: pyg_loader.DataLoader,
    test_loader: pyg_loader.DataLoader,
    repeat: int,
) -> Dict[str, float]:
    """A function for training and testing a single repeat of the experiment.

    Parameters
    ----------
    experiment_config : cfg.ExperimentConfig
        Experiment configuration.
    experiment_name : str
        Name of the experiment.
    training_dataset : pyg.Data
        Dataset for training.
    test_dataset : pyg.Data
        Dataset for testing.
    model_arch_cfg : cfg.GNNArchConfigs
        Architecture configuration for the GNN model.
    model_train_cfg : cfg.DatasetTrainConfig
        Model training configuration.
    gb_train_cfg : cfg.GradientBoostingConfigs
        Gradient boosting configuration.
    train_loader : pyg_loader.DataLoader
        Training data loader.
    test_loader : pyg_loader.DataLoader
        Test data loader.
    repeat : int
        Repeat of the experiment.

    Returns
    -------
    Dict[str, float]
        Dictionary of results
    """
    gnn_model = training.train_aml_magic(
        repeat,
        experiment_name,
        f"{experiment_name}_GNN_cv{repeat}",
        model_arch_cfg,
        test_loader,
        train_loader,
        model_train_cfg,
    )
    data_for_gb = training.prepare_embedding_for_gradient_boosting(
        gnn_model, training_dataset, test_dataset
    )
    gb_results = training.train_gb(
        datas_for_gb=data_for_gb,
        experiment_name=experiment_name,
        run_name=f"{experiment_name}_{experiment_config.gradient_boosting_impl}_cv{repeat}",
        config_for_gb=gb_train_cfg,
    )

    return {
        "Macro Precision": gb_results.prec_macro,
        "Macro Recall": gb_results.rec_macro,
        "Macro F1": gb_results.f1_macro,
        "Ilicit Precision": gb_results.prec_macro,
        "Ilicit Recall": gb_results.rec_illicit,
        "Ilicit F1": gb_results.f1_illicit,
    }


def process_dataset(
    ds: str,
    experiment_config: cfg.ExperimentConfig,
    training_configs: cfg.GNNConfig,
    data_configs: cfg.DatasetConfig,
    gradient_boosting_configs: cfg.GradientBoostingConfigs,
):
    """A function for processing a single dataset - it runs multiple repeats of the same
    experiment and saves the results.

    Parameters
    ----------
    ds : str
        Dataset to be processed.
    experiment_config : cfg.ExperimentConfig
        Experiment configuration.
    training_configs : cfg.GNNConfig
        Configuration for the GNN model.
    data_configs : cfg.DatasetConfig
        Data configuration.
    gradient_boosting_configs : cfg.GradientBoostingConfigs
        Gradient boosting configuration.
    """
    # Configure the experiment for the dataset
    experiment_name = f"{ds}_crossval"

    dataset_path = Path(data_configs.output_path, ds)
    training_dataset = joblib.load(dataset_path / "train_graphs.pkl")
    test_dataset = joblib.load(dataset_path / "test_graphs.pkl")

    model_arch_cfg = training_configs.get_achitecture_for_dataset(ds)
    model_train_cfg = training_configs.get_training_config_for_dataset(ds)
    gb_train_cfg = gradient_boosting_configs.get_config_for_gb_impl(
        experiment_config.gradient_boosting_impl
    )
    gb_train_cfg_ds = gb_train_cfg.get_training_config_for_dataset(ds)

    train_loader = pyg_loader.DataLoader(
        training_dataset, batch_size=model_train_cfg.batch_size, shuffle=False
    )
    test_loader = pyg_loader.DataLoader(
        test_dataset, shuffle=False, batch_size=len(test_dataset)
    )

    metric_scores = {
        "Macro Precision": [],
        "Macro Recall": [],
        "Macro F1": [],
        "Ilicit Precision": [],
        "Ilicit Recall": [],
        "Ilicit F1": [],
    }

    for repeat in tqdm(range(experiment_config.n_repeats)):
        metric_res = train_and_test_repeat(
            experiment_config,
            experiment_name,
            training_dataset,
            test_dataset,
            model_arch_cfg,
            model_train_cfg,
            gb_train_cfg_ds,
            train_loader,
            test_loader,
            repeat,
        )
        for metric, val in metric_res.items():
            metric_scores[metric].append(val)

    results_path = Path(cc.RESULTS_DIR, "study")
    results_path.mkdir(parents=True, exist_ok=True)

    metrics.save_results_summary(
        metric_scores,
        f"{ds}_CI_SUMMARY",
        f"MAGIC+{experiment_config.gradient_boosting_impl}",
        experiment_config.n_repeats,
        results_path,
    )


def main():
    # Prepare the configuration
    training_params = cfg.load_params(cc.STAGE_GNN_TRAINING, cc.STAGE_PARAMS_DIR)
    data_params = cfg.load_params(cc.STAGE_PREPARE_DATA, cc.STAGE_PARAMS_DIR)
    gb_params = cfg.load_params(cc.STAGE_GB_TRAINING, cc.STAGE_PARAMS_DIR)
    experiment_params = cfg.load_params(cc.MAIN_PARAMS_FILE, key_name=None)

    experiment_config = cfg.ExperimentConfig(**experiment_params)
    training_configs = cfg.GNNConfig(**training_params)
    data_configs = cfg.DatasetsConfig(**data_params)
    gradient_boosting_configs = cfg.GradientBoostingConfigs(**gb_params)

    # Process each dataset
    for ds in tqdm(experiment_config.datasets_to_train):
        process_dataset(
            ds,
            experiment_config,
            training_configs,
            data_configs,
            gradient_boosting_configs,
        )


if __name__ == "__main__":
    typer.run(main)
