import autoroot  # noqa
import autorootcwd  # noqa
import torch as th
import typer
import torch_geometric as pyg
from tqdm.auto import tqdm
import joblib
from sklearn.metrics import f1_score
import aml_magic.src.utils.configs as cfg
import aml_magic.src.consts as cc
import aml_magic.src.models.training as training
import optuna as opt
from dataclasses import dataclass
from pathlib import Path
from copy import deepcopy

PARAMS_GRID = {
    "n_convs": [1, 2, 3, 6],
    "gnn_sizes": [8, 16, 32, 64],
    "embed_reduction_mode": ["concat", "mean"],
    "aggr": ["('add')", "('softmax')", "('add', 'min', 'max')"],
}


@dataclass
class HpConfig:
    n_trials: int = 50
    max_epochs: int = 20
    batch_size: int = 256
    tuning_out_path: str = "data/hp_results/"


@dataclass
class Configs:
    """Helper class that stores all the configurations for the experiment"""

    experiment_config: cfg.ExperimentConfig
    training_configs: cfg.GNNConfig
    data_configs: cfg.DatasetsConfig
    gradient_boosting_configs: cfg.GradientBoostingConfigs

    gnn_training_cfg_4_ds: cfg.DatasetTrainConfig
    gb_training_cfg_4_ds: cfg.LightGBMConfig

    dsname: str


@dataclass
class DataContainer:
    """Helper class that stores all the data for the experiment"""

    train_loader: pyg.loader.DataLoader
    val_loader: pyg.loader.DataLoader
    train_dataset: pyg.data.Dataset
    val_dataset: pyg.data.Dataset


def prepare_configs(dsname: str) -> Configs:
    training_params = cfg.load_params(cc.STAGE_GNN_TRAINING, cc.STAGE_PARAMS_DIR)
    data_params = cfg.load_params(cc.STAGE_PREPARE_DATA, cc.STAGE_PARAMS_DIR)
    gb_params = cfg.load_params(cc.STAGE_GB_TRAINING, cc.STAGE_PARAMS_DIR)
    experiment_params = cfg.load_params(cc.MAIN_PARAMS_FILE, key_name=None)

    experiment_config = cfg.ExperimentConfig(**experiment_params)
    training_configs = cfg.GNNConfig(**training_params)
    data_configs = cfg.DatasetsConfig(**data_params)
    gradient_boosting_configs = cfg.GradientBoostingConfigs(**gb_params)

    training_conf = training_configs.get_training_config_for_dataset(dsname)
    gb_training_conf = gradient_boosting_configs.get_config_for_gb_impl(
        experiment_config.gradient_boosting_impl
    )
    gb_train_cfg_ds = gb_training_conf.get_training_config_for_dataset(dsname)

    return Configs(
        experiment_config=experiment_config,
        training_configs=training_configs,
        data_configs=data_configs,
        gradient_boosting_configs=gradient_boosting_configs,
        gnn_training_cfg_4_ds=training_conf,
        gb_training_cfg_4_ds=gb_train_cfg_ds,
        dsname=dsname,
    )


def prepare_data_for_dataset(
    configs: Configs, train_size: float = 0.75
) -> DataContainer:
    dataset_path = Path(configs.data_configs.output_path, configs.dsname)
    training_dataset = joblib.load(dataset_path / "train_graphs.pkl")

    ntrain = int(len(training_dataset) * train_size)
    train_dataset = training_dataset[:ntrain]
    val_dataset = training_dataset[ntrain:]

    train_loader = pyg.loader.DataLoader(
        train_dataset,
        batch_size=configs.gnn_training_cfg_4_ds.batch_size,
        shuffle=False,
    )
    val_loader = pyg.loader.DataLoader(
        val_dataset, batch_size=configs.gnn_training_cfg_4_ds.batch_size, shuffle=False
    )

    return DataContainer(
        train_loader=train_loader,
        val_loader=val_loader,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
    )


def objective(trial, configs: Configs, data_container: DataContainer):
    params = {
        "n_convs": trial.suggest_categorical("n_convs", PARAMS_GRID["n_convs"]),
        "gnn_sizes": trial.suggest_categorical("gnn_sizes", PARAMS_GRID["gnn_sizes"]),
        "embed_reduction_mode": trial.suggest_categorical(
            "embed_reduction_mode", PARAMS_GRID["embed_reduction_mode"]
        ),
        "aggr": eval(trial.suggest_categorical("aggr", PARAMS_GRID["aggr"])),
    }
    base_arch = configs.training_configs.get_achitecture_for_dataset(configs.dsname)
    arch = deepcopy(base_arch)
    arch.conv_sizes = tuple([params["gnn_sizes"]] * params["n_convs"])
    arch.embed_reduction_mode = params["embed_reduction_mode"]
    arch.aggr = params["aggr"]
    gnn_model = training.train_aml_magic(
        1,
        f"tuning_{configs.dsname}",
        f"tune_{configs.dsname}",
        arch,
        data_container.val_loader,
        data_container.train_loader,
        configs.gnn_training_cfg_4_ds,
        mlflow_logging=False,
    )
    gnn_model.eval()
    yhats = []
    ys = []
    with th.no_grad():
        for batch in data_container.val_loader:
            yhat = gnn_model(batch).numpy().argmax(-1)
            y = batch.y.numpy()
            yhats.extend(yhat)
            ys.extend(y)
    f1_macro = f1_score(y_pred=yhats, y_true=ys, average="macro")
    f1_illicit = f1_score(y_pred=yhats, y_true=ys, average="binary", labels=[0])
    model_size = params["gnn_sizes"] * params["n_convs"]
    return [f1_macro, f1_illicit, model_size]


def run_optimization_study(
    configs: Configs, data_container: DataContainer, hp_config: HpConfig = HpConfig()
):
    configs.gnn_training_cfg_4_ds.batch_size = hp_config.batch_size
    configs.gnn_training_cfg_4_ds.max_epochs = hp_config.max_epochs

    study = opt.create_study(
        directions=["maximize", "maximize", "minimize"],
        sampler=opt.samplers.TPESampler(seed=42),
    )
    study.optimize(
        lambda trial: objective(trial, configs, data_container),
        n_trials=hp_config.n_trials,
    )
    return study


def main():
    params = cfg.load_params(cc.STAGE_HP_TUNE, cc.STAGE_PARAMS_DIR)
    hp_config = HpConfig(
        max_epochs=params["max_epochs"],
        batch_size=params["batch_size"],
        n_trials=params["n_trials"],
        tuning_out_path=params["tuning_out_path"],
    )
    datasets = params["tuning_datasets"]
    for ds in tqdm(datasets):
        configs = prepare_configs(ds)
        data_container = prepare_data_for_dataset(configs)
        study = run_optimization_study(configs, data_container, hp_config)
        result_path = Path(hp_config.tuning_out_path)
        result_path.mkdir(parents=True, exist_ok=True)
        joblib.dump(study, result_path / f"{ds}_study.pkl")


if __name__ == "__main__":
    typer.run(main)
