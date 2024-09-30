"""Module that contains model-specific training functions."""

import autoroot  # noqa
import torch
import numpy as np
import torch_geometric as pyg
import pytorch_lightning as pl
import torch_geometric.data as pyg_data
import mlflow as mlf
from dataclasses import dataclass
from sklearn.metrics import precision_score, recall_score, f1_score

import aml_magic.src.utils.configs as cfg
from aml_magic.src.models.magic import MAGICPl


@dataclass
class DataForGB:
    """Helper class that stores data for training Gradient Boosting"""

    X_test: np.ndarray
    X_train: np.ndarray
    y_test: np.ndarray
    y_train: np.ndarray


@dataclass
class GBTrainingResults:
    """Helper class for storing Gradient Boosting training results"""

    model: object
    prec_macro: float
    rec_macro: float
    f1_macro: float
    prec_illicit: float
    rec_illicit: float
    f1_illicit: float


def train_aml_magic(
    execution: int,
    experiment_name: str,
    run_name: str,
    model_config: cfg.GNNArchConfigs,
    test_loader: pyg.data.DataLoader,
    train_loader: pyg.data.DataLoader,
    training_config: cfg.DatasetTrainConfig,
    mlflow_tracking_uri: str,
) -> pl.LightningModule:
    """Training AML MAGIC model.

    Parameters
    ----------
    execution : int
        Execution number.
    experiment_name : str
        Name of the experiment.
    run_name : str
        Name of the MLFlow run.
    model_config : cfg.GNNArchConfigs
        Config for GNN model.
    test_loader : pyg.data.DataLoader
        Test data loader
    train_loader : pyg.data.DataLoader
        Train data loader
    training_config : cfg.DatasetTrainConfig
        Training configuration.
    mlflow_tracking_uri: str

    Returns
    -------
    pl.LightningModule
        Trained model.
    """
    pyg.seed_everything(execution + 100)
    torch.manual_seed(execution + 100)
    torch.use_deterministic_algorithms(True)
    lit_model = MAGICPl(**model_config.model_dump())
    loggers = [
        pl.loggers.MLFlowLogger(
            tracking_uri=mlflow_tracking_uri,
            experiment_name=experiment_name,
            run_name=f"{run_name}",
        ),
        pl.loggers.TensorBoardLogger(save_dir="./tb_logs", name=experiment_name),
    ]
    early_stop = pl.callbacks.early_stopping.EarlyStopping(
        monitor="train_f1", patience=10, mode="max"
    )
    checkpoint = pl.callbacks.model_checkpoint.ModelCheckpoint(
        monitor="train_f1", mode="max", save_top_k=1
    )
    callbacks = [early_stop, checkpoint]
    if training_config.use_swa:
        swa = pl.callbacks.StochasticWeightAveraging(swa_lrs=training_config.swa_lrs)
        callbacks.append(swa)
    trainer = pl.Trainer(
        max_epochs=training_config.max_epochs,
        deterministic=True,
        accelerator=training_config.accelerator,
        log_every_n_steps=5,
        enable_checkpointing=True,
        callbacks=callbacks,
        logger=loggers,
    )
    trainer.fit(lit_model, train_dataloaders=train_loader)

    # Model evaluation
    trainer.test(lit_model, dataloaders=test_loader, ckpt_path="best")
    path = trainer.checkpoint_callback.best_model_path
    lightning_model = MAGICPl.load_from_checkpoint(checkpoint_path=path, model=MAGICPl)
    lightning_model.magic_linkpred.eval()
    return lightning_model


def prepare_embedding_for_gradient_boosting(
    model: pl.LightningModule, train_data: pyg_data.Data, test_data: pyg_data.Data
) -> DataForGB:
    """Uses the trained GNN model to prepare embedding matrix of features for Gradient Boosting.
    Outputs the embeddings for the test and train data.
    Dimensionality of the embeddings is: (num_samples, num_features).

    Parameters
    ----------
    model : pl.LightningModule
        Model used to generate embeddings.
    train_data : pyg_data.Data
        Training data
    test_data : pyg_data.Data
        Testing data

    Returns
    -------
    DataForGB
        A tuple of np.ndarray, np.ndarray, np.ndarray, np.ndarray, containing:
        1. Embeddings for test data
        2. Embeddings for train data
        3. Labels for test data
        4. Labels for train data
    """
    with torch.no_grad():
        X_train_4_gb = []
        y_train_4_gb = []
        X_test_gb = []
        y_test_gb = []
        for data in train_data:
            data.to("cpu")
            logits, yhat = model.magic_linkpred(data.x, data.edge_index, data.edge_attr)
            X_train_4_gb.append(logits.numpy())
            y_train_4_gb.append(data.y)
        X_train_4_gb = np.concatenate(X_train_4_gb)
        y_train_4_gb = np.concatenate(y_train_4_gb)

        for data in test_data:
            data.to("cpu")
            logits, yhat = model.magic_linkpred(data.x, data.edge_index, data.edge_attr)
            X_test_gb.append(logits.numpy())
            y_test_gb.append(data.y)
        X_test_gb = np.concatenate(X_test_gb)
        y_test_gb = np.concatenate(y_test_gb)
    return DataForGB(X_test_gb, X_train_4_gb, y_test_gb, y_train_4_gb)


def train_gb(
    datas_for_gb: DataForGB,
    experiment_name: str,
    run_name: str,
    config_for_gb: cfg.XGBoostConfig | cfg.LightGBMConfig,
    mlflow_tracking_uri: str,
) -> GBTrainingResults:
    """A generic function for training the Gradient Boosting model - either the
    XGBoost or LightGBM.

    Parameters
    ----------
    datas_for_gb : DataForGB
        Data for training the Gradient Boosting model.
    experiment_name : str
        Name of the experiment, to be logged to MLFlow.
    run_name : str
        Name of the run.
    config_for_gb : cfg.XGBoostConfig | cfg.LightGBMConfig
        A config for either XGBoost or LightGBM.
    mlflow_tracking_uri: str

    Returns
    -------
    GBTrainingResults
        Training results for the Gradient Boosting model.
    """
    mlf.set_tracking_uri(mlflow_tracking_uri)
    mlf.set_experiment(experiment_name=experiment_name)
    with mlf.start_run(run_name=run_name) as run:
        mlf.lightgbm.autolog(log_models=False, log_datasets=False)
        mlf.xgboost.autolog(log_models=False, log_datasets=False)

        model = config_for_gb.build_model()
        model.fit(datas_for_gb.X_train, datas_for_gb.y_train)
        y_pred_lgbm = model.predict(datas_for_gb.X_test)

        prec_macro = precision_score(datas_for_gb.y_test, y_pred_lgbm, average="macro")
        rec_macro = recall_score(datas_for_gb.y_test, y_pred_lgbm, average="macro")
        f1_macro = f1_score(datas_for_gb.y_test, y_pred_lgbm, average="macro")

        prec_0 = precision_score(
            datas_for_gb.y_test, y_pred_lgbm, average="binary", labels=[0]
        )
        rec_0 = recall_score(
            datas_for_gb.y_test, y_pred_lgbm, average="binary", labels=[0]
        )
        f1_0 = f1_score(datas_for_gb.y_test, y_pred_lgbm, average="binary", labels=[0])

        print(f"\n Precision macro: {prec_macro}")
        print(f"Recall macro: {rec_macro}")
        print(f"F1 macro {f1_macro}")
        print(f"\n Precision ilicit: {prec_0}")
        print(f"Recall ilicit: {rec_0}")
        print(f"F1 ilict: {f1_0}\n")

        mlf.log_metric("Macro Precision", prec_macro)
        mlf.log_metric("Macro Recall", rec_macro)
        mlf.log_metric("Macro F1", f1_macro)

        mlf.log_metric("Ilicit Precision", prec_0)
        mlf.log_metric("Ilicit Recall", rec_0)
        mlf.log_metric("Ilicit F1", f1_0)
    return GBTrainingResults(
        model, prec_macro, rec_macro, f1_macro, prec_0, rec_0, f1_0
    )
