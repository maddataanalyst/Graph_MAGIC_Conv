"""This module holds utility classses used to define the model configuration and training setup"""

import yaml
from pydantic import BaseModel
from typing import Any, Tuple, Optional, Union, List, Dict, Literal
from pathlib import Path
from enum import Enum

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier


def load_params(name: str, dir_name: str = None, key_name: str = "params") -> Dict:
    """Loads parms from a yaml file.

    Parameters
    ----------
    name : str
        Name to load.
    dir_name : str, optional
        Directory for stage-specific params
    key_name : str, optional
        Name of the key to return.

    Returns
    -------
    Dict
        Dictionary with proper params
    """
    pth = Path(f"{name}.yaml") if not dir_name else Path(dir_name, f"{name}.yaml")
    params_dict = yaml.safe_load(open(pth, "r"))
    if key_name:
        return params_dict[key_name]
    return params_dict


class EmbedReduction(str, Enum):
    """Enum class to define the different types of reduction operations that can be applied to the embeddings"""

    CONCAT = "concat"
    MULT = "mult"
    MEAN = "mean"


class DataConfig(BaseModel):
    """Config specific for a given dataset, focused on the filenames, paths and experiment names"""

    output_filename: str
    run_name: str
    experiment_name: str


class DatasetTrainConfig(BaseModel):
    """Config specific for a given dataset"""

    max_epochs: int
    batch_size: int
    accelerator: str = "cpu"
    use_swa: bool = False
    swa_lrs: float = 0.0
    weight: List[float] = None


class DatasetConfig(BaseModel):
    """General config for a specific dataset"""

    name: str
    data_config: DataConfig


class DatasetsConfig(BaseModel):
    """General config for all datasets and default training"""

    data_path: str
    output_path: str
    datasets: Dict[str, DatasetConfig]
    train_percentage: float = 0.8

    def get_config(self, dataset: str) -> DatasetConfig:
        if dataset not in self.datasets:
            raise ValueError(f"Dataset {dataset} not found in the config")
        return self.datasets[dataset]

    def get_dataset_input_path(self, dataset: str) -> Path:
        return Path(self.data_path) / dataset

    def get_dataset_output_path(self, dataset: str) -> Path:
        return Path(self.output_path) / dataset


class ModelConfig(BaseModel):
    """MAGIC model config class"""

    dim_input: int
    dim_edge: int
    conv_sizes: Tuple[int, ...]
    gin_inner_layers: int
    linkpred_sizes: Tuple[int, ...]
    embed_reduction_mode: EmbedReduction
    eps: float
    n_cls: int
    lr: float
    weight: Optional[List[float]]
    batch_norm: bool
    aggr: Optional[Union[str, List[str]]] = "add"
    aggr_kwargs: Optional[Dict[str, Any]] = {}
    conv_act_f: str = "leaky_relu"
    conv_act_f_kwargs: dict = {"negative_slope": 0.2}
    linkpred_act_f: str = "leaky_relu"
    linkpred_act_f_kwargs: dict = {"negative_slope": 0.2}


class GNNArchConfigs(BaseModel):
    """General experimental config class used to define per-dataset model configurations"""

    default: ModelConfig
    dataset_configs: Dict[str, ModelConfig]

    def __getitem__(self, dataset_config: str) -> ModelConfig:
        if dataset_config in self.dataset_configs:
            return self.dataset_configs[dataset_config]
        else:
            return self.default


class GNNTrainConfig(BaseModel):
    """Training config per dataset"""

    default: DatasetTrainConfig
    dataset_training_configs: Dict[str, DatasetTrainConfig]

    def __getitem__(self, dataset_config: str) -> DatasetTrainConfig:
        if dataset_config in self.dataset_training_configs:
            return self.dataset_training_configs[dataset_config]
        else:
            return self.default


class GNNConfig(BaseModel):
    """Configuration for training GNN per each dataset"""

    model_architectures: GNNArchConfigs
    training_configs: GNNTrainConfig

    def get_achitecture_for_dataset(self, dataset: str) -> ModelConfig:
        return self.model_architectures[dataset]

    def get_training_config_for_dataset(self, dataset: str) -> DatasetTrainConfig:
        return self.training_configs[dataset]


class LightGBMConfig(BaseModel):
    random_state: int = 42
    num_leaves: int = 256
    learning_rate: float = 0.05
    n_estimators: int = 250
    n_jobs: int = -1
    boosting_type: str = "gbdt"
    class_weight: Optional[Dict[object, float]] = None

    def build_model(self):
        return LGBMClassifier(
            random_state=self.random_state,
            num_leaves=self.num_leaves,
            learning_rate=self.learning_rate,
            n_estimators=self.n_estimators,
            n_jobs=self.n_jobs,
            boosting_type=self.boosting_type,
            class_weight=self.class_weight,
        )


class XGBoostConfig(BaseModel):
    random_state: int = 42
    n_estimators: int = 250
    learning_rate: float = 0.05
    n_jobs: int = -1
    scale_pos_weight: Optional[float] = 1

    def build_model(self):
        return XGBClassifier(
            random_state=self.random_state,
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            n_jobs=self.n_jobs,
            scale_pos_weight=self.scale_pos_weight,
        )


class XGBoostConfigs(BaseModel):
    default: XGBoostConfig
    dataset_configs: Dict[str, XGBoostConfig] = dict()

    def get_training_config_for_dataset(self, dataset: str) -> LightGBMConfig:
        if dataset in self.dataset_configs:
            return self.dataset_configs[dataset]
        else:
            return self.default


class LightGBMConfigs(BaseModel):
    """Configuration for the LightGBM model"""

    default: LightGBMConfig
    dataset_configs: Dict[str, LightGBMConfig]

    def get_training_config_for_dataset(self, dataset: str) -> LightGBMConfig:
        if dataset in self.dataset_configs:
            return self.dataset_configs[dataset]
        else:
            return self.default


class GradientBoostingConfigs(BaseModel):
    """Configuration for the Gradient Boosting model"""

    lightgbm: LightGBMConfigs
    xgboost: XGBoostConfigs

    def get_config_for_gb_impl(self, gb_impl_type: str):
        if gb_impl_type == "xgboost":
            return self.xgboost
        elif gb_impl_type == "lightgbm":
            return self.lightgbm
        else:
            raise ValueError(
                f"Gradient boosting implementation {gb_impl_type} not found"
            )


class ExperimentConfig(BaseModel):
    datasets_to_train: List[str]
    gradient_boosting_impl: Literal["xgboost", "lightgbm"]
    n_repeats: int
