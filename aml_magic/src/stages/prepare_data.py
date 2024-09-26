"""First stage of the processing - data preparation and parsing."""

import autoroot  # noqa
import typer
import torch
import torch_geometric as pyg
import joblib
from typing import List, Tuple
from tqdm.auto import tqdm

import aml_magic.src.consts as cc
import aml_magic.src.utils.configs as cfg


def load_graphs(
    dataset: str, config: cfg.DatasetsConfig
) -> Tuple[List[pyg.data.Data], List[pyg.data.Data]]:
    """Loads the graphs for a dataset. Each dataset is divided into 366
    pt files. Each file contains a single graph. The function loads all
    the graphs and divides them into training and testing sets based on
    the train_percentage in the config.

    Parameters
    ----------
    dataset : str
        The name of the dataset to load.
    config : cfg.DatasetsConfig
        Configuration object for the datasets.

    Returns
    -------
    Tuple[List[pyg.data.Data], List[pyg.data.Data]]
        Tuple containing trianing and testing graphs for a given dataset.
    """
    graphs = []
    dataset_path = config.get_dataset_input_path(dataset)
    for ptfile in dataset_path.iterdir():
        if ptfile.suffix == ".pt":
            ds = torch.load(ptfile)
            ds_pyg = pyg.data.Data(
                x=ds.x.T.to(torch.float),
                edge_index=ds.edge_index,
                edge_attr=ds.edge_attr.T.to(torch.float),
                y=ds.y,
            )
            graphs.append(ds_pyg)
    n_graphs = len(graphs)
    train_graphs = graphs[0 : int(n_graphs * config.train_percentage)]
    test_graphs = graphs[int(n_graphs * config.train_percentage) :]
    return train_graphs, test_graphs


def save_processed_graphs(
    train_graphs: List[pyg.data.Data],
    test_graphs: List[pyg.data.Data],
    dataset: str,
    config: cfg.DatasetsConfig,
):
    """Saves the processed graphs to the output directory.

    Parameters
    ----------
    train_graphs : List[pyg.data.Data]
        List of training graphs.
    test_graphs : List[pyg.data.Data]
        List of testing graphs.
    dataset : str
        Name of the dataset.
    config : cfg.DatasetsConfig
        Configuration object for the datasets.
    """
    dataset_output_path = config.get_dataset_output_path(dataset)
    dataset_output_path.mkdir(parents=True, exist_ok=True)

    joblib.dump(train_graphs, dataset_output_path / "train_graphs.pkl")
    joblib.dump(test_graphs, dataset_output_path / "test_graphs.pkl")


def main(datasets: List[str] = ["amlsim_31", "amlsim_51", "amlsim_101", "amlsim_201"]):
    params = cfg.load_params(cc.STAGE_PREPARE_DATA, cc.STAGE_PARAMS_DIR)
    datasets_config = cfg.DatasetsConfig(**params)
    for ds in tqdm(datasets):
        graphs = load_graphs(ds, datasets_config)
        save_processed_graphs(*graphs, ds, datasets_config)


if __name__ == "__main__":
    typer.run(main)
