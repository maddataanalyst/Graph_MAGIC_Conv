"""Main module with the MAGIC model and its components."""

from typing import Any, Dict, List, Optional, Tuple, Union

import torch as th
import torch.nn as nn
import torch_geometric as pyg
import torch_geometric.nn as pyg_nn
import pytorch_lightning as pl
import torcheval.metrics.functional as metrics
from torch_geometric.nn.aggr import Aggregation
from torch_geometric.nn.inits import reset

import aml_magic.src.utils.configs as cfg

pyg.seed_everything(2)


class MAGICConv(pyg_nn.MessagePassing):
    """A convolution model that is a part of main contribution of the paper."""

    def __init__(
        self,
        net: th.nn.Module,
        dim_in: int,
        eps: float = 0.0,
        train_eps: bool = False,
        edge_dim: Optional[int] = None,
        **kwargs,
    ):
        kwargs.setdefault("aggr", "add")
        super().__init__(**kwargs)
        self.n_agg = 1 if isinstance(self.aggr, str) else len(self.aggr)
        self.nn = net
        self.initial_eps = eps
        if train_eps:
            self.eps = th.nn.Parameter(th.empty(1))
        else:
            self.register_buffer("eps", th.empty(1))
        if edge_dim is not None:
            self.anet = nn.Linear(edge_dim, dim_in)
        else:
            self.anet = None
        self.reset_parameters()

    def reset_parameters(self):
        reset(self.nn)
        self.eps.data.fill_(self.initial_eps)
        if self.anet is not None:
            self.anet.reset_parameters()

    def forward(self, x, edge_index, edge_attr, size=None) -> th.Tensor:
        if isinstance(x, th.Tensor):
            x = (x, x)

        # propagate_type: (x: OptPairTensor, edge_attr: OptTensor)
        out = self.propagate(edge_index, x=x, edge_attr=edge_attr, size=size)

        x_r = x[1]
        if x_r is not None:
            # Equation (8) from paper
            # Skip connection: concatenation of neighbor nodes embeddings
            # with the 'current node' embeddings.
            out = th.cat([out, (1 + self.eps) * x_r], dim=-1)

        return self.nn(out)

    def message(self, x_j: th.Tensor, edge_attr: th.Tensor) -> th.Tensor:
        if self.anet is None and x_j.size(-1) != edge_attr.size(-1):
            raise ValueError(
                "Node and edge feature dimensionalities do not "
                "match. Consider setting the 'edge_dim' "
                "attribute of 'GINEConv'"
            )

        if self.anet is not None:
            """Equation (7) from paper
            Alignment net (Anet) used to adjust dimensionality
            of edges to the dimensinality of nodes.
            """
            edge_attr = self.anet(edge_attr)

        return (x_j + edge_attr).relu()

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(nn={self.nn})"


class MAGICModel(nn.Module):
    """A model that uses MAGIC convolution layers for building nodes embeddings"""

    def __init__(
        self,
        dim_input: int,
        dim_edge: int,
        conv_sizes: Tuple[int, ...],
        gin_inner_layers: int = 1,
        eps: float = 0.1e-3,
        aggr: Optional[Union[str, List[str], Aggregation]] = "add",
        aggr_kwargs: Optional[Dict[str, Any]] = {},
        act_f: str = "leaky_relu",
        act_f_kwargs: dict = {"negative_slope": 0.2},
    ):
        super().__init__()
        convs = []
        all_dims = (dim_input,) + conv_sizes
        agg_size = 1 if isinstance(aggr, str) else len(aggr)
        for layer_idx, (dim_in, dim_out) in enumerate(zip(all_dims[:-1], all_dims[1:])):
            mlp = self.build_mlp(
                dim_in * agg_size + dim_in,
                dim_out,
                act_f,
                act_f_kwargs,
                gin_inner_layers,
            )
            if aggr == "power_mean":
                aggr = pyg_nn.PowerMeanAggregation(p=0.5, learn=True, channels=1)
            elif aggr == "softmax":
                aggr = pyg_nn.SoftmaxAggregation(**aggr_kwargs, learn=True)
            convs.append(
                MAGICConv(
                    mlp,
                    dim_in=dim_in,
                    eps=eps,
                    train_eps=True,
                    edge_dim=dim_edge,
                    aggr=aggr,
                )
            )

        self.convs = nn.ModuleList(convs)

    def build_mlp(
        self, dim_in, dim_out, act_f, act_f_kwargs, gin_inner_layers: int = 1
    ):
        last_in_dim = dim_in
        lin_layers = []
        for i in range(gin_inner_layers):
            lin = nn.Linear(last_in_dim, dim_out)
            nn.init.xavier_uniform_(lin.weight)
            lin_layers.append(lin)
            act_layer = None
            if act_f == "leaky_relu":
                act_layer = nn.LeakyReLU(inplace=True, **act_f_kwargs)
            elif act_f == "relu":
                act_layer = nn.ReLU(inplace=True)
            elif act_f == "tanh":
                act_layer = nn.Tanh()
            else:
                raise ValueError(f"Unknown activation function: {act_f}")

            lin_layers.append(act_layer)
            last_in_dim = dim_out
        mlp = nn.Sequential(*lin_layers)
        return mlp

    def forward(self, x, edge_index, edge_attr):
        h = x
        for conv in self.convs:
            h = conv(h, edge_index, edge_attr)
        return h.relu()


class MAGIC_LinkPredictor(nn.Module):
    """Magic model with a link predictor on top of the node embeddings"""

    def __init__(
        self,
        dim_input: int,
        dim_edge: int,
        conv_sizes: Tuple[int, ...],
        gin_inner_layers: int,
        linkpred_sizes: Tuple[int, ...],
        embed_reduction_mode: cfg.EmbedReduction = cfg.EmbedReduction.MULT,
        eps: float = 0.1e-3,
        n_cls: int = 2,
        batch_norm: bool = False,
        aggr: Optional[Union[str, List[str], Aggregation]] = "add",
        aggr_kwargs: Optional[Dict[str, Any]] = {},
        conv_act_f: str = "leaky_relu",
        conv_act_f_kwargs: dict = {"negative_slope": 0.2},
        linkpred_act_f: str = "leaky_relu",
        linkpred_act_f_kwargs: dict = {"negative_slope": 0.2},
    ):
        super().__init__()
        self.magic = MAGICModel(
            dim_input,
            dim_edge,
            conv_sizes,
            gin_inner_layers,
            eps,
            aggr,
            aggr_kwargs,
            conv_act_f,
            conv_act_f_kwargs,
        )
        self.embed_reduction_mode = embed_reduction_mode
        self.batch_norm = batch_norm
        self.mlp = self._build_link_predictor(
            conv_sizes[-1], linkpred_sizes, n_cls, linkpred_act_f, linkpred_act_f_kwargs
        )

    def _build_link_predictor(
        self,
        dim_embed: int,
        linkpred_sizes: Tuple[int, ...],
        n_cls: int = 2,
        act_f: str = "leaky_relu",
        act_f_kwargs: dict = {"negative_slope": 0.2},
    ):
        embed_dim = (
            dim_embed * 2
            if self.embed_reduction_mode == cfg.EmbedReduction.CONCAT
            else dim_embed
        )
        all_sizes = (embed_dim,) + linkpred_sizes
        mlp = []
        if self.batch_norm:
            mlp.append(nn.BatchNorm1d(embed_dim))
        for dim_in, dim_out in zip(all_sizes[:-1], all_sizes[1:]):
            lin = nn.Linear(dim_in, dim_out)
            mlp.append(lin)

            activation = None
            if act_f == "leaky_relu":
                activation = nn.LeakyReLU(inplace=True, **act_f_kwargs)
            elif act_f == "relu":
                activation = nn.ReLU(inplace=True)
            elif act_f == "tanh":
                activation = nn.Tanh()
            else:
                raise ValueError(f"Unknown activation function: {act_f}")
            mlp.append(activation)
        lin_out = nn.Linear(dim_out, n_cls)
        mlp.append(lin_out)
        return nn.Sequential(*mlp)

    def forward(self, x, edge_index, edge_attr):
        h = self.magic(x, edge_index, edge_attr)

        h_from = edge_index[0, :]
        h_to = edge_index[1, :]

        if self.embed_reduction_mode == cfg.EmbedReduction.CONCAT:
            h = th.cat([h[h_from, :], h[h_to, :]], dim=1)
        elif self.embed_reduction_mode == cfg.EmbedReduction.MULT:
            h = h[h_from, :] * h[h_to, :]
        elif self.embed_reduction_mode == cfg.EmbedReduction.MEAN:
            h = (h[h_from, :] + h[h_to, :]) / 2.0
        else:
            raise ValueError(
                f"Unknown embed reduction mode: {self.embed_reduction_mode}"
            )

        logits = self.mlp(h)
        yhat = nn.functional.softmax(logits, dim=-1)

        return h, yhat


class MAGICPl(pl.LightningModule):
    """Pytorch Lightning wrapper for the MAGIC model link predictor module."""

    def __init__(
        self,
        dim_input: int,
        dim_edge: int,
        conv_sizes: Tuple[int, ...],
        gin_inner_layers: int,
        linkpred_sizes: Tuple[int, ...],
        embed_reduction_mode: cfg.EmbedReduction = cfg.EmbedReduction.MULT,
        eps: float = 0.1e-3,
        n_cls: int = 2,
        lr: float = 0.001,
        batch_norm: bool = False,
        weight: float = None,
        aggr: Optional[Union[str, List[str], Aggregation]] = "add",
        aggr_kwargs: Optional[Dict[str, Any]] = {},
        conv_act_f: str = "leaky_relu",
        conv_act_f_kwargs: dict = {"negative_slope": 0.2},
        linkpred_act_f: str = "leaky_relu",
        linkpred_act_f_kwargs: dict = {"negative_slope": 0.2},
    ):
        super().__init__()
        self.magic_linkpred = MAGIC_LinkPredictor(
            dim_input,
            dim_edge,
            conv_sizes,
            gin_inner_layers,
            linkpred_sizes,
            embed_reduction_mode,
            eps,
            n_cls,
            batch_norm,
            aggr,
            aggr_kwargs,
            conv_act_f,
            conv_act_f_kwargs,
            linkpred_act_f,
            linkpred_act_f_kwargs,
        )
        self.lr = lr
        self.loss = th.nn.CrossEntropyLoss(weight=th.tensor(weight) if weight else None)
        self.save_hyperparameters()

    def _step(
        self,
        x: th.Tensor,
        edge_index: th.Tensor,
        edge_attr: th.Tensor,
        y: th.Tensor,
        phase: str = "train",
    ):
        _, yhat = self.magic_linkpred(x, edge_index, edge_attr)
        loss = self.loss(th.atleast_2d(yhat.squeeze()), y)
        cls = th.argmax(yhat, dim=1)

        f1 = metrics.binary_f1_score(cls, y)
        precision = metrics.binary_precision(cls, y)
        recall = metrics.binary_recall(cls, y)

        self.log(
            f"{phase}_loss",
            loss,
            prog_bar=True,
            on_epoch=True,
            on_step=False,
            batch_size=y.size(0),
        )
        self.log(
            f"{phase}_f1",
            f1,
            prog_bar=True,
            on_epoch=True,
            on_step=False,
            batch_size=y.size(0),
        )
        self.log(
            f"{phase}_precision",
            precision,
            prog_bar=True,
            on_epoch=True,
            on_step=False,
            batch_size=y.size(0),
        )
        self.log(
            f"{phase}_recall",
            recall,
            prog_bar=True,
            on_epoch=True,
            on_step=False,
            batch_size=y.size(0),
        )
        return loss

    def training_step(self, data: pyg.data.Data):
        return self._step(data.x, data.edge_index, data.edge_attr, data.y, "train")

    def validation_step(self, data: pyg.data.Data):
        return self._step(data.x, data.edge_index, data.edge_attr, data.y, "val")

    def test_step(self, data: pyg.data.Data):
        return self._step(data.x, data.edge_index, data.edge_attr, data.y, "test")

    def configure_optimizers(self):
        return th.optim.Adam(self.parameters(), lr=self.lr)

    def forward(self, data: pyg.data.Data) -> th.Tensor:
        _, yhat = self.magic_linkpred(data.x, data.edge_index, data.edge_attr)
        return yhat
