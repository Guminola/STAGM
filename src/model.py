import inspect
from typing import Any

import torch
from mamba_ssm import Mamba as Mamba
from torch import nn

from torch_geometric.nn import GCNConv, MessagePassing
from torch_geometric.nn.inits import reset
from torch_geometric.nn.resolver import activation_resolver, normalization_resolver
from torch_geometric.typing import Adj, OptTensor, Tensor
from torch_geometric.utils import degree, sort_edge_index, to_dense_batch


def _permute_within_batch(node_features: Tensor, batch: Tensor) -> Tensor:
    permuted_indices = [
        (batch == b).nonzero().squeeze()[torch.randperm((batch == b).sum().item())]
        for b in torch.unique(batch)
    ]
    return torch.cat(permuted_indices)


def _reverse_within_batch(batch: Tensor) -> Tensor:
    reversed_indices = [
        (batch == b).nonzero(as_tuple=True)[0].flip(0) for b in torch.unique(batch)
    ]
    return torch.cat(reversed_indices)


class GPSConv(torch.nn.Module):
    def __init__(
        self,
        channels: int,
        conv: MessagePassing | None,
        dropout: float = 0.0,
        act: str = "relu",
        act_kwargs: dict[str, Any] | None = None,
        norm: str | None = "batch_norm",
        norm_kwargs: dict[str, Any] | None = None,
        order_by_degree: bool = False,
        shuffle_ind: int = 0,
        d_state: int = 16,
        d_conv: int = 4,
        bidirectional: bool = False,
    ):
        super().__init__()

        assert not (order_by_degree and shuffle_ind != 0), (
            f"order_by_degree={order_by_degree} and shuffle_ind={shuffle_ind} "
            "are mutually exclusive"
        )

        self.channels = channels
        self.conv = conv
        self.dropout = dropout
        self.order_by_degree = order_by_degree
        self.shuffle_ind = shuffle_ind
        self.bidirectional = bidirectional

        self.mamba = Mamba(d_model=channels, d_state=d_state, d_conv=d_conv, expand=1)
        if self.bidirectional:
            self.mamba_rev = Mamba(
                d_model=channels, d_state=d_state, d_conv=d_conv, expand=1
            )
            self.bidirectional_proj = nn.Linear(channels * 2, channels)

        self.mlp = nn.Sequential(
            nn.Linear(channels, channels * 2),
            activation_resolver(act, **(act_kwargs or {})),
            nn.Dropout(dropout),
            nn.Linear(channels * 2, channels),
            nn.Dropout(dropout),
        )

        norm_kwargs = norm_kwargs or {}
        self.norm1 = normalization_resolver(norm, channels, **norm_kwargs)
        self.norm2 = normalization_resolver(norm, channels, **norm_kwargs)
        self.norm3 = normalization_resolver(norm, channels, **norm_kwargs)

        self.norm_with_batch = False
        if self.norm1 is not None:
            signature = inspect.signature(self.norm1.forward)
            self.norm_with_batch = "batch" in signature.parameters

    def reset_parameters(self):
        if self.conv is not None:
            self.conv.reset_parameters()
        reset(self.mlp)
        if self.bidirectional:
            self.bidirectional_proj.reset_parameters()
        for norm in (self.norm1, self.norm2, self.norm3):
            if norm is not None:
                norm.reset_parameters()

    def _run_mamba(self, node_features: Tensor, batch: Tensor) -> Tensor:
        dense, mask = to_dense_batch(node_features, batch)
        forward_out = self.mamba(dense)[mask]

        if not self.bidirectional:
            return forward_out

        rev_idx = _reverse_within_batch(batch)
        dense_rev, mask_rev = to_dense_batch(node_features[rev_idx], batch[rev_idx])
        backward_out = self.mamba_rev(dense_rev)[mask_rev][rev_idx]
        return self.bidirectional_proj(torch.cat([forward_out, backward_out], dim=-1))

    def _apply_norm(
        self, norm: nn.Module | None, node_features: Tensor, batch: Tensor
    ) -> Tensor:
        if norm is None:
            return node_features
        return (
            norm(node_features, batch=batch)
            if self.norm_with_batch
            else norm(node_features)
        )

    def forward(
        self,
        node_features: Tensor,
        edge_index: Adj,
        batch: Tensor,
        **kwargs,
    ) -> Tensor:
        branch_outputs = []

        if self.conv is not None:
            local_out = self.conv(node_features, edge_index, **kwargs)
            local_out = nn.functional.dropout(
                local_out, p=self.dropout, training=self.training
            )
            local_out = local_out + node_features
            local_out = self._apply_norm(self.norm1, local_out, batch)
            branch_outputs.append(local_out)

        mamba_input = node_features
        if self.order_by_degree:
            node_degree = degree(edge_index[0], mamba_input.size(0)).to(torch.long)
            order_tensor = torch.stack([batch, node_degree], dim=1).T
            _, mamba_input = sort_edge_index(order_tensor, edge_attr=mamba_input)

        if self.shuffle_ind == 0:
            global_out = self._run_mamba(mamba_input, batch)
        else:
            shuffled_outputs = []
            for _ in range(self.shuffle_ind):
                perm = _permute_within_batch(mamba_input, batch)
                shuffled_outputs.append(self._run_mamba(mamba_input[perm], batch)[perm])
            global_out = sum(shuffled_outputs) / self.shuffle_ind

        global_out = nn.functional.dropout(
            global_out, p=self.dropout, training=self.training
        )
        global_out = global_out + node_features
        global_out = self._apply_norm(self.norm2, global_out, batch)
        branch_outputs.append(global_out)

        combined = sum(branch_outputs)
        combined = combined + self.mlp(combined)
        return self._apply_norm(self.norm3, combined, batch)


class Encoder(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        activation,
        base_model=GCNConv,
        num_layers: int = 2,
        dropout: float = 0.0,
        order_by_degree: bool = False,
        shuffle_ind: int = 0,
        d_state: int = 16,
        d_conv: int = 4,
        bidirectional: bool = False,
    ):
        super().__init__()
        assert num_layers >= 1

        self.num_layers = num_layers
        self.activation = activation

        hidden_channels = out_channels if num_layers == 1 else 2 * out_channels

        self.input_proj = (
            nn.Identity()
            if in_channels == hidden_channels
            else nn.Linear(in_channels, hidden_channels)
        )

        def make_gps_layer(channels: int) -> GPSConv:
            return GPSConv(
                channels=channels,
                conv=base_model(channels, channels),
                dropout=dropout,
                order_by_degree=order_by_degree,
                shuffle_ind=shuffle_ind,
                d_state=d_state,
                d_conv=d_conv,
                bidirectional=bidirectional,
            )

        self.gps_layers = nn.ModuleList(
            make_gps_layer(hidden_channels) for _ in range(num_layers)
        )

        self._needs_output_proj = num_layers > 1
        if self._needs_output_proj:
            self.output_proj = nn.Linear(hidden_channels, out_channels)

    def reset_parameters(self):
        if isinstance(self.input_proj, nn.Linear):
            self.input_proj.reset_parameters()
        for layer in self.gps_layers:
            layer.reset_parameters()
        if self._needs_output_proj:
            self.output_proj.reset_parameters()

    def forward(
        self,
        node_features: Tensor,
        edge_index: Tensor,
        batch: Tensor,
    ) -> Tensor:
        node_emb = self.input_proj(node_features)

        for layer in self.gps_layers:
            node_emb = self.activation(layer(node_emb, edge_index, batch))

        if self._needs_output_proj:
            node_emb = self.output_proj(node_emb)

        return node_emb


class _ProjectionMixin(torch.nn.Module):
    def __init__(self, num_hidden: int, num_proj_hidden: int, tau: float):
        super().__init__()
        self.tau = tau
        self.forward_proj_1 = nn.Linear(num_hidden, num_proj_hidden)
        self.forward_proj_2 = nn.Linear(num_proj_hidden, num_hidden)

    def projection(self, gnn_embedding: Tensor) -> Tensor:
        projected = nn.functional.elu(self.forward_proj_1(gnn_embedding))
        return self.forward_proj_2(projected)

    def similarity_matrix(self, emb_a: Tensor, emb_b: Tensor) -> Tensor:
        return torch.mm(
            nn.functional.normalize(emb_a), nn.functional.normalize(emb_b).t()
        )

    def tau_scaling(self, sim: Tensor) -> Tensor:
        return torch.exp(sim / self.tau)

    def _reduce(self, per_node_loss: Tensor, mean: bool) -> Tensor:
        return per_node_loss.mean() if mean else per_node_loss.sum()

    @staticmethod
    def _strip_self_loops(adj: Tensor) -> Tensor:
        adj = adj - torch.diag_embed(adj.diag())
        adj[adj > 0] = 1
        return adj

    @staticmethod
    def _positive_pair_counts(adj: Tensor) -> Tensor:
        return torch.sum(adj, 1).mul(2).add(1).squeeze()


class MVmodel(_ProjectionMixin):
    def __init__(
        self,
        encoder: Encoder,
        num_hidden: int,
        num_proj_hidden: int,
        tau: float = 0.5,
    ):
        super().__init__(num_hidden, num_proj_hidden, tau)
        self.encoder = encoder

    def forward(
        self, node_features: Tensor, edge_index: Tensor, batch: Tensor
    ) -> Tensor:
        gnn_embedding = self.encoder(node_features, edge_index, batch)
        return self.projection(gnn_embedding)

    def _neighbor_contrastive_term(
        self,
        emb_a: Tensor,
        emb_b: Tensor,
        adj: Tensor,
        pseudo_labels: Tensor | None = None,
    ) -> Tensor:
        adj = self._strip_self_loops(adj)
        positive_pair_counts = self._positive_pair_counts(adj)

        intra_sim = self.tau_scaling(self.similarity_matrix(emb_a, emb_a))
        inter_sim = self.tau_scaling(self.similarity_matrix(emb_a, emb_b))

        if pseudo_labels is not None:
            negative_mask = (
                pseudo_labels.view(-1, 1) != pseudo_labels.view(1, -1)
            ).float()
            intra_sim_denom = intra_sim * negative_mask
            inter_sim_denom = inter_sim * negative_mask
        else:
            intra_sim_denom = intra_sim
            inter_sim_denom = inter_sim

        numerator = (
            inter_sim.diag() + intra_sim.mul(adj).sum(1) + inter_sim.mul(adj).sum(1)
        )
        denominator = intra_sim_denom.sum(1) + inter_sim_denom.sum(1) - intra_sim.diag()

        return -torch.log((numerator / denominator) / positive_pair_counts)

    def contrastive_loss_biased(
        self,
        emb_a: Tensor,
        emb_b: Tensor,
        adj: Tensor,
        pseudo_labels: Tensor,
        mean: bool = True,
    ) -> Tensor:
        per_node = (
            self._neighbor_contrastive_term(emb_a, emb_b, adj, pseudo_labels)
            + self._neighbor_contrastive_term(emb_b, emb_a, adj, pseudo_labels)
        ) * 0.5
        return self._reduce(per_node, mean)


class SVmodel(_ProjectionMixin):
    def __init__(
        self,
        encoder: Encoder,
        num_hidden: int,
        num_proj_hidden: int,
        tau: float = 0.5,
    ):
        super().__init__(num_hidden, num_proj_hidden, tau)
        self.encoder = encoder

    def forward(
        self, node_features: Tensor, edge_index: Tensor, batch: Tensor
    ) -> Tensor:
        gnn_embedding = self.encoder(node_features, edge_index, batch)
        return self.projection(gnn_embedding)

    def _neighbor_contrastive_term(
        self,
        emb_a: Tensor,
        emb_b: Tensor,
        adj: Tensor,
        sample_mask: Tensor | None = None,
    ) -> Tensor:
        adj = self._strip_self_loops(adj)
        positive_pair_counts = self._positive_pair_counts(adj)

        intra_sim = self.tau_scaling(self.similarity_matrix(emb_a, emb_a))
        inter_sim = self.tau_scaling(self.similarity_matrix(emb_a, emb_b))

        if sample_mask is not None:
            intra_sim = intra_sim * sample_mask
            inter_sim = inter_sim * sample_mask

        numerator = (
            inter_sim.diag() + intra_sim.mul(adj).sum(1) + inter_sim.mul(adj).sum(1)
        )
        denominator = intra_sim.sum(1) + inter_sim.sum(1) - intra_sim.diag()

        return -torch.log((numerator / denominator) / positive_pair_counts)

    def contrastive_loss(
        self,
        emb_a: Tensor,
        emb_b: Tensor,
        adj: Tensor,
        sample_mask: Tensor | None = None,
        mean: bool = True,
    ) -> Tensor:
        per_node = (
            self._neighbor_contrastive_term(emb_a, emb_b, adj, sample_mask)
            + self._neighbor_contrastive_term(emb_b, emb_a, adj, sample_mask)
        ) * 0.5
        return self._reduce(per_node, mean)


def drop_feature(node_features: Tensor, drop_prob: float) -> Tensor:
    drop_mask = (
        torch.empty(node_features.size(1), device=node_features.device).uniform_(0, 1)
        < drop_prob
    )
    node_features = node_features.clone()
    node_features[:, drop_mask] = 0
    return node_features


def filter_adj(
    row: Tensor, col: Tensor, edge_attr: OptTensor, keep_mask: Tensor
) -> tuple[Tensor, Tensor, OptTensor]:
    filtered_attr = None if edge_attr is None else edge_attr[keep_mask]
    return row[keep_mask], col[keep_mask], filtered_attr


def dropout_adj(
    edge_index: Tensor,
    edge_attr: Tensor,
    force_undirected: bool = False,
    training: bool = True,
) -> tuple[Tensor, Tensor]:
    if not training:
        return edge_index, edge_attr

    row, col = edge_index

    if force_undirected:
        upper_tri_mask = row <= col
        row, col, edge_attr = (
            row[upper_tri_mask],
            col[upper_tri_mask],
            edge_attr[upper_tri_mask],
        )

    keep_mask = torch.rand(edge_attr.size(0), device=edge_attr.device) >= edge_attr
    row, col, edge_attr = filter_adj(row, col, edge_attr, keep_mask)

    if force_undirected:
        edge_index = torch.stack([torch.cat([row, col]), torch.cat([col, row])], dim=0)
        edge_attr = torch.cat([edge_attr, edge_attr], dim=0)
    else:
        edge_index = torch.stack([row, col], dim=0)

    return edge_index, edge_attr


def multiple_dropout_average(
    edge_index: Tensor,
    edge_attr: Tensor,
    force_undirected: bool = False,
    training: bool = True,
) -> tuple[Tensor, Tensor]:
    if not training:
        return edge_index, edge_attr
    return dropout_adj(edge_index, edge_attr, force_undirected=force_undirected)
