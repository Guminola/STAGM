# Standard Library
import inspect
from typing import Any, Dict, Optional, Tuple

# PyTorch & Deep Learning
import torch
import torch.nn as nn
from mamba_ssm import Mamba

# PyTorch Geometric (Graph Neural Networks)
from torch_geometric.nn import GCNConv, MessagePassing
from torch_geometric.nn.inits import reset
from torch_geometric.nn.resolver import activation_resolver, normalization_resolver
from torch_geometric.typing import Adj, OptTensor, Tensor
from torch_geometric.utils import (
    degree,
    sort_edge_index,
    to_dense_batch,
)


def _permute_within_batch(node_features: Tensor, batch: Tensor) -> Tensor:
    """Returns a permuted index tensor that shuffles nodes within each graph."""
    permuted_indices = [
        (batch == b).nonzero().squeeze()[torch.randperm((batch == b).sum().item())]
        for b in torch.unique(batch)
    ]
    return torch.cat(permuted_indices)


def _reverse_within_batch(batch: Tensor) -> Tensor:
    """
    Returns an index tensor that reverses node order within each graph.

    Unlike a random permutation, this index is self-inverse: gathering with
    it twice returns the original order. That lets the backward Mamba pass
    reuse the same index both to build the reversed sequence and to restore
    the original node order afterwards.
    """
    reversed_indices = [
        (batch == b).nonzero(as_tuple=True)[0].flip(0) for b in torch.unique(batch)
    ]
    return torch.cat(reversed_indices)


class GPSConv(torch.nn.Module):
    """
    GPS-style layer combining a local MPNN with a global Mamba SSM.

    NOTE: `conv` must be a square map (its input and output width must both
    equal `channels`). GPSConv's residual connections (`h + node_features`)
    and its feed-forward block all operate at a fixed width of `channels`,
    so any dimensionality change must happen outside this layer.

    Args:
        channels:         Node feature dimensionality (in == out).
        conv:             Local message-passing layer (e.g. GCNConv), mapping
                           channels -> channels.
        dropout:          Dropout applied after each sub-layer.
        act:              Activation name for the feed-forward MLP.
        act_kwargs:       Extra kwargs forwarded to the activation resolver.
        norm:             Normalisation layer name (or None).
        norm_kwargs:      Extra kwargs forwarded to the normalisation resolver.
        order_by_degree:  Sort nodes by degree before feeding into Mamba.
        shuffle_ind:      Number of random permutations to average (0 = no shuffle).
        d_state:          Mamba SSM state size.
        d_conv:           Mamba conv kernel size.
        bidirectional:    Run a second Mamba over the reversed sequence and fuse
                           it with the forward pass, so every node gets context
                           from both directions of the global scan instead of
                           only from nodes that preceded it.
    """

    def __init__(
        self,
        channels: int,
        conv: Optional[MessagePassing],
        dropout: float = 0.0,
        act: str = "relu",
        act_kwargs: Optional[Dict[str, Any]] = None,
        norm: Optional[str] = "batch_norm",
        norm_kwargs: Optional[Dict[str, Any]] = None,
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
            # Separate weights for the backward scan (standard BiMamba practice);
            # fuse forward + backward hidden states back down to `channels`.
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
            sig = inspect.signature(self.norm1.forward)
            self.norm_with_batch = "batch" in sig.parameters

    def reset_parameters(self):
        if self.conv is not None:
            self.conv.reset_parameters()
        reset(self.mlp)
        if self.bidirectional:
            self.bidirectional_proj.reset_parameters()
        for norm in (self.norm1, self.norm2, self.norm3):
            if norm is not None:
                norm.reset_parameters()

    def _run_mamba(self, x: Tensor, batch: Tensor) -> Tensor:
        """
        Runs the forward Mamba scan (and, if `bidirectional`, a second scan
        over the reversed sequence) and returns node-level output aligned
        with `x`'s own node order.
        """
        dense, mask = to_dense_batch(x, batch)
        h = self.mamba(dense)[mask]

        if self.bidirectional:
            rev_idx = _reverse_within_batch(batch)
            dense_rev, mask_rev = to_dense_batch(x[rev_idx], batch[rev_idx])
            # rev_idx is self-inverse: gathering with it again undoes the
            # reversal and restores x's original node order.
            h_rev = self.mamba_rev(dense_rev)[mask_rev][rev_idx]
            h = self.bidirectional_proj(torch.cat([h, h_rev], dim=-1))

        return h

    def _apply_norm(self, norm, node_features: Tensor, batch: Tensor) -> Tensor:
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

        # --- Local MPNN branch ---
        if self.conv is not None:
            h = self.conv(node_features, edge_index, **kwargs)
            h = nn.functional.dropout(h, p=self.dropout, training=self.training)
            h = h + node_features  # residual (requires conv output width == channels)
            h = self._apply_norm(self.norm1, h, batch)
            branch_outputs.append(h)

        # --- Global Mamba branch ---
        x = node_features
        if self.order_by_degree:
            deg = degree(edge_index[0], x.size(0)).to(torch.long)
            order_tensor = torch.stack([batch, deg], dim=1).T
            _, x = sort_edge_index(order_tensor, edge_attr=x)

        if self.shuffle_ind == 0:
            h = self._run_mamba(x, batch)
        else:
            shuffled = []
            for _ in range(self.shuffle_ind):
                perm = _permute_within_batch(x, batch)
                h_i = self._run_mamba(x[perm], batch)[perm]
                shuffled.append(h_i)
            h = sum(shuffled) / self.shuffle_ind

        h = nn.functional.dropout(h, p=self.dropout, training=self.training)
        h = h + node_features  # residual
        h = self._apply_norm(self.norm2, h, batch)
        branch_outputs.append(h)

        # --- Combine branches + feed-forward ---
        out = sum(branch_outputs)
        out = out + self.mlp(out)
        out = self._apply_norm(self.norm3, out, batch)

        return out


class Encoder(torch.nn.Module):
    """
    Multi-layer graph encoder using GPSConv (local GCNConv + global Mamba).

    Every GPSConv layer runs at a fixed `hidden_channels` width (required by
    GPSConv's residual connections). Width schedule:
      - 1 layer:  in_channels -> [GPSConv @ out_channels] -> out_channels
      - k layers: in_channels -> [k x GPSConv @ 2*out_channels] -> Linear -> out_channels

    Args:
        in_channels:      Input feature size.
        out_channels:     Output embedding size.
        activation:       Activation applied after each GPSConv.
        base_model:       Local MPNN constructor (default: GCNConv), must accept
                           (channels, channels).
        num_layers:       Total number of GPSConv layers.
        dropout:          Dropout rate inside each GPSConv.
        order_by_degree:  Sort nodes by degree before Mamba.
        shuffle_ind:      Number of random permutation averages (0 = none).
        d_state:          Mamba SSM state size.
        d_conv:           Mamba conv kernel size.
        bidirectional:    Give every GPSConv layer a backward Mamba scan too
                           (see `GPSConv`).
    """

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

        # Project raw node features into the working width
        self.input_proj = (
            nn.Identity()
            if in_channels == hidden_channels
            else nn.Linear(in_channels, hidden_channels)
        )

        def _make_gps(channels: int) -> GPSConv:
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

        # All GPSConv layers keep the same width (hidden_channels); narrowing
        # to out_channels, when needed, happens once via `output_proj` below.
        self.gps_layers = nn.ModuleList(
            _make_gps(hidden_channels) for _ in range(num_layers)
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
        """
        Args:
            node_features: (N, in_channels)
            edge_index:    (2, E)
            batch:         (N,)  graph-assignment vector
        Returns:
            Node embeddings of shape (N, out_channels)
        """
        node_emb = self.input_proj(node_features)

        for layer in self.gps_layers:
            node_emb = self.activation(layer(node_emb, edge_index, batch))

        if self._needs_output_proj:
            node_emb = self.output_proj(node_emb)

        return node_emb


class _ProjectionMixin(torch.nn.Module):
    """
    Shared projection head and cosine-similarity helpers for MV/SV models.

    Args:
        num_hidden:       Dimensionality of the GNN encoder output.
        num_proj_hidden:  Dimensionality of the projection head.
        tau:              Temperature parameter for NT-Xent loss.

    Returns:
        Projection head and similarity helpers for contrastive loss computation.
    """

    def __init__(self, num_hidden: int, num_proj_hidden: int, tau: float):
        super().__init__()
        self.tau = tau
        self.forward_proj_1 = nn.Linear(num_hidden, num_proj_hidden)
        self.forward_proj_2 = nn.Linear(num_proj_hidden, num_hidden)

    def projection(self, gnn_embedding: Tensor) -> Tensor:
        projected = nn.functional.elu(self.forward_proj_1(gnn_embedding))
        return self.forward_proj_2(projected)

    def similarity_matrix(self, emb_a: Tensor, emb_b: Tensor) -> Tensor:
        """Normalised dot-product similarity matrix."""
        return torch.mm(
            nn.functional.normalize(emb_a), nn.functional.normalize(emb_b).t()
        )

    def tau_scaling(self, sim: Tensor) -> Tensor:
        """Temperature-scaled exponential (NT-Xent numerator/denominator)."""
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
        """2 * |N_i| + 1  (intra + inter neighbours + self inter-view)."""
        return torch.sum(adj, 1).mul(2).add(1).squeeze()


class MVmodel(_ProjectionMixin):
    """
    Multi-view contrastive model with class-biased negative sampling.
    `batch` is required because the Mamba encoder needs it.
    """

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
        pseudo_labels: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Neighbour-aware NT-Xent term. When `pseudo_labels` is given, pairs
        sharing a pseudo-label are excluded from the negative-pair denominator
        (class-biased negative sampling); otherwise all non-self pairs count.
        """
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
    """
    Single-view contrastive model.
    `batch` is required because the Mamba encoder needs it.
    """

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
        sample_mask: Optional[Tensor] = None,
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
        sample_mask: Optional[Tensor] = None,
        mean: bool = True,
    ) -> Tensor:
        per_node = (
            self._neighbor_contrastive_term(emb_a, emb_b, adj, sample_mask)
            + self._neighbor_contrastive_term(emb_b, emb_a, adj, sample_mask)
        ) * 0.5
        return self._reduce(per_node, mean)


def drop_feature(node_features: Tensor, drop_prob: float) -> Tensor:
    """Randomly zeros out feature dimensions with probability `drop_prob`."""
    drop_mask = (
        torch.empty(node_features.size(1), device=node_features.device).uniform_(0, 1)
        < drop_prob
    )
    node_features = node_features.clone()
    node_features[:, drop_mask] = 0
    return node_features


def filter_adj(
    row: Tensor, col: Tensor, edge_attr: OptTensor, keep_mask: Tensor
) -> Tuple[Tensor, Tensor, OptTensor]:
    filtered_attr = None if edge_attr is None else edge_attr[keep_mask]
    return row[keep_mask], col[keep_mask], filtered_attr


def dropout_adj(
    edge_index: Tensor,
    edge_attr: Tensor,
    force_undirected: bool = False,
    training: bool = True,
) -> Tuple[Tensor, Tensor]:
    """
    Probability-weighted edge dropout: edge (u, v) is kept with probability
    `1 - edge_attr[u, v]`. All work happens on `edge_index`/`edge_attr`'s own
    device — callers are responsible for placing those tensors correctly.
    """
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
) -> Tuple[Tensor, Tensor]:
    """Thin wrapper around `dropout_adj` used to build each contrastive view."""
    if not training:
        return edge_index, edge_attr
    return dropout_adj(edge_index, edge_attr, force_undirected=force_undirected)
