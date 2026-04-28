# Standard Library
import inspect
from typing import Any, Dict, Optional, Tuple

# PyTorch & Deep Learning
import torch
import torch.nn as nn
from mamba_ssm import Mamba

# PyTorch Geometric (Graph Neural Networks)
from torch_geometric.nn import GCNConv, MessagePassing
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


class GPSConv(torch.nn.Module):
    """
    GPS-style layer combining a local MPNN with a global Mamba SSM.

    Args:
        channels:         Node feature dimensionality.
        conv:             Local message-passing layer (e.g. GCNConv).
        dropout:          Dropout applied after each sub-layer.
        act:              Activation name for the feed-forward MLP.
        act_kwargs:       Extra kwargs forwarded to the activation resolver.
        norm:             Normalisation layer name (or None).
        norm_kwargs:      Extra kwargs forwarded to the normalisation resolver.
        order_by_degree:  Sort nodes by degree before feeding into Mamba.
        shuffle_ind:      Number of random permutations to average (0 = no shuffle).
        d_state:          Mamba SSM state size.
        d_conv:           Mamba conv kernel size.
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

        self.mamba = Mamba(d_model=channels, d_state=d_state, d_conv=d_conv, expand=1)

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
        for norm in (self.norm1, self.norm2, self.norm3):
            if norm is not None:
                norm.reset_parameters()

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
            h = h + node_features  # residual
            h = self._apply_norm(self.norm1, h, batch)
            branch_outputs.append(h)

        # --- Global Mamba branch ---
        x = node_features
        if self.order_by_degree:
            deg = degree(edge_index[0], x.size(0)).to(torch.long)
            order_tensor = torch.stack([batch, deg], dim=1).T
            _, x = sort_edge_index(order_tensor, edge_attr=x)

        if self.shuffle_ind == 0:
            dense, mask = to_dense_batch(x, batch)
            h = self.mamba(dense)[mask]
        else:
            shuffled = []
            for _ in range(self.shuffle_ind):
                perm = _permute_within_batch(x, batch)
                dense, mask = to_dense_batch(x[perm], batch)
                h_i = self.mamba(dense)[mask][perm]
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

    The width schedule mirrors the original:
      - 1 layer:  in_channels -> out_channels
      - k layers: in_channels -> 2*out_channels (hidden) -> out_channels

    Args:
        in_channels:      Input feature size.
        out_channels:     Output embedding size.
        activation:       Activation applied after each GPSConv.
        base_model:       Local MPNN constructor (default: GCNConv).
        num_layers:       Total number of GPSConv layers.
        dropout:          Dropout rate inside each GPSConv.
        order_by_degree:  Sort nodes by degree before Mamba.
        shuffle_ind:      Number of random permutation averages (0 = none).
        d_state:          Mamba SSM state size.
        d_conv:           Mamba conv kernel size.
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

        def _make_gps(in_ch: int, out_ch: int) -> GPSConv:
            return GPSConv(
                channels=in_ch,
                conv=base_model(in_ch, out_ch),
                dropout=dropout,
                order_by_degree=order_by_degree,
                shuffle_ind=shuffle_ind,
                d_state=d_state,
                d_conv=d_conv,
            )

        gps_layers: list[nn.Module] = []

        if num_layers == 1:
            # Single layer: GPSConv input/output both at out_channels
            gps_layers.append(_make_gps(out_channels, out_channels))
        else:
            # Hidden layers stay at 2*out_channels
            for _ in range(num_layers - 1):
                gps_layers.append(_make_gps(2 * out_channels, 2 * out_channels))
            # Final layer: local conv narrows to out_channels;
            # GPSConv still outputs `channels` (= 2*out_channels) so we
            # add a linear readout to squeeze to out_channels.
            gps_layers.append(_make_gps(2 * out_channels, out_channels))
            self.output_proj = nn.Linear(2 * out_channels, out_channels)

        self.gps_layers = nn.ModuleList(gps_layers)
        self._needs_output_proj = num_layers > 1

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
    Multi-view contrastive model.
    `batch` is now a required argument because the Mamba encoder needs it.
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
        projected_embedding = self.projection(gnn_embedding)
        return projected_embedding

    # -- Basic (non-neighbour-aware) contrastive loss -----------------------

    def _semi_loss(self, emb_a: Tensor, emb_b: Tensor) -> Tensor:
        self_sim = self.tau_scaling(self.similarity_matrix(emb_a, emb_a))
        cross_sim = self.tau_scaling(self.similarity_matrix(emb_a, emb_b))
        return -torch.log(
            cross_sim.diag() / (self_sim.sum(1) + cross_sim.sum(1) - self_sim.diag())
        )

    def loss(
        self,
        proj_emb_1: Tensor,
        proj_emb_2: Tensor,
        mean: bool = True,
        batch_size: int = 0,
    ) -> Tensor:
        per_node = (
            self._semi_loss(proj_emb_1, proj_emb_2)
            + self._semi_loss(proj_emb_2, proj_emb_1)
        ) * 0.5
        return self._reduce(per_node, mean)

    # -- Neighbour-aware contrastive loss -----------------------------------

    def _neighbor_contrastive_loss(
        self, emb_a: Tensor, emb_b: Tensor, adj: Tensor
    ) -> Tensor:
        adj = self._strip_self_loops(adj)
        positive_pair_counts = self._positive_pair_counts(adj)

        intra_sim = self.tau_scaling(self.similarity_matrix(emb_a, emb_a))
        inter_sim = self.tau_scaling(self.similarity_matrix(emb_a, emb_b))

        numerator = (
            inter_sim.diag() + intra_sim.mul(adj).sum(1) + inter_sim.mul(adj).sum(1)
        )
        denominator = intra_sim.sum(1) + inter_sim.sum(1) - intra_sim.diag()

        return -torch.log((numerator / denominator) / positive_pair_counts)

    def contrastive_loss(
        self, emb_a: Tensor, emb_b: Tensor, adj: Tensor, mean: bool = True
    ) -> Tensor:
        per_node = (
            self._neighbor_contrastive_loss(emb_a, emb_b, adj)
            + self._neighbor_contrastive_loss(emb_b, emb_a, adj)
        ) * 0.5
        return self._reduce(per_node, mean)

    # -- Biased neighbour-aware contrastive loss ----------------------------

    def _neighbor_contrastive_loss_biased(
        self, emb_a: Tensor, emb_b: Tensor, adj: Tensor, pseudo_labels: Tensor
    ) -> Tensor:
        adj = self._strip_self_loops(adj)
        positive_pair_counts = self._positive_pair_counts(adj)

        intra_sim = self.tau_scaling(self.similarity_matrix(emb_a, emb_a))
        inter_sim = self.tau_scaling(self.similarity_matrix(emb_a, emb_b))

        negative_mask = (pseudo_labels.view(-1, 1) != pseudo_labels.view(1, -1)).float()
        masked_intra_sim = intra_sim * negative_mask
        masked_inter_sim = inter_sim * negative_mask

        numerator = (
            inter_sim.diag() + intra_sim.mul(adj).sum(1) + inter_sim.mul(adj).sum(1)
        )
        denominator = (
            masked_intra_sim.sum(1) + masked_inter_sim.sum(1) - intra_sim.diag()
        )

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
            self._neighbor_contrastive_loss_biased(emb_a, emb_b, adj, pseudo_labels)
            + self._neighbor_contrastive_loss_biased(emb_b, emb_a, adj, pseudo_labels)
        ) * 0.5
        return self._reduce(per_node, mean)


class SVmodel(_ProjectionMixin):
    """
    Single-view contrastive model.
    `batch` is now a required argument because the Mamba encoder needs it.
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
        projected_embedding = self.projection(gnn_embedding)
        return projected_embedding

    def _neighbor_contrastive_loss(
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
            self._neighbor_contrastive_loss(emb_a, emb_b, adj, sample_mask)
            + self._neighbor_contrastive_loss(emb_b, emb_a, adj, sample_mask)
        ) * 0.5
        return self._reduce(per_node, mean)


def drop_feature(node_features: Tensor, drop_prob: float) -> Tensor:
    """Randomly zeros out feature dimensions with probability `drop_prob`."""
    drop_mask = (
        torch.empty(node_features.size(1), device=torch.device("cpu")).uniform_(0, 1)
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
    num_nodes: Optional[int] = None,
    training: bool = True,
) -> Tuple[Tensor, Tensor]:
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

    keep_mask = torch.rand(
        edge_attr.size(0), device=torch.device("cpu")
    ) >= edge_attr.to("cpu")
    row, col, edge_attr = filter_adj(row, col, edge_attr, keep_mask)

    if force_undirected:
        edge_index = torch.stack([torch.cat([row, col]), torch.cat([col, row])], dim=0)
    else:
        edge_index = torch.stack([row, col], dim=0)

    return edge_index, edge_attr


def multiple_dropout_average(
    edge_index: Tensor,
    edge_attr: Tensor,
    num_trials: int = 10,
    force_undirected: bool = False,
    num_nodes: Optional[int] = None,
    threshold_ratio: float = 0.5,
    training: bool = True,
    device: str = "cuda",
) -> Tuple[Tensor, Tensor]:
    if not training:
        return edge_index, edge_attr

    if num_nodes is None:
        num_nodes = int(edge_index.max().item()) + 1

    edge_index = edge_index.to(device)
    edge_attr = edge_attr.to(device)

    use_simulation = False
    if use_simulation:
        edge_count = torch.zeros(
            (num_nodes, num_nodes), dtype=torch.int32, device=device
        )
        for _ in range(num_trials):
            dropped_edge_index, _ = dropout_adj(edge_index, edge_attr, force_undirected)
            dropped_edge_index = dropped_edge_index.to(device)
            src, dst = dropped_edge_index
            edge_count[src, dst] += 1
            if force_undirected:
                edge_count[dst, src] += 1
        threshold = int(num_trials * threshold_ratio)
        final_edge_index = (edge_count >= threshold).nonzero().t().contiguous()
    else:
        final_edge_index, _ = dropout_adj(edge_index, edge_attr, force_undirected)

    return final_edge_index, edge_attr


def random_dropout_adj(
    edge_index: Tensor,
    edge_attr: OptTensor = None,
    p: float = 0.5,
    force_undirected: bool = False,
    num_nodes: Optional[int] = None,
    training: bool = True,
) -> Tuple[Tensor, OptTensor]:
    if not 0.0 <= p <= 1.0:
        raise ValueError(f"Dropout probability has to be between 0 and 1 (got {p})")

    if not training or p == 0.0:
        return edge_index, edge_attr

    row, col = edge_index
    keep_mask = torch.rand(row.size(0), device=torch.device("cpu")) >= p

    if force_undirected:
        keep_mask[row > col] = False

    row, col, edge_attr = filter_adj(row, col, edge_attr, keep_mask)

    if force_undirected:
        edge_index = torch.stack([torch.cat([row, col]), torch.cat([col, row])], dim=0)
        if edge_attr is not None:
            edge_attr = torch.cat([edge_attr, edge_attr], dim=0)
    else:
        edge_index = torch.stack([row, col], dim=0)

    return edge_index, edge_attr


# Discriminator for DGI-style loss (not used in main training loop)
class Discriminator(nn.Module):
    _NUM_LAYERS = 1
    _HIDDEN_DIM = 64
    _DROPOUT = 0.2
    _INPUT_DROPOUT = 0.1

    def __init__(self, input_dim: int):
        super().__init__()

        layers: list[nn.Module] = [nn.Dropout(self._INPUT_DROPOUT)]
        for i in range(self._NUM_LAYERS + 1):
            layer_in = input_dim if i == 0 else self._HIDDEN_DIM
            layer_out = 1 if i == self._NUM_LAYERS else self._HIDDEN_DIM
            layers.append(nn.Linear(layer_in, layer_out))
            if i < self._NUM_LAYERS:
                layers += [nn.ReLU(), nn.Dropout(self._DROPOUT)]

        self.layers = nn.Sequential(*layers)

    def forward(self, node_features: Tensor) -> Tensor:
        return self.layers(node_features).view(-1)
