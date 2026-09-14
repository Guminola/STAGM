import argparse
from time import perf_counter

import numpy as np
import scanpy as sc
import squidpy as sq
import torch
from scipy.sparse import csr_matrix, issparse
from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.metrics import davies_bouldin_score, silhouette_score
from torch import nn

from torch_geometric.data import Data
from torch_geometric.loader import NeighborLoader
from torch_geometric.nn import GCNConv
from tqdm import tqdm

from .clustering import clustering
from .model import Encoder, MVmodel, SVmodel, drop_feature, multiple_dropout_average

_ACTIVATIONS = {"relu": nn.functional.relu, "prelu": nn.PReLU()}

_SUPPORTED_CLUSTER_TOOLS = ("leiden", "louvain")


def generate_pseudo_labels(img_emb: np.ndarray, n_clusters: int = 300) -> torch.Tensor:
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(img_emb)
    return torch.tensor(kmeans.labels_)


def _to_dense_array(matrix) -> np.ndarray:
    return matrix.toarray() if issparse(matrix) else np.asarray(matrix)


def _sparse_to_edge_index(adj) -> torch.Tensor:
    adj = adj.tocoo()
    row = torch.from_numpy(adj.row.astype(np.int64))
    col = torch.from_numpy(adj.col.astype(np.int64))
    return torch.stack([row, col], dim=0)


def adj_to_edge_index(adj) -> torch.Tensor:
    if issparse(adj):
        return _sparse_to_edge_index(adj)
    if torch.is_tensor(adj):
        row, col = torch.where(adj != 0)
        return torch.stack([row, col], dim=0)
    return _sparse_to_edge_index(csr_matrix(adj))


def convert_edge_probabilities(
    adj_matrix: torch.Tensor, edge_prob_matrix: torch.Tensor
) -> torch.Tensor:
    row, col = torch.where(adj_matrix != 0)
    return edge_prob_matrix[row, col]


class STAGM:
    def __init__(
        self,
        args: argparse.Namespace,
        config: argparse.Namespace,
    ) -> None:
        self.args = args
        self.config = config

        # Model architecture
        self.num_gene: int = getattr(config, "num_gene", 3000)
        self.num_hidden: int = getattr(config, "num_hidden", 128)
        self.num_proj_hidden: int = getattr(config, "num_proj_hidden", 128)
        self.num_layers: int = getattr(config, "num_layers", 2)
        self.num_clusters: int = getattr(config, "num_clusters", 7)
        self.k: int | None = getattr(config, "k", None)
        self.activation = _ACTIVATIONS[getattr(config, "activation", "relu")]

        # Encoder options
        self.base_model = getattr(config, "base_model", GCNConv)
        self.dropout: float = getattr(config, "dropout", 0.0)
        self.shuffle_ind: bool = getattr(config, "shuffle_ind", False)
        self.d_state: int = getattr(config, "d_state", 16)
        self.d_conv: int = getattr(config, "d_conv", 4)
        self.order_by_degree: bool = getattr(config, "order_by_degree", False)
        self.bidirectional: bool = getattr(config, "bidirectional", True)

        # Training hyperparameters
        self.learning_rate: float = getattr(config, "learning_rate", 1e-3)
        self.weight_decay: float = getattr(config, "weight_decay", 0.0)
        self.num_epochs: int = getattr(config, "num_epochs", 200)
        self.tau: float = getattr(config, "tau", 0.5)
        self.drop_feature_rate_1: float = getattr(config, "drop_feature_rate_1", 0.3)
        self.drop_feature_rate_2: float = getattr(config, "drop_feature_rate_2", 0.4)

        # Experiment / runtime options
        self.single: bool = getattr(config, "single", False)
        self.refine: bool = getattr(config, "refine", True)
        self.checkpoint_path: str = getattr(config, "checkpoint_path", "model.pt")

        # Mini-batch options
        self.use_minibatch: bool | None = getattr(config, "use_minibatch", None)
        self.minibatch_size: int = getattr(config, "minibatch_size", 4096)
        self.minibatch_num_neighbors: list[int] | None = getattr(
            config, "minibatch_num_neighbors", None
        )
        self.minibatch_node_threshold: int = getattr(
            config, "minibatch_node_threshold", 20_000
        )

        # Clustering options
        self.radius: int = getattr(config, "radius", 15)
        self.tool: str = getattr(config, "tool", "leiden")
        self.mask_slices: bool = getattr(config, "mask_slices", True)

        # Progress-bar style
        self.bar_format: str = getattr(
            config,
            "bar_format",
            "{l_bar}{bar}| [{elapsed}<{remaining}, {rate_fmt}{postfix}]",
        )

        # Derived / runtime state
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.adata = None
        self._loss_curve: list[float] = []
        self._training_time_seconds: float | None = None

        # Model, projection head, optimizer
        self.encoder = Encoder(
            in_channels=self.num_gene,
            out_channels=self.num_hidden,
            activation=self.activation,
            base_model=self.base_model,
            num_layers=self.num_layers,
            dropout=self.dropout,
            order_by_degree=self.order_by_degree,
            shuffle_ind=self.shuffle_ind,
            d_state=self.d_state,
            d_conv=self.d_conv,
            bidirectional=self.bidirectional,
        ).to(self.device)

        model_cls = SVmodel if self.single else MVmodel
        self.model = model_cls(
            self.encoder,
            self.num_hidden,
            self.num_proj_hidden,
            self.tau,
        ).to(self.device)

        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )

    def _load_graph_structure(self) -> tuple[object, torch.Tensor]:
        if self.adata is None:
            raise ValueError("adata not loaded!")
        graph_neigh_raw = self.adata.obsm["graph_neigh"]
        edge_index = adj_to_edge_index(graph_neigh_raw).to(self.device)
        return graph_neigh_raw, edge_index

    def _prepare_graph_tensors(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        features_matrix = torch.FloatTensor(self.adata.obsm["feat"].copy()).to(
            self.device
        )
        graph_neigh_raw, edge_index = self._load_graph_structure()
        graph_neigh = torch.FloatTensor(_to_dense_array(graph_neigh_raw)).to(
            self.device
        )
        batch = torch.zeros(
            features_matrix.size(0), dtype=torch.long, device=self.device
        )
        return features_matrix, graph_neigh, edge_index, batch

    def _positive_pseudo_labels(self, to_device: bool = True) -> torch.Tensor:
        if "pseudo_labels" in self.adata.obs:
            pseudo_labels = torch.as_tensor(self.adata.obs["pseudo_labels"].cat.codes)
        elif self.k:
            pseudo_labels = generate_pseudo_labels(self.adata.obsm["img_emb"], self.k)
        else:
            pseudo_labels = generate_pseudo_labels(self.adata.obsm["img_emb"])
        return pseudo_labels.to(self.device) if to_device else pseudo_labels

    def train(self) -> None:
        n_nodes = self.adata.obsm["feat"].shape[0]
        use_minibatch = self.use_minibatch
        if use_minibatch is None:
            use_minibatch = n_nodes > self.minibatch_node_threshold

        if use_minibatch:
            print(f"=== train (mini-batch, {n_nodes} nodes) ===")
            self._train_minibatch()
        else:
            self._train_full_batch()

    def _train_full_batch(self) -> None:
        features_matrix, graph_neigh, edge_index, batch = self._prepare_graph_tensors()

        edge_probabilities = torch.FloatTensor(
            _to_dense_array(self.adata.obsm["edge_probabilities"])
        ).to(self.device)
        edge_probs = convert_edge_probabilities(graph_neigh, edge_probabilities)

        mask_neigh = None
        pseudo_labels = None
        if self.single:
            if "mask_neigh" in self.adata.obsm and self.mask_slices:
                print("Consider intra slice")
                mask_neigh = torch.FloatTensor(
                    _to_dense_array(self.adata.obsm["mask_neigh"])
                ).to(self.device)
        else:
            pseudo_labels = self._positive_pseudo_labels()

        self._loss_curve = []
        train_start = perf_counter()

        print("=== train ===")
        for _ in tqdm(range(1, self.num_epochs + 1), bar_format=self.bar_format):
            self.model.train()
            self.optimizer.zero_grad()

            edge_index_1, _ = multiple_dropout_average(
                edge_index, edge_probs, force_undirected=True
            )
            edge_index_2, _ = multiple_dropout_average(
                edge_index, edge_probs, force_undirected=True
            )
            x_1 = drop_feature(features_matrix, self.drop_feature_rate_1)
            x_2 = drop_feature(features_matrix, self.drop_feature_rate_2)

            z1 = self.model(x_1, edge_index_1, batch)
            z2 = self.model(x_2, edge_index_2, batch)

            if self.single:
                loss = self.model.contrastive_loss(
                    z1, z2, graph_neigh, sample_mask=mask_neigh
                )
            else:
                loss = self.model.contrastive_loss_biased(
                    z1, z2, graph_neigh, pseudo_labels
                )

            loss.backward()
            self.optimizer.step()
            self._loss_curve.append(loss.item())

        self._training_time_seconds = perf_counter() - train_start
        torch.save(self.model.state_dict(), self.checkpoint_path)

    def _minibatch_loader(self) -> NeighborLoader:
        features_matrix = torch.FloatTensor(self.adata.obsm["feat"].copy())
        edge_index = adj_to_edge_index(self.adata.obsm["graph_neigh"])
        data = Data(
            x=features_matrix, edge_index=edge_index, num_nodes=features_matrix.size(0)
        )
        num_neighbors = self.minibatch_num_neighbors or ([-1] * self.num_layers)
        return NeighborLoader(
            data,
            num_neighbors=num_neighbors,
            batch_size=self.minibatch_size,
            shuffle=True,
        )

    def _train_minibatch(self) -> None:
        loader = self._minibatch_loader()

        edge_probabilities_raw = self.adata.obsm["edge_probabilities"]
        edge_probabilities = (
            edge_probabilities_raw.tocsr()
            if issparse(edge_probabilities_raw)
            else csr_matrix(edge_probabilities_raw)
        )

        pseudo_labels_full = None
        if not self.single:
            pseudo_labels_full = self._positive_pseudo_labels(to_device=False)

        same_slice = None
        if self.single and self.mask_slices and "batch" in self.adata.obs:
            same_slice = self.adata.obs["batch"].astype(str).to_numpy()

        self._loss_curve = []
        train_start = perf_counter()

        for _ in tqdm(range(1, self.num_epochs + 1), bar_format=self.bar_format):
            self.model.train()
            epoch_losses = []

            for sub in loader:
                self.optimizer.zero_grad()

                n_id = sub.n_id.numpy()
                num_seed = sub.batch_size

                global_row = n_id[sub.edge_index[0].numpy()]
                global_col = n_id[sub.edge_index[1].numpy()]
                edge_probs_sub = torch.as_tensor(
                    np.asarray(edge_probabilities[global_row, global_col]).ravel(),
                    dtype=torch.float32,
                    device=self.device,
                )

                x = sub.x.to(self.device)
                sub_edge_index = sub.edge_index.to(self.device)
                sub_batch = torch.zeros(x.size(0), dtype=torch.long, device=self.device)

                seed_mask = (sub_edge_index[0] < num_seed) & (
                    sub_edge_index[1] < num_seed
                )
                seed_row = sub_edge_index[0][seed_mask]
                seed_col = sub_edge_index[1][seed_mask]
                adj = torch.zeros((num_seed, num_seed), device=self.device)
                adj[seed_row, seed_col] = 1.0

                mask_neigh = None
                pseudo_labels = None
                if self.single:
                    if same_slice is not None:
                        seed_slices = same_slice[n_id[:num_seed]]
                        mask_neigh = torch.as_tensor(
                            (seed_slices[:, None] == seed_slices[None, :]).astype(
                                np.float32
                            ),
                            device=self.device,
                        )
                else:
                    pseudo_labels = pseudo_labels_full[n_id[:num_seed]].to(self.device)

                edge_index_1, _ = multiple_dropout_average(
                    sub_edge_index, edge_probs_sub, force_undirected=True
                )
                edge_index_2, _ = multiple_dropout_average(
                    sub_edge_index, edge_probs_sub, force_undirected=True
                )
                x_1 = drop_feature(x, self.drop_feature_rate_1)
                x_2 = drop_feature(x, self.drop_feature_rate_2)

                z1_full = self.model(x_1, edge_index_1, sub_batch)
                z2_full = self.model(x_2, edge_index_2, sub_batch)
                z1 = z1_full[:num_seed]
                z2 = z2_full[:num_seed]

                if self.single:
                    loss = self.model.contrastive_loss(
                        z1, z2, adj, sample_mask=mask_neigh
                    )
                else:
                    loss = self.model.contrastive_loss_biased(
                        z1, z2, adj, pseudo_labels
                    )

                loss.backward()
                self.optimizer.step()
                epoch_losses.append(loss.item())

            self._loss_curve.append(float(np.mean(epoch_losses)))

        self._training_time_seconds = perf_counter() - train_start
        torch.save(self.model.state_dict(), self.checkpoint_path)

    def eva(self) -> None:
        print("=== load ===")
        self.model.load_state_dict(torch.load(self.checkpoint_path))
        self.model.eval()

        features_matrix = torch.FloatTensor(self.adata.obsm["feat"].copy()).to(
            self.device
        )
        _, edge_index = self._load_graph_structure()
        batch = torch.zeros(
            features_matrix.size(0), dtype=torch.long, device=self.device
        )

        with torch.no_grad():
            self.adata.obsm["emb"] = (
                self.model(features_matrix, edge_index, batch).detach().cpu().numpy()
            )

        self._print_diagnostics()
        print("embedding generated, go clustering")

    def _print_diagnostics(self) -> None:
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(
            p.numel() for p in self.model.parameters() if p.requires_grad
        )

        param_bytes = sum(p.numel() * p.element_size() for p in self.model.parameters())
        buffer_bytes = sum(b.numel() * b.element_size() for b in self.model.buffers())
        model_mb = (param_bytes + buffer_bytes) / 1024**2

        if self.device.type == "cuda":
            peak_gpu_mb = torch.cuda.max_memory_allocated(self.device) / 1024**2
            torch.cuda.reset_peak_memory_stats(self.device)
        else:
            peak_gpu_mb = None

        loss_curve = self._loss_curve
        if loss_curve:
            loss_summary = (
                f"  first={loss_curve[0]:.4f}  "
                f"last={loss_curve[-1]:.4f}  "
                f"min={min(loss_curve):.4f}  "
                f"max={max(loss_curve):.4f}"
            )
        else:
            loss_summary = "  (no loss history — train() was not called this session)"

        emb = self.adata.obsm["emb"]
        emb_norm = np.linalg.norm(emb, axis=1)

        print("\n========== Model Diagnostics ==========")
        print(f"  Total parameters:     {total_params:,}")
        print(f"  Trainable parameters: {trainable_params:,}")
        print(f"  Model weight size:    {model_mb:.2f} MB")
        if peak_gpu_mb is not None:
            print(f"  Peak GPU memory:      {peak_gpu_mb:.2f} MB")
        if self._training_time_seconds is not None:
            mins, secs = divmod(self._training_time_seconds, 60)
            print(
                f"  Training time:        {int(mins)}m {secs:.1f}s  "
                f"({self._training_time_seconds:.1f}s total)"
            )
            if loss_curve:
                print(
                    "  Time per epoch:       "
                    f"{self._training_time_seconds / len(loss_curve) * 1000:.1f} ms"
                )
        print(f"\n  Loss curve:          {loss_summary}")
        print(f"\n  Embedding shape:      {emb.shape}")
        print(
            f"  Embedding norm — mean={emb_norm.mean():.4f}  "
            f"std={emb_norm.std():.4f}  "
            f"min={emb_norm.min():.4f}  "
            f"max={emb_norm.max():.4f}"
        )
        print("=======================================\n")

    def cluster(self, label: bool = True) -> None:
        if self.tool not in _SUPPORTED_CLUSTER_TOOLS:
            raise ValueError(
                f"Unsupported clustering tool '{self.tool}'. "
                f"Choose one of {_SUPPORTED_CLUSTER_TOOLS}."
            )

        clustering(
            self.adata,
            self.num_clusters,
            radius=self.radius,
            key="emb",
            method=self.tool,
            start=0.01,
            end=0.27,
            increment=0.005,
            refinement=self.refine,
        )

        if label:
            print("calculate metric ARI")
            ari = metrics.adjusted_rand_score(
                self.adata.obs["domain"], self.adata.obs["ground_truth"]
            )
            nmi = metrics.normalized_mutual_info_score(
                self.adata.obs["domain"], self.adata.obs["ground_truth"]
            )
            self.adata.uns["ari"] = ari
            self.adata.uns["nmi"] = nmi
            print("ARI:", ari)
            print("NMI:", nmi)
        else:
            print("calculate SC and DB")
            sc_score = silhouette_score(
                self.adata.obsm["emb"], self.adata.obs["domain"]
            )
            db_score = davies_bouldin_score(
                self.adata.obsm["emb"], self.adata.obs["domain"]
            )
            self.adata.uns["sc"] = sc_score
            self.adata.uns["db"] = db_score
            print("SC:", sc_score)
            print("DB:", db_score)

    def draw_spatial(self, p: str = "") -> None:
        sq.pl.spatial_scatter(
            self.adata,
            size=1.2,
            figsize=(7, 7),
            color=["ground_truth", "domain"],
            save=p + str(self.args.slide) + ".png",
        )

    def draw_single_spatial(self) -> None:
        sq.pl.spatial_scatter(
            self.adata,
            color="domain",
            size=100,
            save=str(self.args.slide) + ".png",
        )

    def draw_xenium_spatial(
        self,
        color: str | list[str] = "domain",
        size: float = 2.0,
        dpi: int = 200,
        p: str = "",
    ) -> None:
        n_cells = self.adata.n_obs
        if n_cells > 200_000:
            print(
                f"{n_cells:,} cells — rendering may be slow; consider "
                "subsetting self.adata before calling this, or use "
                "napari-spatialdata for interactive large-scale viewing."
            )

        sq.pl.spatial_scatter(
            self.adata,
            shape=None,
            color=color,
            size=size,
            dpi=dpi,
            figsize=(7, 7),
            save=p + str(self.args.slide) + "_xenium.png",
        )

    def draw_umap(self) -> None:
        print("start umap")
        sc.pp.neighbors(self.adata, use_rep="emb")
        sc.tl.umap(self.adata)
        sc.pl.umap(
            self.adata,
            color="domain",
            show=True,
            save=str(self.args.slide) + "domain.pdf",
        )
        if getattr(self.args, "label", False):
            sc.pl.umap(
                self.adata,
                color="ground_truth",
                show=True,
                save=str(self.args.slide) + "_label.pdf",
            )

    def draw_horizontal(self) -> None:
        adata_batch_0 = self.adata[self.adata.obs["batch"] == "0", :]
        sq.pl.spatial_scatter(adata_batch_0, color="domain", save="0.png")

        adata_batch_1 = self.adata[self.adata.obs["batch"] == "1", :]
        sq.pl.spatial_scatter(adata_batch_1, color="domain", save="1.png")
