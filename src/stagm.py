# Standard Library
import argparse
from time import perf_counter

# Scientific Stack & General Data Processing
import numpy as np

# Bioinformatics & Spatial Analysis
import scanpy as sc
import squidpy as sq

# PyTorch & Deep Learning
import torch

# Scikit-learn: Metrics, Decomposition, and Preprocessing
from scipy.sparse import issparse
from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.metrics import davies_bouldin_score, silhouette_score
from torch import nn

# PyTorch Geometric (Graph Neural Networks)
from torch_geometric.nn import GCNConv
from tqdm import tqdm

from .clustering import clustering

# STAGM-specific modules
from .model import Encoder, MVmodel, SVmodel, drop_feature, multiple_dropout_average

# Only GCNConv is used as the local-MPNN branch inside GPSConv
_ACTIVATIONS = {"relu": nn.functional.relu, "prelu": nn.PReLU()}

# clustering.py only implements resolution search for these tools.
_SUPPORTED_CLUSTER_TOOLS = ("leiden", "louvain")


def generate_pseudo_labels(img_emb: np.ndarray, n_clusters: int = 300) -> torch.Tensor:
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(img_emb)
    return torch.tensor(kmeans.labels_)


def _to_dense_array(matrix) -> np.ndarray:
    return matrix.toarray() if issparse(matrix) else np.asarray(matrix)


def adj_to_edge_index(adj: torch.Tensor) -> torch.Tensor:
    row, col = torch.where(adj != 0)
    return torch.stack([row, col], dim=0)


def convert_edge_probabilities(
    adj_matrix: torch.Tensor, edge_prob_matrix: torch.Tensor
) -> torch.Tensor:
    row, col = torch.where(adj_matrix != 0)
    return edge_prob_matrix[row, col]


class STAGM:
    """
    Trains and evaluates a GPSConv (local GCNConv + global Mamba SSM)
    contrastive encoder on spatial transcriptomics data, then clusters the
    resulting embedding via leiden/louvain resolution search.
    """

    def __init__(
        self,
        args: argparse.Namespace,
        config: dict,
        single: bool = False,
        refine: bool = True,
        checkpoint_path: str = "model.pt",
    ) -> None:
        self.args = args
        self.single = single
        self.config = config
        self.checkpoint_path = checkpoint_path

        self.learning_rate: float = config.learning_rate
        self.num_hidden: int = config.num_hidden
        self.num_proj_hidden: int = config.num_proj_hidden
        self.activation = _ACTIVATIONS[config.activation]
        self.num_layers: int = config.num_layers
        self.drop_feature_rate_1: float = config.drop_feature_rate_1
        self.drop_feature_rate_2: float = config.drop_feature_rate_2
        self.tau: float = config.tau
        self.num_epochs: int = config.num_epochs
        self.weight_decay: float = config.weight_decay
        self.num_clusters: int = config.num_clusters
        self.num_gene: int = config.num_gene
        self.refine = refine
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.radius: int = 15
        self.tool: str = "leiden"  # supported: "leiden", "louvain"
        self.mask_slices: bool = True
        self.bar_format: str = (
            "{l_bar}{bar}| [{elapsed}<{remaining}, {rate_fmt}{postfix}]"
        )

        self.adata = None
        self._loss_curve: list[float] = []
        self._training_time_seconds: float | None = None

        self.encoder = Encoder(
            self.num_gene,
            self.num_hidden,
            self.activation,
            base_model=GCNConv,
            num_layers=self.num_layers,
            dropout=self.config.dropout,
            shuffle_ind=self.config.shuffle_ind,
            d_state=self.config.d_state,
            d_conv=self.config.d_conv,
            order_by_degree=self.config.order_by_degree,
            bidirectional=getattr(self.config, "bidirectional", True),
        ).to(self.device)

        model_cls = SVmodel if single else MVmodel
        self.model = model_cls(
            self.encoder, self.num_hidden, self.num_proj_hidden, self.tau
        ).to(self.device)

        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )

    # Shared tensor prep (used by both train() and eva())
    def _prepare_graph_tensors(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Load feature/graph tensors from self.adata onto self.device."""
        if self.adata is None:
            raise ValueError("adata not loaded!")

        features_matrix = torch.FloatTensor(self.adata.obsm["feat"].copy()).to(
            self.device
        )
        graph_neigh = torch.FloatTensor(
            _to_dense_array(self.adata.obsm["graph_neigh"])
        ).to(self.device)
        edge_index = adj_to_edge_index(graph_neigh)
        batch = torch.zeros(
            features_matrix.size(0), dtype=torch.long, device=self.device
        )
        return features_matrix, graph_neigh, edge_index, batch

    def _positive_pseudo_labels(self) -> torch.Tensor:
        """Fetch/derive per-node pseudo labels used by MVmodel's biased loss."""
        if "pseudo_labels" in self.adata.obs:
            pseudo_labels = torch.as_tensor(self.adata.obs["pseudo_labels"].cat.codes)
        elif self.config.k:
            pseudo_labels = generate_pseudo_labels(
                self.adata.obsm["img_emb"], self.config.k
            )
        else:
            pseudo_labels = generate_pseudo_labels(self.adata.obsm["img_emb"])
        return pseudo_labels.to(self.device)

    # Training / evaluation
    def train(self) -> None:
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

    def eva(self) -> None:
        print("=== load ===")
        self.model.load_state_dict(torch.load(self.checkpoint_path))
        self.model.eval()

        features_matrix, _, edge_index, batch = self._prepare_graph_tensors()

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

    # Clustering / metrics
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
            ARI = metrics.adjusted_rand_score(
                self.adata.obs["domain"], self.adata.obs["ground_truth"]
            )
            NMI = metrics.normalized_mutual_info_score(
                self.adata.obs["domain"], self.adata.obs["ground_truth"]
            )
            self.adata.uns["ari"] = ARI
            self.adata.uns["nmi"] = NMI
            print("ARI:", ARI)
            print("NMI:", NMI)
        else:
            print("calculate SC and DB")
            SC = silhouette_score(self.adata.obsm["emb"], self.adata.obs["domain"])
            DB = davies_bouldin_score(self.adata.obsm["emb"], self.adata.obs["domain"])
            self.adata.uns["sc"] = SC
            self.adata.uns["db"] = DB
            print("SC:", SC)
            print("DB:", DB)

    # Visualization
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


# # Standard Library
# import argparse
# from time import perf_counter

# # Scientific Stack & General Data Processing
# import numpy as np

# # Bioinformatics & Spatial Analysis
# import scanpy as sc
# import squidpy as sq

# # PyTorch & Deep Learning
# import torch

# # Scikit-learn: Metrics, Decomposition, and Preprocessing
# from sklearn import metrics
# from sklearn.cluster import KMeans
# from sklearn.metrics import davies_bouldin_score, silhouette_score
# from torch import nn

# # PyTorch Geometric (Graph Neural Networks)
# from torch_geometric.nn import GCNConv
# from tqdm import tqdm

# from .clustering import clustering

# # STAGM-specific modules
# from .model import Encoder, MVmodel, SVmodel, drop_feature, multiple_dropout_average

# # Only GCNConv is used as the local-MPNN branch inside GPSConv
# _ACTIVATIONS = {"relu": nn.functional.relu, "prelu": nn.PReLU()}

# # clustering.py only implements resolution search for these tools.
# _SUPPORTED_CLUSTER_TOOLS = ("leiden", "louvain")


# def generate_pseudo_labels(img_emb: np.ndarray, n_clusters: int = 300) -> torch.Tensor:
#     kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(img_emb)
#     return torch.tensor(kmeans.labels_)


# def adj_to_edge_index(adj: torch.Tensor) -> torch.Tensor:
#     row, col = torch.where(adj != 0)
#     return torch.stack([row, col], dim=0)


# def convert_edge_probabilities(
#     adj_matrix: torch.Tensor, edge_prob_matrix: torch.Tensor
# ) -> torch.Tensor:
#     row, col = torch.where(adj_matrix != 0)
#     return edge_prob_matrix[row, col]


# class STAGM:
#     """
#     Trains and evaluates a GPSConv (local GCNConv + global Mamba SSM)
#     contrastive encoder on spatial transcriptomics data, then clusters the
#     resulting embedding via leiden/louvain resolution search.
#     """

#     def __init__(
#         self,
#         args: argparse.Namespace,
#         config: dict,
#         single: bool = False,
#         refine: bool = True,
#         checkpoint_path: str = "model.pt",
#     ) -> None:
#         self.args = args
#         self.single = single
#         self.config = config
#         self.checkpoint_path = checkpoint_path

#         self.learning_rate: float = config.learning_rate
#         self.num_hidden: int = config.num_hidden
#         self.num_proj_hidden: int = config.num_proj_hidden
#         self.activation = _ACTIVATIONS[config.activation]
#         self.num_layers: int = config.num_layers
#         self.drop_feature_rate_1: float = config.drop_feature_rate_1
#         self.drop_feature_rate_2: float = config.drop_feature_rate_2
#         self.tau: float = config.tau
#         self.num_epochs: int = config.num_epochs
#         self.weight_decay: float = config.weight_decay
#         self.num_clusters: int = config.num_clusters
#         self.num_gene: int = config.num_gene
#         self.refine = refine
#         self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#         self.radius: int = 15
#         self.tool: str = "leiden"  # supported: "leiden", "louvain"
#         self.mask_slices: bool = True
#         self.bar_format: str = (
#             "{l_bar}{bar}| [{elapsed}<{remaining}, {rate_fmt}{postfix}]"
#         )

#         self.adata = None
#         self._loss_curve: list[float] = []
#         self._training_time_seconds: float | None = None

#         self.encoder = Encoder(
#             self.num_gene,
#             self.num_hidden,
#             self.activation,
#             base_model=GCNConv,
#             num_layers=self.num_layers,
#             dropout=self.config.dropout,
#             shuffle_ind=self.config.shuffle_ind,
#             d_state=self.config.d_state,
#             d_conv=self.config.d_conv,
#             order_by_degree=self.config.order_by_degree,
#             bidirectional=getattr(self.config, "bidirectional", True),
#         ).to(self.device)

#         model_cls = SVmodel if single else MVmodel
#         self.model = model_cls(
#             self.encoder, self.num_hidden, self.num_proj_hidden, self.tau
#         ).to(self.device)

#         self.optimizer = torch.optim.Adam(
#             self.model.parameters(),
#             lr=self.learning_rate,
#             weight_decay=self.weight_decay,
#         )

#     # Shared tensor prep (used by both train() and eva())
#     def _prepare_graph_tensors(
#         self,
#     ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
#         """Load feature/graph tensors from self.adata onto self.device."""
#         if self.adata is None:
#             raise ValueError("adata not loaded!")

#         features_matrix = torch.FloatTensor(self.adata.obsm["feat"].copy()).to(
#             self.device
#         )
#         graph_neigh = torch.FloatTensor(self.adata.obsm["graph_neigh"].copy()).to(
#             self.device
#         )
#         edge_index = adj_to_edge_index(graph_neigh)
#         batch = torch.zeros(
#             features_matrix.size(0), dtype=torch.long, device=self.device
#         )
#         return features_matrix, graph_neigh, edge_index, batch

#     def _positive_pseudo_labels(self) -> torch.Tensor:
#         """Fetch/derive per-node pseudo labels used by MVmodel's biased loss."""
#         if "pseudo_labels" in self.adata.obs:
#             pseudo_labels = torch.as_tensor(self.adata.obs["pseudo_labels"].cat.codes)
#         elif self.config.k:
#             pseudo_labels = generate_pseudo_labels(
#                 self.adata.obsm["img_emb"], self.config.k
#             )
#         else:
#             pseudo_labels = generate_pseudo_labels(self.adata.obsm["img_emb"])
#         return pseudo_labels.to(self.device)

#     # Training / evaluation
#     def train(self) -> None:
#         features_matrix, graph_neigh, edge_index, batch = self._prepare_graph_tensors()

#         edge_probabilities = torch.FloatTensor(
#             self.adata.obsm["edge_probabilities"].copy()
#         ).to(self.device)
#         edge_probs = convert_edge_probabilities(graph_neigh, edge_probabilities)

#         mask_neigh = None
#         pseudo_labels = None
#         if self.single:
#             if "mask_neigh" in self.adata.obsm and self.mask_slices:
#                 print("Consider intra slice")
#                 mask_neigh = torch.FloatTensor(self.adata.obsm["mask_neigh"].copy()).to(
#                     self.device
#                 )
#         else:
#             pseudo_labels = self._positive_pseudo_labels()

#         self._loss_curve = []
#         train_start = perf_counter()

#         print("=== train ===")
#         for _ in tqdm(range(1, self.num_epochs + 1), bar_format=self.bar_format):
#             self.model.train()
#             self.optimizer.zero_grad()

#             edge_index_1, _ = multiple_dropout_average(
#                 edge_index, edge_probs, force_undirected=True
#             )
#             edge_index_2, _ = multiple_dropout_average(
#                 edge_index, edge_probs, force_undirected=True
#             )
#             x_1 = drop_feature(features_matrix, self.drop_feature_rate_1)
#             x_2 = drop_feature(features_matrix, self.drop_feature_rate_2)

#             z1 = self.model(x_1, edge_index_1, batch)
#             z2 = self.model(x_2, edge_index_2, batch)

#             if self.single:
#                 loss = self.model.contrastive_loss(
#                     z1, z2, graph_neigh, sample_mask=mask_neigh
#                 )
#             else:
#                 loss = self.model.contrastive_loss_biased(
#                     z1, z2, graph_neigh, pseudo_labels
#                 )

#             loss.backward()
#             self.optimizer.step()
#             self._loss_curve.append(loss.item())

#         self._training_time_seconds = perf_counter() - train_start
#         torch.save(self.model.state_dict(), self.checkpoint_path)

#     def eva(self) -> None:
#         print("=== load ===")
#         self.model.load_state_dict(torch.load(self.checkpoint_path))
#         self.model.eval()

#         features_matrix, _, edge_index, batch = self._prepare_graph_tensors()

#         with torch.no_grad():
#             self.adata.obsm["emb"] = (
#                 self.model(features_matrix, edge_index, batch).detach().cpu().numpy()
#             )

#         self._print_diagnostics()
#         print("embedding generated, go clustering")

#     def _print_diagnostics(self) -> None:
#         total_params = sum(p.numel() for p in self.model.parameters())
#         trainable_params = sum(
#             p.numel() for p in self.model.parameters() if p.requires_grad
#         )

#         param_bytes = sum(p.numel() * p.element_size() for p in self.model.parameters())
#         buffer_bytes = sum(b.numel() * b.element_size() for b in self.model.buffers())
#         model_mb = (param_bytes + buffer_bytes) / 1024**2

#         if self.device.type == "cuda":
#             peak_gpu_mb = torch.cuda.max_memory_allocated(self.device) / 1024**2
#             torch.cuda.reset_peak_memory_stats(self.device)
#         else:
#             peak_gpu_mb = None

#         loss_curve = self._loss_curve
#         if loss_curve:
#             loss_summary = (
#                 f"  first={loss_curve[0]:.4f}  "
#                 f"last={loss_curve[-1]:.4f}  "
#                 f"min={min(loss_curve):.4f}  "
#                 f"max={max(loss_curve):.4f}"
#             )
#         else:
#             loss_summary = "  (no loss history — train() was not called this session)"

#         emb = self.adata.obsm["emb"]
#         emb_norm = np.linalg.norm(emb, axis=1)

#         print("\n========== Model Diagnostics ==========")
#         print(f"  Total parameters:     {total_params:,}")
#         print(f"  Trainable parameters: {trainable_params:,}")
#         print(f"  Model weight size:    {model_mb:.2f} MB")
#         if peak_gpu_mb is not None:
#             print(f"  Peak GPU memory:      {peak_gpu_mb:.2f} MB")
#         if self._training_time_seconds is not None:
#             mins, secs = divmod(self._training_time_seconds, 60)
#             print(
#                 f"  Training time:        {int(mins)}m {secs:.1f}s  "
#                 f"({self._training_time_seconds:.1f}s total)"
#             )
#             if loss_curve:
#                 print(
#                     "  Time per epoch:       "
#                     f"{self._training_time_seconds / len(loss_curve) * 1000:.1f} ms"
#                 )
#         print(f"\n  Loss curve:          {loss_summary}")
#         print(f"\n  Embedding shape:      {emb.shape}")
#         print(
#             f"  Embedding norm — mean={emb_norm.mean():.4f}  "
#             f"std={emb_norm.std():.4f}  "
#             f"min={emb_norm.min():.4f}  "
#             f"max={emb_norm.max():.4f}"
#         )
#         print("=======================================\n")

#     # Clustering / metrics
#     def cluster(self, label: bool = True) -> None:
#         if self.tool not in _SUPPORTED_CLUSTER_TOOLS:
#             raise ValueError(
#                 f"Unsupported clustering tool '{self.tool}'. "
#                 f"Choose one of {_SUPPORTED_CLUSTER_TOOLS}."
#             )

#         clustering(
#             self.adata,
#             self.num_clusters,
#             radius=self.radius,
#             key="emb",
#             method=self.tool,
#             start=0.01,
#             end=0.27,
#             increment=0.005,
#             refinement=self.refine,
#         )

#         if label:
#             print("calculate metric ARI")
#             ARI = metrics.adjusted_rand_score(
#                 self.adata.obs["domain"], self.adata.obs["ground_truth"]
#             )
#             NMI = metrics.normalized_mutual_info_score(
#                 self.adata.obs["domain"], self.adata.obs["ground_truth"]
#             )
#             self.adata.uns["ari"] = ARI
#             self.adata.uns["nmi"] = NMI
#             print("ARI:", ARI)
#             print("NMI:", NMI)
#         else:
#             print("calculate SC and DB")
#             SC = silhouette_score(self.adata.obsm["emb"], self.adata.obs["domain"])
#             DB = davies_bouldin_score(self.adata.obsm["emb"], self.adata.obs["domain"])
#             self.adata.uns["sc"] = SC
#             self.adata.uns["db"] = DB
#             print("SC:", SC)
#             print("DB:", DB)

#     # Visualization
#     def draw_spatial(self, p: str = "") -> None:
#         sq.pl.spatial_scatter(
#             self.adata,
#             size=1.2,
#             figsize=(7, 7),
#             color=["ground_truth", "domain"],
#             save=p + str(self.args.slide) + ".png",
#         )

#     def draw_single_spatial(self) -> None:
#         sq.pl.spatial_scatter(
#             self.adata,
#             color="domain",
#             size=100,
#             save=str(self.args.slide) + ".png",
#         )

#     def draw_umap(self) -> None:
#         print("start umap")
#         sc.pp.neighbors(self.adata, use_rep="emb")
#         sc.tl.umap(self.adata)
#         sc.pl.umap(
#             self.adata,
#             color="domain",
#             show=True,
#             save=str(self.args.slide) + "domain.pdf",
#         )
#         if getattr(self.args, "label", False):
#             sc.pl.umap(
#                 self.adata,
#                 color="ground_truth",
#                 show=True,
#                 save=str(self.args.slide) + "_label.pdf",
#             )

#     def draw_horizontal(self) -> None:
#         adata_batch_0 = self.adata[self.adata.obs["batch"] == "0", :]
#         sq.pl.spatial_scatter(adata_batch_0, color="domain", save="0.png")

#         adata_batch_1 = self.adata[self.adata.obs["batch"] == "1", :]
#         sq.pl.spatial_scatter(adata_batch_1, color="domain", save="1.png")
