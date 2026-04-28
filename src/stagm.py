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
import torch.nn as nn

# Scikit-learn: Metrics, Decomposition, and Preprocessing
from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.metrics import (
    davies_bouldin_score,
    silhouette_score,
)

# PyTorch Geometric (Graph Neural Networks)
from torch_geometric.nn import GCNConv
from tqdm import tqdm

# STAGM-specific modules
from .model import Encoder, MVmodel, SVmodel, multiple_dropout_average, drop_feature
from .clustering import clustering  # , BatchKL


def generate_pseudo_labels(img_emb: np.ndarray, n_clusters: int = 300) -> torch.Tensor:
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(img_emb)
    pseudo_labels = kmeans.labels_
    return torch.tensor(pseudo_labels)


def adj_to_edge_index(adj: torch.Tensor) -> torch.Tensor:
    row, col = torch.where(adj != 0)
    edge_index = torch.stack([row, col], dim=0)
    return edge_index


def convert_edge_probabilities(
    adj_matrix: torch.Tensor, edge_prob_matrix: torch.Tensor
) -> torch.Tensor:
    row, col = torch.where(adj_matrix != 0)
    edge_probs = edge_prob_matrix[row, col]
    return edge_probs


class STAGM:
    def __init__(
        self,
        args: argparse.Namespace,
        config: dict,
        single: bool = False,
        refine: bool = True,
    ) -> None:
        self.args: argparse.Namespace = args
        self.single: bool = single
        self.config: dict = config
        self.learning_rate: float = config.learning_rate
        self.num_hidden: int = config.num_hidden
        self.num_proj_hidden: int = config.num_proj_hidden
        self.activation = ({"relu": nn.functional.relu, "prelu": nn.PReLU()})[
            config.activation
        ]
        self.base_model = ({"GCNConv": GCNConv})[config.base_model]
        self.num_layers: int = config.num_layers
        self.drop_feature_rate_1: float = config.drop_feature_rate_1
        self.drop_feature_rate_2: float = config.drop_feature_rate_2
        self.tau: float = config.tau
        self.num_epochs: int = config.num_epochs
        self.weight_decay: float = config.weight_decay
        self.num_clusters: int = config.num_clusters
        self.num_gene: int = config.num_gene
        self.refine: bool = refine
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.radius: int = 15
        self.tool: str = "leiden"  # mclust, leiden, and louvain
        self.bar_format: str = (
            "{l_bar}{bar}| [{elapsed}<{remaining}, {rate_fmt}{postfix}]"
        )
        self.encoder = Encoder(
            self.num_gene,
            self.num_hidden,
            self.activation,
            base_model=self.base_model,
            num_layers=self.num_layers,
            dropout=self.config.dropout,
            order_by_degree=self.config.order_by_degree,
            shuffle_ind=self.config.shuffle_ind,
            d_state=self.config.d_state,
            d_conv=self.config.d_conv,
        ).to(self.device)
        self.adata = None
        self.mask_slices = True

        if single:
            self.model = SVmodel(
                self.encoder, self.num_hidden, self.num_proj_hidden, self.tau
            ).to(self.device)

        else:
            self.model = MVmodel(
                self.encoder, self.num_hidden, self.num_proj_hidden, self.tau
            ).to(self.device)

        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )

    def train(self) -> None:
        print("=== prepare for training ===")
        if self.adata is None:
            raise ValueError("adata not load!")

        features_matrix = torch.FloatTensor(self.adata.obsm["feat"].copy()).to(
            self.device
        )

        graph_neigh = torch.FloatTensor(self.adata.obsm["graph_neigh"].copy()).to(
            self.device
        )
        edge_probabilities = torch.FloatTensor(
            self.adata.obsm["edge_probabilities"].copy()
        ).to(self.device)

        edge_index = adj_to_edge_index(graph_neigh)
        edge_probs = convert_edge_probabilities(graph_neigh, edge_probabilities)
        batch = torch.zeros(
            features_matrix.size(0), dtype=torch.long, device=self.device
        )

        if self.single:
            if ("mask_neigh" in self.adata.obsm) and (self.mask_slices):
                print("Consider intra slice")
                mask_neigh = torch.FloatTensor(self.adata.obsm["mask_neigh"].copy()).to(
                    self.device
                )
            else:
                mask_neigh = None
        else:
            if "pseudo_labels" in self.adata.obs:
                pseudo_labels = torch.tensor(
                    self.adata.obs["pseudo_labels"].cat.codes
                ).to(self.device)
            else:
                if self.config.k:
                    pseudo_labels = generate_pseudo_labels(
                        self.adata.obsm["img_emb"], self.config.k
                    )
                else:
                    pseudo_labels = generate_pseudo_labels(self.adata.obsm["img_emb"])
                pseudo_labels = pseudo_labels.to(self.device)

        self._loss_curve: list[float] = []
        train_start = perf_counter()

        print("=== train ===")
        for _ in tqdm(range(1, self.num_epochs + 1), bar_format=self.bar_format):
            self.model.train()
            self.optimizer.zero_grad()
            edge_index_1 = multiple_dropout_average(
                edge_index, edge_probs, force_undirected=True
            )[0]
            edge_index_2 = multiple_dropout_average(
                edge_index, edge_probs, force_undirected=True
            )[0]
            x_1 = drop_feature(features_matrix, self.drop_feature_rate_1)
            x_2 = drop_feature(features_matrix, self.drop_feature_rate_2)
            z1 = self.model(x_1, edge_index_1, batch)
            z2 = self.model(x_2, edge_index_2, batch)

            if self.single:
                loss = self.model.contrastive_loss(z1, z2, graph_neigh, mask=mask_neigh)
            else:
                loss = self.model.contrastive_loss_biased(
                    z1, z2, graph_neigh, pseudo_labels
                )

            loss.backward()
            self.optimizer.step()
            self._loss_curve.append(loss.item())

        self._training_time_seconds: float = perf_counter() - train_start

        if not self.single:
            torch.save(self.model.state_dict(), "model.pt")

    def eva(self):
        print("=== load ===")
        self.model.load_state_dict(torch.load("model.pt"))
        self.model.eval()

        features_matrix = torch.FloatTensor(self.adata.obsm["feat"].copy()).to(
            self.device
        )

        graph_neigh = torch.FloatTensor(self.adata.obsm["graph_neigh"].copy()).to(
            self.device
        )

        edge_index = adj_to_edge_index(graph_neigh)
        batch = torch.zeros(
            features_matrix.size(0), dtype=torch.long, device=self.device
        )

        with torch.no_grad():
            self.adata.obsm["emb"] = (
                self.model(features_matrix, edge_index, batch).detach().cpu().numpy()
            )

        # --- parameter counts ---
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(
            p.numel() for p in self.model.parameters() if p.requires_grad
        )

        # --- static memory footprint (weights + buffers, fp32 equivalent) ---
        param_bytes = sum(p.numel() * p.element_size() for p in self.model.parameters())
        buffer_bytes = sum(b.numel() * b.element_size() for b in self.model.buffers())
        model_mb = (param_bytes + buffer_bytes) / 1024**2

        # --- peak GPU memory allocated during the *current process* ---
        if self.device.type == "cuda":
            peak_gpu_mb = torch.cuda.max_memory_allocated(self.device) / 1024**2
            torch.cuda.reset_peak_memory_stats(self.device)  # reset for next run
        else:
            peak_gpu_mb = None

        # --- training time (set by train(); graceful fallback if eva() called standalone) ---
        training_time = getattr(self, "_training_time_seconds", None)

        # --- loss curve summary ---
        loss_curve = getattr(self, "_loss_curve", [])
        if loss_curve:
            loss_summary = (
                f"  first={loss_curve[0]:.4f}  "
                f"last={loss_curve[-1]:.4f}  "
                f"min={min(loss_curve):.4f}  "
                f"max={max(loss_curve):.4f}"
            )
        else:
            loss_summary = "  (no loss history — train() was not called this session)"

        # --- embedding stats ---
        emb = self.adata.obsm["emb"]
        emb_norm = np.linalg.norm(emb, axis=1)

        print("\n========== Model Diagnostics ==========")
        print(f"  Total parameters:     {total_params:,}")
        print(f"  Trainable parameters: {trainable_params:,}")
        print(f"  Model weight size:    {model_mb:.2f} MB")
        if peak_gpu_mb is not None:
            print(f"  Peak GPU memory:      {peak_gpu_mb:.2f} MB")
        if training_time is not None:
            mins, secs = divmod(training_time, 60)
            print(
                f"  Training time:        {int(mins)}m {secs:.1f}s  ({training_time:.1f}s total)"
            )
            if loss_curve:
                print(
                    f"  Time per epoch:       {training_time / len(loss_curve) * 1000:.1f} ms"
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
        print(self.adata.obsm["emb"])
        print("embedding generated, go clustering")

    def cluster(self, label=True):
        if self.tool == "mclust":
            clustering(
                self.adata,
                self.num_clusters,
                radius=self.radius,
                method=self.tool,
                refinement=self.refine,
            )  # For DLPFC dataset, we use optional refinement step.
        elif self.tool in ["leiden", "louvain"]:
            clustering(
                self.adata,
                self.num_clusters,
                radius=self.radius,
                method=self.tool,
                start=0.01,
                end=0.27,
                increment=0.005,
                refinement=self.refine,
            )

        if label:
            print("calculate metric ARI")
            # calculate metric ARI
            ARI = metrics.adjusted_rand_score(
                self.adata.obs["domain"], self.adata.obs["ground_truth"]
            )
            self.adata.uns["ari"] = ARI
            NMI = metrics.normalized_mutual_info_score(
                self.adata.obs["domain"], self.adata.obs["ground_truth"]
            )
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
        if "batch" in (self.adata.obs):
            BatchKL(self.adata)
            ILISI = hm.compute_lisi(
                self.adata.obsm["emb"],
                self.adata.obs[["batch"]],
                label_colnames=["batch"],
            )[:, 0]
            median_ILISI = np.median(ILISI)
            print(f"Median ILISI: {median_ILISI:.2f}")

    def draw_spatial(self, p=""):
        sq.pl.spatial_scatter(
            self.adata,
            size=1.2,
            figsize=(7, 7),
            color=["ground_truth", "domain"],
            save=p + str(self.args.slide) + ".png",
        )

    def draw_single_spatial(self):
        sq.pl.spatial_scatter(
            self.adata,
            color="domain",
            size=100,
            save=str(self.args.slide) + ".png",
        )

    def draw_umap(self):
        print("start umap")
        sc.pp.neighbors(self.adata, use_rep="emb")
        sc.tl.umap(self.adata)
        sc.pl.umap(
            self.adata,
            color="domain",
            show=True,
            save=str(self.args.slide) + "domain.pdf",
        )
        if self.args.label == True:
            sc.pl.umap(
                self.adata,
                color="ground_truth",
                show=True,
                save=str(self.args.slide) + "_label.pdf",
            )

    def draw_horizontal(self):
        adata_batch_0 = self.adata[self.adata.obs["batch"] == "0", :]
        sq.pl.spatial_scatter(adata_batch_0, color="domain", save="0.png")

        adata_batch_1 = self.adata[self.adata.obs["batch"] == "1", :]
        sq.pl.spatial_scatter(adata_batch_1, color="domain", save="1.png")
