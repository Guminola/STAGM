from __future__ import annotations

import os
import warnings
from collections.abc import Sequence

import numpy as np
import pandas as pd
import scanpy as sc
import squidpy as sq
from anndata import AnnData, concat
from scipy.linalg import block_diag
from scipy.sparse import block_diag as sparse_block_diag
from scipy.sparse import csr_matrix, issparse
from scipy.special import softmax
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import paired_distances
from sklearn.neighbors import kneighbors_graph
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm


def generate_pseudo_labels(img_emb: np.ndarray, n_clusters: int) -> np.ndarray:
    """KMeans pseudo-labels over an embedding (e.g. `obsm['img_emb']`)."""
    return KMeans(n_clusters=n_clusters, random_state=0).fit(img_emb).labels_


class _BaseAdataLoader:
    """Shared graph / feature / edge-weight logic for the loaders below."""

    n_neighbors: int = 3

    @staticmethod
    def _read_visium(
        path: str, counts_file: str = "filtered_feature_bc_matrix.h5"
    ) -> AnnData:
        """Read a 10x Visium sample and immediately dedupe var_names."""
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", message="Variable names are not unique", category=UserWarning
            )
            adata = sq.read.visium(path, counts_file=counts_file, load_images=True)
        adata.var_names_make_unique()
        return adata

    @staticmethod
    def _read_xenium(path: str, **kwargs) -> AnnData:
        from spatialdata_io import xenium as _read_xenium_bundle

        sdata = _read_xenium_bundle(path, **kwargs)
        adata = sdata.tables["table"]
        adata.var_names_make_unique()
        return adata

    def _spatial_graph(self, adata: AnnData) -> csr_matrix:
        """Symmetric 0/1 sparse KNN adjacency over `adata.obsm['spatial']`."""
        coords = adata.obsm["spatial"]
        n_neighbors = min(self.n_neighbors, coords.shape[0] - 1)

        adj = kneighbors_graph(
            coords, n_neighbors=n_neighbors, mode="connectivity", include_self=False
        )
        adj = adj.maximum(adj.T)
        adj.data[:] = 1.0
        return adj.tocsr()

    @staticmethod
    def _dense(adata: AnnData) -> np.ndarray:
        X = adata.X
        return X.toarray() if issparse(X) else np.asarray(X)

    @classmethod
    def _extract_feat(cls, adata: AnnData) -> np.ndarray:
        """Dense feature matrix for the genes flagged `highly_variable`."""
        adata_vars = adata[:, adata.var["highly_variable"]]
        return cls._dense(adata_vars)

    @staticmethod
    def _load_image_embedding(path: str, n_components: int = 128) -> np.ndarray:
        data = np.load(os.path.join(path, "embeddings.npy"))
        data = data.reshape(data.shape[0], -1)
        data = StandardScaler().fit_transform(data)
        return PCA(n_components=n_components, random_state=42).fit_transform(data)

    @staticmethod
    def _load_ground_truth(adata: AnnData, path: str, filter_na: bool) -> AnnData:
        df_meta = pd.read_csv(os.path.join(path, "truth.txt"), sep="\t", header=None)
        adata.obs["ground_truth"] = df_meta[1].values
        if filter_na:
            adata = adata[~pd.isnull(adata.obs["ground_truth"])].copy()
        return adata

    @staticmethod
    def _edge_probabilities(
        adj: csr_matrix, node_emb: np.ndarray, metric: str
    ) -> csr_matrix:
        """Per-node softmax over edge distances, computed only for existing edges."""
        if metric not in ("euclidean", "cosine"):
            raise ValueError(f"Unsupported metric '{metric}'")

        adj = adj.tocsr()
        adj.eliminate_zeros()
        indptr = adj.indptr
        cols = adj.indices
        rows = np.repeat(np.arange(adj.shape[0]), np.diff(indptr))

        dist = paired_distances(node_emb[rows], node_emb[cols], metric=metric)
        if metric == "euclidean":
            dist = np.log(dist)

        edge_probabilities = np.zeros_like(dist)
        for i in tqdm(range(adj.shape[0]), desc="Edge probabilities"):
            start, end = indptr[i], indptr[i + 1]
            if start == end:
                continue
            edge_probabilities[start:end] = softmax(dist[start:end])

        return csr_matrix((edge_probabilities, cols, indptr), shape=adj.shape)

    def _gene_edge_probabilities(self, adj: csr_matrix, feat: np.ndarray) -> csr_matrix:
        embedding = StandardScaler().fit_transform(feat)
        embedding = PCA(n_components=64, random_state=42).fit_transform(embedding)
        return self._edge_probabilities(adj, embedding, metric="cosine")

    def _image_edge_probabilities(
        self, adj: csr_matrix, img_emb: np.ndarray
    ) -> csr_matrix:
        return self._edge_probabilities(adj, img_emb, metric="euclidean")


class SingleAdataLoader(_BaseAdataLoader):
    """
    Loads one spatial slice and prepares it for STAGM.

    Args:
        path: For `source="visium"`, the 10x Visium sample directory
            (must contain `filtered_feature_bc_matrix.h5`, `truth.txt` if
            `label=True`, and `embeddings.npy` if `image_emb=True`). For
            `source="h5ad"`, the path to a `.h5ad` file (labels/images are
            looked up next to it, in its parent directory). For
            `source="xenium"`, the Xenium output bundle directory read via
            `spatialdata_io.xenium` (labels/images are looked up in the same
            directory, same as `visium`).
        source: "visium" (`squidpy.read.visium`), "h5ad" (`sc.read_h5ad`),
            or "xenium" (`spatialdata_io.xenium`).
        n_top_genes: Highly-variable-gene count. Clamped to the number of
            genes actually present (relevant for targeted Xenium panels).
        n_neighbors: KNN degree for the spatial graph.
        image_emb: Whether to load and use `embeddings.npy` for edge
            weighting (falls back to gene-expression edge weighting if
            False).
        label: Whether to attach `obs['ground_truth']` from `truth.txt`.
        filter_na: Drop spots with a missing ground-truth label.
        normalize: Run `normalize_total` + `log1p` + `scale` after HVG
            selection. Set False if the input is already normalized.
        xenium_kwargs: Extra kwargs forwarded to `spatialdata_io.xenium`
            when `source="xenium"`.
    """

    def __init__(
        self,
        path: str,
        source: str = "visium",
        n_top_genes: int = 3000,
        n_neighbors: int = 3,
        image_emb: bool = False,
        label: bool = True,
        filter_na: bool = True,
        normalize: bool = True,
        xenium_kwargs: dict | None = None,
    ) -> None:
        if source not in ("visium", "h5ad", "xenium"):
            raise ValueError("source must be 'visium', 'h5ad' or 'xenium'")
        self.path = path
        self.source = source
        self.n_top_genes = n_top_genes
        self.n_neighbors = n_neighbors
        self.image_emb = image_emb
        self.label = label
        self.filter_na = filter_na
        self.normalize = normalize
        self.xenium_kwargs = xenium_kwargs or {}
        self.adata: AnnData | None = None

    def _load_data(self) -> None:
        if self.source == "visium":
            self.adata = self._read_visium(self.path)
        elif self.source == "xenium":
            self.adata = self._read_xenium(self.path, **self.xenium_kwargs)
        else:
            self.adata = sc.read_h5ad(self.path)
            self.adata.var_names_make_unique()

    def _preprocess(self) -> None:
        n_top_genes = min(self.n_top_genes, self.adata.n_vars)
        sc.pp.highly_variable_genes(
            self.adata, flavor="seurat_v3", n_top_genes=n_top_genes
        )
        if self.normalize:
            sc.pp.normalize_total(self.adata, target_sum=1e4)
            sc.pp.log1p(self.adata)
            sc.pp.scale(self.adata, zero_center=False, max_value=10)

    def run(self) -> AnnData:
        self._load_data()
        if self.label:
            self.adata = self._load_ground_truth(self.adata, self.path, self.filter_na)
        self._preprocess()

        self.adata.obsm["graph_neigh"] = self._spatial_graph(self.adata)
        self.adata.obsm["feat"] = self._extract_feat(self.adata)

        if self.image_emb:
            img_emb = self._load_image_embedding(self.path)
            self.adata.obsm["img_emb"] = img_emb
            self.adata.obsm["edge_probabilities"] = self._image_edge_probabilities(
                self.adata.obsm["graph_neigh"], img_emb
            )
        else:
            self.adata.obsm["edge_probabilities"] = self._gene_edge_probabilities(
                self.adata.obsm["graph_neigh"], self.adata.obsm["feat"]
            )

        print("adata load done")
        return self.adata


class BatchAdataLoader(_BaseAdataLoader):
    """Loads and merges multiple spatial slices for STAGM's batch mode.

    Args:
        slices: Pre-loaded per-slice AnnData objects.
        dataset_path: Root directory containing one subdirectory per slice.
        file_list: Subdirectory names under `dataset_path`, one per slice.
        source: "visium" (`squidpy.read.visium`) or "xenium"
            (`spatialdata_io.xenium`). Only applies to the
            `dataset_path`/`file_list` loading path.
        n_top_genes: Highly-variable-gene count per slice, clamped to the
            number of genes actually present in that slice (relevant for
            targeted Xenium panels).
        n_neighbors: KNN degree for each slice's local spatial graph.
        image_emb: Load `embeddings.npy` per slice and use it for edge
            weighting. Only valid together with `dataset_path`/`file_list`,
            since pre-loaded `slices` have no associated directory to read
            embeddings from.
        label: Attach `obs['ground_truth']` per slice from `truth.txt`.
            Only valid together with `dataset_path`/`file_list`.
        filter_na: Drop spots with a missing ground-truth label.
        normalize: Run `normalize_total` + `log1p` per slice before HVG
            intersection.
        xenium_kwargs: Extra kwargs forwarded to `spatialdata_io.xenium`
            when `source="xenium"`.
    """

    def __init__(
        self,
        slices: Sequence[AnnData] | None = None,
        dataset_path: str | None = None,
        file_list: Sequence[str] | None = None,
        source: str = "visium",
        n_top_genes: int = 3000,
        n_neighbors: int = 5,
        image_emb: bool = False,
        label: bool = True,
        filter_na: bool = True,
        normalize: bool = True,
        xenium_kwargs: dict | None = None,
    ) -> None:
        if slices is None and (dataset_path is None or file_list is None):
            raise ValueError(
                "Provide either `slices` (pre-loaded AnnData objects) or "
                "`dataset_path` + `file_list` (per-slice directory names)."
            )
        if slices is not None and (image_emb or label):
            raise ValueError(
                "image_emb/label require reading per-slice files from disk; "
                "pass `dataset_path` + `file_list` instead of `slices`."
            )
        if source not in ("visium", "xenium"):
            raise ValueError("source must be 'visium' or 'xenium'")

        self.slices = slices
        self.dataset_path = dataset_path
        self.file_list = file_list
        self.source = source
        self.n_top_genes = n_top_genes
        self.n_neighbors = n_neighbors
        self.image_emb = image_emb
        self.label = label
        self.filter_na = filter_na
        self.normalize = normalize
        self.xenium_kwargs = xenium_kwargs or {}

        self.adata_list: list[AnnData] = []
        self.merged_adata: AnnData | None = None

    def _load_one(self, adata: AnnData, slice_path: str | None = None) -> AnnData:
        adata.var_names_make_unique()
        n_top_genes = min(self.n_top_genes, adata.n_vars)
        sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=n_top_genes)
        if self.normalize:
            sc.pp.normalize_total(adata, target_sum=1e4)
            sc.pp.log1p(adata)
        if self.label:
            adata = self._load_ground_truth(adata, slice_path, self.filter_na)
        if self.image_emb:
            adata.obsm["img_emb"] = self._load_image_embedding(slice_path)
        adata.obsm["local_graph"] = self._spatial_graph(adata)
        return adata

    def _load_data(self) -> None:
        if self.slices is not None:
            for adata in self.slices:
                self.adata_list.append(self._load_one(adata))
        else:
            for name in self.file_list:
                print(f"now load: {name}")
                slice_path = os.path.join(self.dataset_path, name)
                if self.source == "xenium":
                    adata = self._read_xenium(slice_path, **self.xenium_kwargs)
                else:
                    adata = self._read_visium(slice_path)
                self.adata_list.append(self._load_one(adata, slice_path))
        print("load all slices done")

    def _concatenate_slices(self) -> None:
        hvg_sets = [set(a.var.index[a.var["highly_variable"]]) for a in self.adata_list]
        shared_hvg = set.intersection(*hvg_sets)
        if not shared_hvg:
            raise ValueError(
                "No genes are flagged highly_variable in every slice — "
                "empty intersection. Try a larger n_top_genes."
            )

        merged = concat(self.adata_list, join="outer")
        merged.obsm["feat"] = self._dense(merged[:, merged.var.index.isin(shared_hvg)])

        self.merged_adata = merged
        print(f"merged feat shape: {merged.obsm['feat'].shape}")
        print("merge done")

    def _construct_whole_graph(self) -> None:
        local_graphs = [a.obsm["local_graph"] for a in self.adata_list]
        self.merged_adata.obsm["graph_neigh"] = sparse_block_diag(
            local_graphs, format="csr"
        )
        self.merged_adata.obsm["mask_neigh"] = block_diag(
            *[np.ones(g.shape, dtype=int) for g in local_graphs]
        )

    def run(self) -> AnnData:
        self._load_data()
        self._concatenate_slices()
        self._construct_whole_graph()

        graph_neigh = self.merged_adata.obsm["graph_neigh"]
        if self.image_emb:
            img_emb = self.merged_adata.obsm.get("img_emb")
            if img_emb is None:
                raise ValueError(
                    "image_emb=True but no image embedding ended up in "
                    "obsm['img_emb'] after merging."
                )
            self.merged_adata.obsm["edge_probabilities"] = (
                self._image_edge_probabilities(graph_neigh, img_emb)
            )
        else:
            self.merged_adata.obsm["edge_probabilities"] = (
                self._gene_edge_probabilities(
                    graph_neigh, self.merged_adata.obsm["feat"]
                )
            )

        print("merge adata load done")
        return self.merged_adata
