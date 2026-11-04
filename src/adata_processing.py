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
    return KMeans(n_clusters=n_clusters, random_state=0).fit(img_emb).labels_


def hilbert_index_2d(coords: np.ndarray, bits: int = 16) -> np.ndarray:
    if not 1 <= bits <= 31:
        raise ValueError("bits must be between 1 and 31")
    coords = np.asarray(coords, dtype=np.float64)
    if coords.ndim != 2 or coords.shape[1] < 2:
        raise ValueError("coords must have shape (n_cells, >=2)")
    coords = coords[:, :2]
    if coords.shape[0] == 0:
        return np.zeros(0, dtype=np.int64)
    if not np.isfinite(coords).all():
        raise ValueError("coords contain NaN or infinite values")

    side = 1 << bits
    mins = coords.min(axis=0)
    span = float((coords.max(axis=0) - mins).max())
    if span == 0.0:
        return np.zeros(coords.shape[0], dtype=np.int64)

    grid = np.floor((coords - mins) / span * (side - 1)).astype(np.int64)
    x, y = grid[:, 0], grid[:, 1]
    index = np.zeros(coords.shape[0], dtype=np.int64)

    step = side >> 1
    while step > 0:
        rx = (x & step) > 0
        ry = (y & step) > 0
        index += step * step * ((3 * rx.astype(np.int64)) ^ ry.astype(np.int64))
        flip = rx & ~ry
        x = np.where(flip, side - 1 - x, x)
        y = np.where(flip, side - 1 - y, y)
        swap = ~ry
        x, y = np.where(swap, y, x), np.where(swap, x, y)
        step >>= 1
    return index


def hilbert_order(
    coords: np.ndarray, groups: Sequence | None = None, bits: int = 16
) -> np.ndarray:
    coords = np.asarray(coords, dtype=np.float64)
    n_cells = coords.shape[0]
    if groups is None:
        return np.argsort(hilbert_index_2d(coords, bits), kind="stable")

    codes, _ = pd.factorize(np.asarray(groups), use_na_sentinel=False)
    if codes.shape[0] != n_cells:
        raise ValueError("groups must have one entry per cell")
    keys = np.zeros(n_cells, dtype=np.int64)
    for code in np.unique(codes):
        members = codes == code
        keys[members] = hilbert_index_2d(coords[members], bits)
    return np.lexsort((keys, codes))


def hilbert_rank(
    coords: np.ndarray, groups: Sequence | None = None, bits: int = 16
) -> np.ndarray:
    order = hilbert_order(coords, groups=groups, bits=bits)
    rank = np.empty_like(order)
    rank[order] = np.arange(order.shape[0], dtype=order.dtype)
    return rank


class _BaseAdataLoader:
    @staticmethod
    def _read_visium(
        path: str, counts_file: str = "filtered_feature_bc_matrix.h5"
    ) -> AnnData:
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", message="Variable names are not unique", category=UserWarning
            )
            adata = sq.read.visium(path, counts_file=counts_file, load_images=True)
        adata.var_names_make_unique()
        return adata

    @staticmethod
    def _read_xenium(path: str, **kwargs) -> AnnData:
        from spatialdata_io import xenium as read_xenium_bundle

        sdata = read_xenium_bundle(path, **kwargs)
        adata = sdata.tables["table"]
        adata.var_names_make_unique()
        return adata

    def _spatial_graph(self, adata: AnnData) -> csr_matrix:
        coords = adata.obsm["spatial"]
        n_neighbors = min(self.n_neighbors, coords.shape[0] - 1)

        adjacency = kneighbors_graph(
            coords, n_neighbors=n_neighbors, mode="connectivity", include_self=False
        )
        adjacency = adjacency.maximum(adjacency.T)
        adjacency.data[:] = 1.0
        return adjacency.tocsr()

    @staticmethod
    def _dense(adata: AnnData) -> np.ndarray:
        matrix = adata.X
        return matrix.toarray() if issparse(matrix) else np.asarray(matrix)

    @classmethod
    def _extract_feat(cls, adata: AnnData) -> np.ndarray:
        hvg_adata = adata[:, adata.var["highly_variable"]]
        return cls._dense(hvg_adata)

    @staticmethod
    def _load_image_embedding(path: str, n_components: int = 128) -> np.ndarray:
        embedding = np.load(os.path.join(path, "embeddings.npy"))
        embedding = embedding.reshape(embedding.shape[0], -1)
        embedding = StandardScaler().fit_transform(embedding)
        n_components = min(n_components, *embedding.shape)
        return PCA(n_components=n_components, random_state=42).fit_transform(embedding)

    @staticmethod
    def _load_ground_truth(adata: AnnData, path: str, filter_na: bool) -> AnnData:
        ground_truth = pd.read_csv(
            os.path.join(path, "truth.txt"), sep="\t", header=None
        )
        adata.obs["ground_truth"] = ground_truth[1].values
        if filter_na:
            adata = adata[~pd.isnull(adata.obs["ground_truth"])].copy()
        return adata

    @staticmethod
    def _edge_probabilities(
        adj: csr_matrix, node_emb: np.ndarray, metric: str
    ) -> csr_matrix:
        if metric not in ("euclidean", "cosine"):
            raise ValueError(f"Unsupported metric '{metric}'")

        adj = adj.tocsr()
        adj.eliminate_zeros()
        indptr = adj.indptr
        cols = adj.indices
        rows = np.repeat(np.arange(adj.shape[0]), np.diff(indptr))

        distances = paired_distances(node_emb[rows], node_emb[cols], metric=metric)
        if metric == "euclidean":
            distances = np.log(distances)

        edge_probabilities = np.zeros_like(distances)
        for node in tqdm(range(adj.shape[0]), desc="Edge probabilities"):
            start, end = indptr[node], indptr[node + 1]
            if start == end:
                continue
            edge_probabilities[start:end] = softmax(distances[start:end])

        directed = csr_matrix((edge_probabilities, cols, indptr), shape=adj.shape)
        symmetric = ((directed + directed.T) * 0.5).tocsr()
        symmetric.sort_indices()
        return symmetric

    def _gene_edge_probabilities(self, adj: csr_matrix, feat: np.ndarray) -> csr_matrix:
        embedding = StandardScaler().fit_transform(feat)
        n_components = min(64, *embedding.shape)
        embedding = PCA(n_components=n_components, random_state=42).fit_transform(
            embedding
        )
        return self._edge_probabilities(adj, embedding, metric="cosine")

    def _image_edge_probabilities(
        self, adj: csr_matrix, img_emb: np.ndarray
    ) -> csr_matrix:
        return self._edge_probabilities(adj, img_emb, metric="euclidean")


class SingleAdataLoader(_BaseAdataLoader):
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

        merged = concat(
            self.adata_list,
            join="outer",
            label="batch",
            keys=[str(i) for i in range(len(self.adata_list))],
            index_unique="-",
        )
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
