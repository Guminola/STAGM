# Standard Library
from collections import Counter

# Scientific Stack & General Data Processing
import numpy as np

# Bioinformatics & Spatial Analysis
import scanpy as sc

# PyTorch & Deep Learning
from anndata import AnnData

# Scikit-learn: Metrics, Decomposition, and Preprocessing
from sklearn import metrics
from sklearn.neighbors import NearestNeighbors

_SUPPORTED_METHODS = ("leiden", "louvain")

_CLUSTER_KWARGS = {
    "leiden": dict(flavor="igraph", n_iterations=2, directed=False),
    "louvain": dict(flavor="igraph"),
}
_CLUSTER_FN = {"leiden": sc.tl.leiden, "louvain": sc.tl.louvain}


def _run_cluster(adata: AnnData, method: str, resolution: float) -> np.ndarray:
    """Run one leiden/louvain pass (igraph backend) and return the labels."""
    if method not in _SUPPORTED_METHODS:
        raise ValueError(
            f"Unsupported clustering method '{method}'. Choose one of {_SUPPORTED_METHODS}."
        )
    _CLUSTER_FN[method](
        adata, random_state=0, resolution=resolution, **_CLUSTER_KWARGS[method]
    )
    return adata.obs[method].to_numpy()


def clustering(
    adata: AnnData,
    n_clusters: int = 7,
    radius: int = 50,
    key: str = "emb",
    method: str = "leiden",
    start: float = 0.1,
    end: float = 3.0,
    increment: float = 0.01,
    refinement: bool = False,
) -> None:
    """
    Spatial clustering based on the learned representation.

    Args:
        adata:      An AnnData object containing the learned representation in adata.obsm[key].
        n_clusters: Number of clusters.
        radius:     Number of neighbors considered during refinement.
        key:        The key of the learned representation in adata.obsm.
        method:     Clustering tool. Supported tools include 'leiden' and 'louvain'.
        start:      The start value for searching.
        end:        The end value for searching.
        increment:  Step size to increase.
        refinement: Refine the predicted labels or not.

    Returns:
         None. The predicted labels will be stored in adata.obs['domain'].
    """
    _, labels = search_res(
        adata,
        n_clusters,
        method=method,
        use_rep=key,
        start=start,
        end=end,
        increment=increment,
    )
    adata.obs["domain"] = labels

    if refinement:
        adata.obs["domain"] = refine_label(adata, radius, key="domain")


def refine_label(adata: AnnData, radius: int = 50, key: str = "label") -> np.ndarray:
    """Refine the predicted labels by majority voting among neighbors."""
    old_type = adata.obs[key].astype(str).to_numpy()
    position = adata.obsm["spatial"]

    # +1 to include the point itself in the query, then drop it below.
    nbrs = NearestNeighbors(n_neighbors=radius + 1).fit(position)
    _, neighbor_idx = nbrs.kneighbors(position)

    new_type = [Counter(old_type[idx[1:]]).most_common(1)[0][0] for idx in neighbor_idx]
    return np.array(new_type)


def search_res(
    adata: AnnData,
    n_clusters: int,
    method: str = "leiden",
    use_rep: str = "norm_emb",
    start: float = 0.1,
    end: float = 3.0,
    increment: float = 0.01,
) -> tuple[float, np.ndarray]:
    """
    Search the clustering resolution that yields `n_clusters` clusters.

    When `adata.obs['ground_truth']` is present, candidate resolutions that
    produce exactly `n_clusters` clusters are ranked by ARI against it and
    the best-scoring one is kept. Otherwise (no ground truth available,
    e.g. unlabeled data) the first matching resolution encountered — the
    largest one, since the search runs from `end` down to `start` — is used.

    Args:
        adata:      An AnnData object containing the learned representation in adata.obsm[use_rep].
        n_clusters: Targeting number of clusters.
        method:     Tool for clustering. Supported tools include 'leiden' and 'louvain'.
        use_rep:    The indicated representation for clustering.
        start:      The start value for searching.
        end:        The end value for searching.
        increment:  Step size to increase.

    Returns:
        (best_resolution, cluster_labels_at_best_resolution)
    """
    if method not in _SUPPORTED_METHODS:
        raise ValueError(
            f"Unsupported clustering method '{method}'. Choose one of {_SUPPORTED_METHODS}."
        )

    has_ground_truth = "ground_truth" in adata.obs

    print("Searching resolution...")
    sc.pp.neighbors(adata, n_neighbors=20, use_rep=use_rep)

    def _cluster(resolution: float) -> np.ndarray:
        return _run_cluster(adata, method, resolution)

    # Coarsely adjust `end` so the upper-bound cluster count is n_clusters + 2
    labels = _cluster(end)
    count_unique = len(np.unique(labels))
    while count_unique > n_clusters + 2:
        print(f"Cluster count {count_unique} is too large, adjusting end downward...")
        end -= 0.1
        labels = _cluster(end)
        count_unique = len(np.unique(labels))
    while count_unique < n_clusters + 2:
        print(f"Cluster count {count_unique} is too small, adjusting end upward...")
        end += 0.1
        labels = _cluster(end)
        count_unique = len(np.unique(labels))

    # Fine-grained search over [start, end)
    best_res: float | None = None
    best_labels: np.ndarray | None = None
    best_ari = -np.inf
    for res in sorted(np.arange(start, end, increment), reverse=True):
        labels = _cluster(res)
        count_unique = len(np.unique(labels))
        print(f"resolution={res:.4f}, cluster number={count_unique}")

        if count_unique == n_clusters:
            if has_ground_truth:
                ari = metrics.adjusted_rand_score(labels, adata.obs["ground_truth"])
                print(f"ARI: {ari:.4f}")
                if ari > best_ari:
                    best_res, best_labels, best_ari = res, labels, ari
            elif best_res is None:
                best_res, best_labels = res, labels

        if count_unique == n_clusters - 2:
            break

    assert best_res is not None, (
        "Resolution not found. Please try a bigger range or a smaller step."
    )

    if has_ground_truth:
        adata.uns["ARI"] = best_ari
        print(f"Best resolution found: (res={best_res:.4f}, ARI={best_ari:.4f})")
    else:
        print(f"Best resolution found: {best_res:.4f}")

    return best_res, best_labels
