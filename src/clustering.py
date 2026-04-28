# Scientific Stack & General Data Processing
import numpy as np

# Bioinformatics & Spatial Analysis
import scanpy as sc

# PyTorch & Deep Learning
from anndata import AnnData

# SciPy: Sparse matrices, spatial distances, and special functions
from scipy.spatial.distance import cdist

# Scikit-learn: Metrics, Decomposition, and Preprocessing
from sklearn import metrics


def clustering(
    adata: AnnData,
    n_clusters: int = 7,
    radius: int = 50,
    key: str = "emb",
    method: str = "mclust",
    start: float = 0.1,
    end: float = 3.0,
    increment: float = 0.01,
    refinement: bool = False,
) -> None:
    """
    Spatial clustering based the learned representation.

    Args:
        adata:      An AnnData object containing the learned representation in adata.obsm[key].
        n_clusters: Number of clusters.
        radius:     Number of neighbors considered during refinement.
        key:        The key of the learned representation in adata.obsm.
        method:     Clustering tool. Supported tools include 'leiden', and 'louvain'.
        start:      The start value for searching.
        end:        The end value for searching.
        increment:  Step size to increase.
        refinement: Refine the predicted labels or not.

    Returns:
         None. The predicted labels will be stored in adata.obs['domain'].
    """
    res: float = search_res(
        radius,
        adata,
        n_clusters,
        use_rep=key,
        method=method,
        start=start,
        end=end,
        increment=increment,
    )

    if method == "leiden":
        sc.tl.leiden(adata, random_state=0, resolution=res)

    if method == "louvain":
        sc.tl.louvain(adata, random_state=0, resolution=res)

    adata.obs["domain"] = adata.obs[method]

    if refinement:
        new_type = refine_label(adata, radius, key="domain")
        adata.obs["domain"] = new_type


def refine_label(adata: AnnData, radius: int = 50, key: str = "label") -> list[str]:
    n_neigh = radius
    old_type = adata.obs[key].astype(str).values

    # Calculate pairwise euclidean distances between spatial positions
    position = adata.obsm["spatial"]
    distance = cdist(position, position, metric="euclidean")

    n_cell = distance.shape[0]
    new_type: list[str] = []
    for i in range(n_cell):
        index = distance[i, :].argsort()
        neigh_type = [old_type[index[j]] for j in range(1, n_neigh + 1)]
        new_type.append(max(neigh_type, key=neigh_type.count))

    return new_type


def search_res(
    radius: int,
    adata: AnnData,
    n_clusters: int,
    method: str = "leiden",
    use_rep: str = "norm_emb",
    start: float = 0.1,
    end: float = 3.0,
    increment: float = 0.01,
) -> float:
    """
    Searching corresponding resolution according to given cluster number

    Args:
        adata:      An AnnData object containing the learned representation in adata.obsm[use_rep].
        n_clusters: Targetting number of clusters.
        method:     Tool for clustering. Supported tools include 'leiden' and 'louvain'.
        use_rep:    The indicated representation for clustering.
        start:      The start value for searching.
        end:        The end value for searching.
        increment:  Step size to increase.

    Returns:
        Best[0]:    The resolution corresponding to the best ARI score.
    """

    def _cluster(resolution):
        """Run the chosen clustering method and return the unique cluster count."""
        if method == "leiden":
            sc.tl.leiden(
                adata,
                random_state=0,
                resolution=resolution,
                flavor="igraph",
                n_iterations=2,
            )
            return len(adata.obs["leiden"].unique())
        else:
            sc.tl.louvain(
                adata,
                random_state=0,
                resolution=resolution,
                flavor="igraph",
                n_iterations=2,
            )
            return len(adata.obs["louvain"].unique())

    print("Searching resolution...")
    sc.pp.neighbors(adata, n_neighbors=20, use_rep=use_rep)

    # Coarsely adjust `end` so the upper-bound cluster count is n_clusters + 2
    count_unique = _cluster(end)
    while count_unique > n_clusters + 2:
        print(f"Cluster count {count_unique} is too large, adjusting end downward...")
        end -= 0.1
        count_unique = _cluster(end)
    while count_unique < n_clusters + 2:
        print(f"Cluster count {count_unique} is too small, adjusting end upward...")
        end += 0.1
        count_unique = _cluster(end)

    # Fine-grained search over [start, end)
    ress = []
    found = False
    for res in sorted(np.arange(start, end, increment), reverse=True):
        count_unique = _cluster(res)
        print(f"resolution={res:.4f}, cluster number={count_unique}")

        if count_unique == n_clusters:
            new_type = refine_label(adata, radius, key="leiden")
            adata.obs["leiden"] = new_type
            ARI = metrics.adjusted_rand_score(
                adata.obs["leiden"], adata.obs["ground_truth"]
            )
            adata.uns["ARI"] = ARI
            ress.append((res, ARI))
            print(f"ARI: {ARI:.4f}")

        if count_unique == n_clusters - 2:
            found = True
            best = max(ress, key=lambda x: x[1])
            print(f"Best resolution found: {best}")
            break

    assert found, "Resolution not found. Please try a bigger range or a smaller step."

    return best[0]
