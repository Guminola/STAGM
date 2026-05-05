import numpy as np
import pandas as pd
from sklearn import metrics
import scanpy as sc
import ot
from sklearn.preprocessing import StandardScaler


def mclust_R(
    adata, num_cluster, modelNames="EEE", used_obsm="norm_emb", random_seed=2023
):
    """
    Clustering using the mclust algorithm.
    The parameters are the same as those in the R package mclust.
    """
    np.random.seed(random_seed)
    import rpy2.robjects as robjects
    from rpy2.robjects import numpy2ri
    from rpy2.robjects.conversion import localconverter

    robjects.r.library("mclust")

    r_random_seed = robjects.r["set.seed"]
    r_random_seed(random_seed)
    rmclust = robjects.r["Mclust"]

    with localconverter(robjects.default_converter + numpy2ri.converter):
        res = rmclust(numpy2ri.py2rpy(adata.obsm[used_obsm]), num_cluster, modelNames)
        mclust_res = np.array(res[-2])

    adata.obs["mclust"] = mclust_res
    adata.obs["mclust"] = adata.obs["mclust"].astype("int")
    adata.obs["mclust"] = adata.obs["mclust"].astype("category")

    return adata


def clustering(
    adata,
    n_clusters=7,
    radius=50,
    key="emb",
    method="mclust",
    start=0.1,
    end=3.0,
    increment=0.01,
    refinement=False,
):
    """
    Spatial clustering based the learned representation.

    Parameters
    ----------
    adata : anndata
        AnnData object of scanpy package.
    n_clusters : int, optional
        The number of clusters. The default is 7.
    radius : int, optional
        The number of neighbors considered during refinement. The default is 50.
    key : string, optional
        The key of the learned representation in adata.obsm. The default is 'emb'.
    method : string, optional
        The tool for clustering. Supported tools include 'mclust', 'leiden', and 'louvain'. The default is 'mclust'.
    start : float
        The start value for searching. The default is 0.1.
    end : float
        The end value for searching. The default is 3.0.
    increment : float
        The step size to increase. The default is 0.01.
    refinement : bool, optional
        Refine the predicted labels or not. The default is False.

    Returns
    -------
    None.

    """
    if method == "mclust":
        adata = mclust_R(adata, used_obsm=key, num_cluster=n_clusters)
        adata.obs["domain"] = adata.obs["mclust"]

    elif method in ("leiden", "louvain"):
        res = search_res(
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

        else:
            sc.tl.louvain(adata, random_state=0, resolution=res)

        adata.obs["domain"] = adata.obs[method]

    if refinement:
        new_type = refine_label(adata, radius, key="domain")
        adata.obs["domain"] = new_type


def refine_label(adata, radius=50, key="label"):
    n_neigh = radius
    old_type = adata.obs[key].values

    # Calculate pairwise euclidean distances between spatial positions
    position = adata.obsm["spatial"]
    distance = ot.dist(position, position, metric="euclidean")

    n_cell = distance.shape[0]
    new_type = []
    for i in range(n_cell):
        index = distance[i, :].argsort()
        neigh_type = [old_type[index[j]] for j in range(1, n_neigh + 1)]
        new_type.append(max(neigh_type, key=neigh_type.count))

    return [str(i) for i in new_type]


def search_res(
    radius,
    adata,
    n_clusters,
    method="leiden",
    use_rep="norm_emb",
    start=0.1,
    end=3.0,
    increment=0.01,
):
    """
    Searching corresponding resolution according to given cluster number

    Parameters
    ----------
    adata : anndata
        AnnData object of spatial data.
    n_clusters : int
        Targetting number of clusters.
    method : string
        Tool for clustering. Supported tools include 'leiden' and 'louvain'. The default is 'leiden'.
    use_rep : string
        The indicated representation for clustering.
    start : float
        The start value for searching.
    end : float
        The end value for searching.
    increment : float
        The step size to increase.

    Returns
    -------
    res : float
        Resolution.

    """

    def _cluster(resolution):
        """Run the chosen clustering method and return the unique cluster count."""
        if method == "leiden":
            sc.tl.leiden(adata, random_state=0, resolution=resolution)
            return len(adata.obs["leiden"].unique())
        else:
            sc.tl.louvain(adata, random_state=0, resolution=resolution)
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
