from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

import numpy as np
import scanpy as sc
from anndata import AnnData
from sklearn import metrics
from sklearn.neighbors import NearestNeighbors

_SUPPORTED_METHODS = ("leiden", "louvain")

_CLUSTER_KWARGS = {
    "leiden": dict(flavor="igraph", n_iterations=2, directed=False),
    "louvain": dict(flavor="igraph"),
}
_CLUSTER_FN = {"leiden": sc.tl.leiden, "louvain": sc.tl.louvain}


def _validate_method(method: str) -> None:
    if method not in _SUPPORTED_METHODS:
        raise ValueError(
            f"Unsupported clustering method '{method}'. Choose one of {_SUPPORTED_METHODS}."
        )


def refine_label(adata: AnnData, radius: int = 50, key: str = "label") -> np.ndarray:
    old_type = adata.obs[key].astype(str).to_numpy()
    position = adata.obsm["spatial"]

    n_neighbors = min(radius + 1, position.shape[0])
    neighbor_idx = (
        NearestNeighbors(n_neighbors=n_neighbors)
        .fit(position)
        .kneighbors(position, return_distance=False)
    )

    return np.array(
        [Counter(old_type[idx[1:]]).most_common(1)[0][0] for idx in neighbor_idx]
    )


@dataclass
class ResolutionSearch:
    adata: AnnData
    n_clusters: int
    method: str = "leiden"
    use_rep: str = "norm_emb"
    n_neighbors: int = 20
    fine_increment: float = 0.005
    fine_window: float = 0.05
    bracket_step: float = 0.1
    max_bracket_expansions: int = 30
    max_bisection_steps: int = 30
    tolerance: float = 1e-3

    _cache: dict[float, tuple[np.ndarray, int]] = field(
        default_factory=dict, init=False, repr=False
    )

    def __post_init__(self) -> None:
        _validate_method(self.method)
        self.has_ground_truth = "ground_truth" in self.adata.obs

    def _cluster_at(self, resolution: float) -> tuple[np.ndarray, int]:
        resolution = round(max(resolution, 1e-4), 6)
        if resolution not in self._cache:
            _CLUSTER_FN[self.method](
                self.adata,
                random_state=0,
                resolution=resolution,
                **_CLUSTER_KWARGS[self.method],
            )
            labels = self.adata.obs[self.method].to_numpy()
            count = len(np.unique(labels))
            print(f"resolution={resolution:.4f}, cluster number={count}")
            self._cache[resolution] = (labels, count)
        return self._cache[resolution]

    def _score(self, labels: np.ndarray) -> float:
        return metrics.adjusted_rand_score(labels, self.adata.obs["ground_truth"])

    def _expand_bracket(self, start: float, end: float) -> tuple[float, float]:
        low, high = start, end
        _, low_count = self._cluster_at(low)
        _, high_count = self._cluster_at(high)

        expansions = 0
        while high_count < self.n_clusters and expansions < self.max_bracket_expansions:
            high += self.bracket_step
            _, high_count = self._cluster_at(high)
            expansions += 1

        expansions = 0
        while low_count > self.n_clusters and expansions < self.max_bracket_expansions:
            low = max(low - self.bracket_step, 1e-4)
            _, low_count = self._cluster_at(low)
            expansions += 1

        if low_count > self.n_clusters or high_count < self.n_clusters:
            raise RuntimeError(
                f"Could not bracket {self.n_clusters} clusters between resolutions "
                f"{low:.4f} (n={low_count}) and {high:.4f} (n={high_count}). "
                "Try a wider start/end range."
            )

        return low, high

    def _bisect_to_target(self, low: float, high: float) -> float:
        for _ in range(self.max_bisection_steps):
            mid = (low + high) / 2
            _, count = self._cluster_at(mid)

            if count == self.n_clusters:
                return mid
            if count < self.n_clusters:
                low = mid
            else:
                high = mid

            if high - low < self.tolerance:
                break

        return min(
            self._cache,
            key=lambda res: abs(self._cache[res][1] - self.n_clusters),
        )

    def _best_around(self, resolution: float) -> tuple[float, np.ndarray]:
        window = np.arange(
            max(resolution - self.fine_window, 1e-4),
            resolution + self.fine_window,
            self.fine_increment,
        )

        best_res, best_labels, best_ari = resolution, None, -np.inf
        for res in window:
            labels, count = self._cluster_at(res)
            if count != self.n_clusters:
                continue
            if not self.has_ground_truth:
                return res, labels
            ari = self._score(labels)
            if ari > best_ari:
                best_res, best_labels, best_ari = res, labels, ari

        if best_labels is None:
            best_labels, count = self._cluster_at(resolution)
            best_res = resolution
            print(
                f"Warning: no resolution in [{resolution - self.fine_window:.4f}, "
                f"{resolution + self.fine_window:.4f}] produced exactly "
                f"{self.n_clusters} clusters; falling back to resolution="
                f"{resolution:.4f} with {count} clusters."
            )
        elif self.has_ground_truth:
            self.adata.uns["ARI"] = best_ari

        return best_res, best_labels

    def run(self, start: float, end: float) -> tuple[float, np.ndarray]:
        print("Searching resolution...")
        sc.pp.neighbors(self.adata, n_neighbors=self.n_neighbors, use_rep=self.use_rep)

        low, high = self._expand_bracket(start, end)
        target_res = self._bisect_to_target(low, high)
        best_res, best_labels = self._best_around(target_res)

        if self.has_ground_truth and "ARI" in self.adata.uns:
            print(
                f"Best resolution found: (res={best_res:.4f}, "
                f"ARI={self.adata.uns['ARI']:.4f})"
            )
        else:
            print(f"Best resolution found: {best_res:.4f}")

        return best_res, best_labels


def search_res(
    adata: AnnData,
    n_clusters: int,
    method: str = "leiden",
    use_rep: str = "norm_emb",
    start: float = 0.1,
    end: float = 3.0,
    increment: float = 0.01,
) -> tuple[float, np.ndarray]:
    searcher = ResolutionSearch(
        adata,
        n_clusters,
        method=method,
        use_rep=use_rep,
        fine_increment=increment,
    )
    return searcher.run(start, end)


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


# from collections import Counter

# import numpy as np
# import scanpy as sc
# from anndata import AnnData
# from sklearn import metrics
# from sklearn.neighbors import NearestNeighbors

# _SUPPORTED_METHODS = ("leiden", "louvain")

# _CLUSTER_KWARGS = {
#     "leiden": dict(flavor="igraph", n_iterations=2, directed=False),
#     "louvain": dict(flavor="igraph"),
# }
# _CLUSTER_FN = {"leiden": sc.tl.leiden, "louvain": sc.tl.louvain}


# def _run_cluster(adata: AnnData, method: str, resolution: float) -> np.ndarray:
#     if method not in _SUPPORTED_METHODS:
#         raise ValueError(
#             f"Unsupported clustering method '{method}'. Choose one of {_SUPPORTED_METHODS}."
#         )
#     _CLUSTER_FN[method](
#         adata, random_state=0, resolution=resolution, **_CLUSTER_KWARGS[method]
#     )
#     return adata.obs[method].to_numpy()


# def clustering(
#     adata: AnnData,
#     n_clusters: int = 7,
#     radius: int = 50,
#     key: str = "emb",
#     method: str = "leiden",
#     start: float = 0.1,
#     end: float = 3.0,
#     increment: float = 0.01,
#     refinement: bool = False,
# ) -> None:
#     _, labels = search_res(
#         adata,
#         n_clusters,
#         method=method,
#         use_rep=key,
#         start=start,
#         end=end,
#         increment=increment,
#     )
#     adata.obs["domain"] = labels

#     if refinement:
#         adata.obs["domain"] = refine_label(adata, radius, key="domain")


# def refine_label(adata: AnnData, radius: int = 50, key: str = "label") -> np.ndarray:
#     old_type = adata.obs[key].astype(str).to_numpy()
#     position = adata.obsm["spatial"]

#     nearest_neighbors = NearestNeighbors(n_neighbors=radius + 1).fit(position)
#     _, neighbor_idx = nearest_neighbors.kneighbors(position)

#     new_type = [Counter(old_type[idx[1:]]).most_common(1)[0][0] for idx in neighbor_idx]
#     return np.array(new_type)


# def search_res(
#     adata: AnnData,
#     n_clusters: int,
#     method: str = "leiden",
#     use_rep: str = "norm_emb",
#     start: float = 0.1,
#     end: float = 3.0,
#     increment: float = 0.01,
# ) -> tuple[float, np.ndarray]:
#     if method not in _SUPPORTED_METHODS:
#         raise ValueError(
#             f"Unsupported clustering method '{method}'. Choose one of {_SUPPORTED_METHODS}."
#         )

#     has_ground_truth = "ground_truth" in adata.obs

#     print("Searching resolution...")
#     sc.pp.neighbors(adata, n_neighbors=20, use_rep=use_rep)

#     def cluster_at(resolution: float) -> np.ndarray:
#         return _run_cluster(adata, method, resolution)

#     labels = cluster_at(end)
#     count_unique = len(np.unique(labels))
#     while count_unique > n_clusters + 2:
#         print(f"Cluster count {count_unique} is too large, adjusting end downward...")
#         end -= 0.1
#         labels = cluster_at(end)
#         count_unique = len(np.unique(labels))
#     while count_unique < n_clusters + 2:
#         print(f"Cluster count {count_unique} is too small, adjusting end upward...")
#         end += 0.1
#         labels = cluster_at(end)
#         count_unique = len(np.unique(labels))

#     best_res: float | None = None
#     best_labels: np.ndarray | None = None
#     best_ari = -np.inf
#     for res in sorted(np.arange(start, end, increment), reverse=True):
#         labels = cluster_at(res)
#         count_unique = len(np.unique(labels))
#         print(f"resolution={res:.4f}, cluster number={count_unique}")

#         if count_unique == n_clusters:
#             if has_ground_truth:
#                 ari = metrics.adjusted_rand_score(labels, adata.obs["ground_truth"])
#                 print(f"ARI: {ari:.4f}")
#                 if ari > best_ari:
#                     best_res, best_labels, best_ari = res, labels, ari
#             elif best_res is None:
#                 best_res, best_labels = res, labels

#         if count_unique == n_clusters - 2:
#             break

#     assert best_res is not None, (
#         "Resolution not found. Please try a bigger range or a smaller step."
#     )

#     if has_ground_truth:
#         adata.uns["ARI"] = best_ari
#         print(f"Best resolution found: (res={best_res:.4f}, ARI={best_ari:.4f})")
#     else:
#         print(f"Best resolution found: {best_res:.4f}")

#     return best_res, best_labels
