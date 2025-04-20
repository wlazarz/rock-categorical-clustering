import heapq
import warnings
from functools import lru_cache
from typing import Any, Dict, List, Tuple, Union

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.preprocessing import OneHotEncoder

warnings.filterwarnings('ignore')


class Rock:
    """
    ROCK (Robust Clustering using links) clustering algorithm for categorical data.
    Attributes:
        k (int): Desired number of clusters.
        eps (float): Similarity threshold for linking objects.
        labels (np.ndarray): Cluster labels after fitting.
    """

    def __init__(self, k: int, eps: Union[float, int]) -> None:
        """
        Initialize ROCK clustering.
        Args:
            k (int): Target number of clusters.
            eps (float): Jaccard similarity threshold (0 < eps <= 1).
        """
        super().__init__()
        self.k: int = k
        self.eps: float = eps
        self.labels: np.ndarray = np.array([])

    @staticmethod
    def encode_data(data: Any) -> csr_matrix:
        """
        Encode categorical input data into a sparse one-hot matrix.
        Args:
            data (Any): Tabular categorical data (e.g., pandas DataFrame or 2D array).
        Returns:
            csr_matrix: One-hot encoded sparse matrix of shape (n_samples, n_features).
        """
        encoder = OneHotEncoder(sparse_output=True, handle_unknown='ignore')
        return encoder.fit_transform(data)

    @staticmethod
    def build_links_graph(B: csr_matrix, eps: float) -> Dict[int, List[Tuple[int, int, int]]]:
        """
        Build a sparse graph of object links based on Jaccard similarity >= eps.
        Args:
            B (csr_matrix): One-hot encoded data matrix.
            eps (float): Similarity threshold.
        Returns:
            Dict[int, List[Tuple[int, int, int]]]:
                Mapping each object index to a list of (neighbor_idx,
                intersection_count, union_count).
        """
        coo = B.dot(B.T).tocoo()
        row_sums = np.ravel(B.sum(axis=1))
        links: Dict[int, List[Tuple[int, int, int]]] = {i: [] for i in range(B.shape[0])}

        for i, j, inter in zip(coo.row, coo.col, coo.data):
            if i >= j:
                continue
            union = row_sums[i] + row_sums[j] - inter
            if union > 0 and inter / union >= eps:
                links[i].append((j, inter, union))
                links[j].append((i, inter, union))
        return links

    @lru_cache(maxsize=None)
    def _denominator(self, size_i: int, size_j: int, func: float) -> float:
        """
        Compute ROCK denominator term: (i+j)^f - i^f - j^f with caching.
        Args:
            size_i (int): Size of cluster i.
            size_j (int): Size of cluster j.
            func (float): Exponent factor dependent on eps.
        Returns:
            float: Denominator value.
        """
        return (size_i + size_j) ** func - size_i ** func - size_j ** func

    def _initialize_heap(self, links: Dict[int, List[Tuple[int, int, int]]], sizes: List[int], func: float) \
            -> List[Tuple[float, int, int]]:
        """
        Build initial max-heap of candidate cluster merges.
        Args:
            links: Link graph.
            sizes: Current cluster sizes.
            func: Denominator exponent.
        Returns:
            list: Heap of (-goodness, i, j) entries.
        """
        heap: List[Tuple[float, int, int]] = []
        for i, neighs in links.items():
            for j, inter, union in neighs:
                denom = self._denominator(sizes[i], sizes[j], func)
                if denom > 0:
                    goodness = inter / denom
                    heapq.heappush(heap, (-goodness, i, j))
        return heap

    def _find_root(self, parent: List[int], u: int) -> int:
        """
        Find root of u with path compression.
        Args:
            parent: Parent pointers.
            u: Node index.
        Returns:
            int: Root index.
        """
        while parent[u] != u:
            parent[u] = parent[parent[u]]
            u = parent[u]
        return u

    def _merge_clusters(self, heap: List[Tuple[float, int, int]], parent: List[int], sizes: List[int],
                        active: List[bool], links: Dict[int, List[Tuple[int, int, int]]], func: float) -> None:
        """
        Perform cluster merges until desired k clusters remain.
        Modifies parent, sizes, active in place.
        """
        clusters_left = len(parent)
        while clusters_left > self.k and heap:
            _, i, j = heapq.heappop(heap)
            ri = self._find_root(parent, i)
            rj = self._find_root(parent, j)
            if ri == rj or not active[ri] or not active[rj]:
                continue

            if sizes[ri] < sizes[rj]:
                ri, rj = rj, ri
            parent[rj] = ri
            sizes[ri] += sizes[rj]
            active[rj] = False
            clusters_left -= 1

            for neigh, inter, union in links[rj]:
                rn = self._find_root(parent, neigh)
                if rn == ri or not active[rn]:
                    continue
                denom = self._denominator(sizes[ri], sizes[rn], func)
                if denom > 0:
                    goodness = inter / denom
                    heapq.heappush(heap, (-goodness, ri, rn))

    def _assign_labels(self, parent: List[int]) -> np.ndarray:
        """
        Assign contiguous cluster labels based on disjoint-set parents.
        Args:
            parent: Final parent pointers.
        Returns:
            np.ndarray: Cluster labels of shape (n_samples,).
        """
        labels: List[int] = [-1] * len(parent)
        mapping: Dict[int, int] = {}
        next_label = 0
        for i in range(len(parent)):
            root = self._find_root(parent, i)
            if root not in mapping:
                mapping[root] = next_label
                next_label += 1
            labels[i] = mapping[root]
        return np.array(labels, dtype=int)

    def fit_predict(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        Fit the ROCK model and return cluster labels.
        Args:
            X (Any): Categorical data to cluster.
        Returns:
            np.ndarray: Cluster labels.
        Raises:
            ValueError: If input contains missing or empty values.
        """
        import pandas as pd

        if isinstance(X, pd.DataFrame):
            X = X.values

        if isinstance(X, np.ndarray):
            if np.any(pd.isnull(X)) or np.any(X == ''):
                raise ValueError("Input contains missing (NaN or None) or empty string values.")

        B = self.encode_data(X)
        n_samples = B.shape[0]
        links = self.build_links_graph(B, self.eps)

        parent = list(range(n_samples))
        sizes = [1] * n_samples
        active = [True] * n_samples
        func = 1.0 + 2.0 * ((1.0 - self.eps) / (1.0 + self.eps))

        heap = self._initialize_heap(links, sizes, func)
        self._merge_clusters(heap, parent, sizes, active, links, func)
        self.labels = self._assign_labels(parent)
        return self.labels