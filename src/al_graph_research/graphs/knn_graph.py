import networkx as nx
import graphlearning.weightmatrix as wm
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh


class KNNGraph:
    def __init__(
        self,
        number_neigbors,
        kernal,
        data,
        construction_method: str = "legacy",
    ) -> None:
        """
        Construct a k-NN graph.

        Parameters
        ----------
        number_neigbors : int
            Number of nearest neighbors.
        kernal : str
            Kernel used by graphlearning.weightmatrix.knn.
        data : array-like of shape (n_samples, n_features)
            Input data matrix.
        construction_method : {"legacy", "partner"}, default="legacy"
            Graph construction backend.

            - "legacy": use the original wm.knn(...) call directly.
            - "partner": use exact knnsearch(..., method="brute") first, then
              build wm.knn(..., knn_data=(J, D), symmetrize=True).
        """
        self.G_nx: nx.Graph = None  # type: ignore
        self.A_sp: sp.csr_matrix = None  # type: ignore
        self.construction_method = construction_method

        self.create_graph(data, kernal, number_neigbors, construction_method)

    def create_graph(self, data, kernal, number_neighbors, construction_method: str = "legacy"):
        """
        Construct a k-nearest neighbor graph from input data.

        Parameters
        ----------
        data : array-like of shape (n_samples, n_features)
            Input dataset where each row represents a data point.
        kernal : str
            Kernel type used to compute edge weights.
        number_neighbors : int
            Number of nearest neighbors used to construct the graph.
        construction_method : {"legacy", "partner"}, default="legacy"
            Choice of graph construction routine.

        Raises
        ------
        ValueError
            If the input data or parameters are invalid.
        """
        X = np.asarray(data)

        if X.ndim != 2:
            raise ValueError("data must be a 2D array-like object")

        n_samples, _ = X.shape

        if n_samples < 2:
            raise ValueError("need at least 2 data pts to construct graph")
        if number_neighbors < 1:
            raise ValueError("number_neighbors parameter must be larger than 0")
        if number_neighbors >= n_samples:
            raise ValueError(
                f"number_neighbors parameter must be smaller than the number of data pts: {n_samples}"
            )
        if construction_method not in {"legacy", "partner"}:
            raise ValueError("construction_method must be 'legacy' or 'partner'.")

        if construction_method == "legacy":
            A = self._build_graph_legacy(X, kernal, number_neighbors)
        else:
            A = self._build_graph_partner(X, kernal, number_neighbors)

        A.data = np.nan_to_num(A.data, nan=0.0, posinf=0.0, neginf=0.0)  # type: ignore
        A = A.tocsr()  # type: ignore
        A.setdiag(0)
        A.eliminate_zeros()

        self.A_sp = A  # type: ignore
        self.G_nx = nx.from_scipy_sparse_array(A)

    @staticmethod
    def _build_graph_legacy(X, kernal, number_neighbors):
        """
        Original graph construction used in your codebase.
        """
        return wm.knn(X, k=number_neighbors, kernel=kernal)

    @staticmethod
    def _build_graph_partner(X, kernal, number_neighbors):
        """
        Graph construction matching your partner's workflow more closely.

        Steps
        -----
        1. Exact neighbor search with knnsearch(..., method="brute").
        2. Build graph with wm.knn(..., symmetrize=True, knn_data=(J, D)).
        """
        J, D = wm.knnsearch(
            X,
            k=number_neighbors + 1,
            method="brute",
            similarity="euclidean",
        )

        A = wm.knn(
            X,
            k=number_neighbors,
            kernel=kernal,
            symmetrize=True,
            knn_data=(J, D),
        )
        return A

    def copy(self):
        """
        Create a copy of the k-NN graph.

        Returns
        -------
        new : KNNGraph
            A new instance of the k-NN graph with the same structure.
        """
        new = self.__class__.__new__(self.__class__)
        new.G_nx = self.G_nx.copy()
        new.A_sp = self.A_sp.copy()
        new.construction_method = self.construction_method
        return new

    def eigv_nd(
        self,
        L: sp.spmatrix,
        starting_idx: int = 1,
        ending_idx: int = 3,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Compute a selected range of eigenpairs of a Laplacian-like matrix.

        Parameters
        ----------
        L : sp.spmatrix
            Sparse Laplacian or signed Laplacian matrix of shape (n, n).
        starting_idx : int, default=1
            Starting index (inclusive) of the eigenpairs after sorting.
        ending_idx : int, default=3
            Ending index (exclusive) of the eigenpairs after sorting.

        Returns
        -------
        vals : np.ndarray
            Selected eigenvalues.
        vecs : np.ndarray
            Corresponding eigenvectors.
        """
        if starting_idx < 0:
            raise ValueError("starting_idx must be nonnegative.")
        if starting_idx >= ending_idx:
            raise ValueError("starting_idx must be strictly less than ending_idx.")

        n = L.shape[0]
        if ending_idx > n:
            raise ValueError("ending_idx cannot exceed the matrix dimension.")

        if ending_idx == n:
            vals, vecs = np.linalg.eigh(L.toarray())  # type: ignore[arg-type]
        else:
            vals, vecs = eigsh(L, k=ending_idx, which="SM")

        vals = np.real(vals)
        vecs = np.real(vecs)

        order = np.argsort(vals)
        vals = vals[order]
        vecs = vecs[:, order]

        return vals[starting_idx:ending_idx], vecs[:, starting_idx:ending_idx]