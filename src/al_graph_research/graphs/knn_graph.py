import networkx as nx
import graphlearning.weightmatrix as wm
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh

class KNNGraph:
        def __init__(self, number_neigbors, kernal, data) -> None:
                self.G_nx = None
                self.A_sp = None

                self.create_graph(data, kernal, number_neigbors)
        
        def create_graph(self, data, kernal, number_neighbors):
                """
                Construct a k-nearest neighbor (k-NN) graph from input data.

                This method builds a weighted adjacency matrix using a specified kernel
                and number of neighbors, then stores both a SciPy sparse representation
                and a corresponding NetworkX graph.

                Parameters
                ----------
                data : array-like of shape (n_samples, n_features)
                Input dataset where each row represents a data point in feature space.
                kernal : str
                Kernel type used to compute edge weights {'uniform','gaussian','symgaussian','singular','distance'}.
                Passed directly to `graphlearning.weightmatrix.knn`.
                number_neighbors : int
                Number of nearest neighbors (k) used to construct the graph.

                Raises
                ------
                ValueError
                If:
                - data is not 2-dimensional,
                - number of samples is less than 2,
                - number_neighbors < 1,
                - number_neighbors >= n_samples.

                Side Effects
                ------------
                self.A_sp : sp.csr_matrix
                Sparse adjacency matrix of the k-NN graph in CSR format.
                self.G_nx : networkx.Graph
                NetworkX graph constructed from the sparse adjacency matrix.

                Notes
                -----
                - The adjacency matrix is constructed using `graphlearning.weightmatrix.knn`.
                - Any NaN or infinite edge weights are replaced with 0.
                - Self-loops are removed by zeroing the diagonal.
                - Explicit zero entries are removed from the sparse structure to maintain efficiency.
                - The final matrix is converted to CSR format for efficient arithmetic operations.
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
                        raise ValueError(f"number_neighbors parameter must be smaller than the number of data pts: {n_samples}")
                
                A = wm.knn(X, k=number_neighbors, kernel=kernal)
                # elminate edges with very small weights, useful if kernal is euclidiean
                A.data = np.nan_to_num(A.data, nan=0.0, posinf=0.0, neginf=0.0) # type: ignore
                # eliminate self loops
                A.setdiag(0) # type: ignore
                #remove explicit 0's from sparce representation
                A.eliminate_zeros() # type: ignore

                A = A.tocsr() # type: ignore

                self.A_sp = A
                self.G_nx = nx.from_scipy_sparse_array(A)
        
        def eigv_nd(self, L: sp.spmatrix, starting_idx: int = 1, ending_idx: int = 3) -> tuple[np.ndarray, np.ndarray]:
                """
                Compute a selected range of eigenpairs of a Laplacian-like matrix.

                This method returns eigenvalues and corresponding eigenvectors of L whose
                sorted indices lie in the interval [starting_idx, ending_idx). Eigenpairs
                are ordered by ascending eigenvalue.

                For large matrices, a sparse eigensolver is used to efficiently compute only
                the smallest required eigenvalues. A full dense eigendecomposition is used
                only when all eigenpairs are requested.

                Parameters
                ----------
                L : sp.spmatrix
                Sparse Laplacian or signed Laplacian matrix of shape (n, n).
                starting_idx : int, default=1
                Starting index (inclusive) of the eigenpairs after sorting by eigenvalue.
                Typically 1 to skip the trivial eigenvector for Laplacian embeddings.
                ending_idx : int, default=3
                Ending index (exclusive) of the eigenpairs after sorting.

                Returns
                -------
                vals : np.ndarray
                Array of shape (ending_idx - starting_idx,) containing the selected
                eigenvalues in ascending order.
                vecs : np.ndarray
                Array of shape (n, ending_idx - starting_idx) whose columns are the
                corresponding eigenvectors.

                Raises
                ------
                ValueError
                If:
                - starting_idx is negative,
                - starting_idx >= ending_idx,
                - ending_idx exceeds the matrix dimension.

                Notes
                -----
                - When ending_idx < n, the method uses `scipy.sparse.linalg.eigsh` to compute
                the smallest eigenvalues efficiently without forming a dense matrix.
                - When ending_idx == n, a full dense eigendecomposition
                (`numpy.linalg.eigh`) is used, which may be expensive for large matrices.
                - Eigenvalues and eigenvectors are cast to real values to remove small
                numerical imaginary components from floating-point computations.
                - This method is suitable for spectral embedding tasks where only a small
                number of eigenvectors (e.g., the first few nontrivial ones) are required.
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