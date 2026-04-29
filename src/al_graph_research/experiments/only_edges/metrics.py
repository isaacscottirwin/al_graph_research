import numpy as np
from scipy.sparse.linalg import eigsh
from al_graph_research.graphs.signed_laplacian import SignedLaplacian


class Metrics:
    @staticmethod
    def _small_eignenvalvec(A, k_small: int = 4) -> tuple[np.ndarray, np.ndarray]:
        """
        Compute the smallest eigenvalues and corresponding eigenvectors of the signed Laplacian.

        Parameters
        ----------
        A : scipy.sparse.spmatrix
            Adjacency matrix.
        k_small : int, default=4
            Number of smallest eigenvalues to compute.

        Returns
        -------
        tuple[ndarray, ndarray]
            Sorted smallest eigenvalues and their corresponding eigenvectors.
        """
        L = SignedLaplacian(A).L_signed
        n, _ = L.shape # type: ignore

        if k_small >= n:
            vals, vecs = np.linalg.eig(L.toarray())
            idx = np.argsort(np.real(vals))[:k_small]
            vals = vals[idx]
            vecs = vecs[:, idx]
        else:
            vals, vecs = eigsh(L, k=k_small, which="SM")
            idx = np.argsort(np.real(vals))
            vals = vals[idx]
            vecs = vecs[:, idx]

        return vals, vecs
    
    @staticmethod
    def eig_k(A, k: int) -> tuple[np.ndarray, np.ndarray]:
        """
        Compute the k smallest eigenvalue–eigenvector pairs of the signed Laplacian.

        Parameters
        ----------
        A : scipy.sparse.spmatrix
            Adjacency matrix.
        k : int
            Number of eigenpairs to compute.

        Returns
        -------
        (vals, vecs) : tuple
            vals : (k,) array of eigenvalues (ascending)
            vecs : (n, k) array of corresponding eigenvectors (columns)
        """
        if k < 1:
            raise ValueError("k must be >= 1")

        vals, vecs = Metrics._small_eignenvalvec(A, k_small=k)

        if len(vals) < k:
            n = A.shape[0]
            return (
                np.full((k,), np.nan),
                np.full((n, k), np.nan),
            )

        return vals, vecs

    @staticmethod
    def sort_history(
        history: list[np.ndarray],
        descending: bool = True
    ) -> np.ndarray:
        """
        Align eigenvector signs over time, then sort each vector independently.
        """
        H = np.stack(history)  # (T, n)

        # --- Sign alignment ---
        H_aligned = H.copy()
        for t in range(1, H.shape[0]):
            if np.dot(H_aligned[t - 1], H_aligned[t]) < 0:
                H_aligned[t] *= -1

        # --- Sorting ---
        order = np.argsort(H_aligned, axis=1)
        if descending:
            order = order[:, ::-1]

        return np.take_along_axis(H_aligned, order, axis=1)