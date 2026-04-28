import numpy as np
from scipy.sparse.linalg import eigsh
from al_graph_research.graphs.signed_laplacian import SignedLaplacian


class Metrics:
    @staticmethod
    def _small_eigenvalues(A, k_small: int = 4) -> np.ndarray:
        """
        Compute the smallest eigenvalues of the signed Laplacian.

        Parameters
        ----------
        A : scipy.sparse.spmatrix
            Adjacency matrix.
        k_small : int, default=4
            Number of smallest eigenvalues to compute.

        Returns
        -------
        ndarray
            Sorted smallest eigenvalues in ascending order.
        """
        L = SignedLaplacian(A).L_signed
        n, _ = L.shape # type: ignore

        if k_small >= n:
            vals = np.linalg.eigvalsh(L.toarray())
            vals = np.sort(np.real(vals))[:k_small]
        else:
            vals = eigsh(L, k=k_small, which="SM", return_eigenvectors=False)
            vals = np.sort(np.real(vals))

        return vals

    @staticmethod
    def _largest_eigenvalue(A) -> float:
        """
        Compute the largest eigenvalue of the signed Laplacian.

        Parameters
        ----------
        A : scipy.sparse.spmatrix
            Adjacency matrix.

        Returns
        -------
        float
            Largest eigenvalue.
        """
        L = SignedLaplacian(A).L_signed
        n, _ = L.shape # type: ignore

        if n <= 3:
            lmax = np.max(np.linalg.eigvalsh(L.toarray()))
        else:
            lmax = eigsh(L, k=1, which="LA", return_eigenvectors=False)[0]
            lmax = np.real(lmax)

        return float(lmax)

    @staticmethod
    def lam_k(A, k: int) -> float:
        """
        Compute the k-th smallest eigenvalue of the signed Laplacian.

        Parameters
        ----------
        A : scipy.sparse.spmatrix
            Adjacency matrix.
        k : int
            Which eigenvalue to return (1 = smallest, 2 = second-smallest, etc.).

        Returns
        -------
        float
            The k-th smallest eigenvalue, or nan if unavailable.
        """
        if k < 1:
            raise ValueError("k must be >= 1")

        vals = Metrics._small_eigenvalues(A, k_small=k)
        return float(vals[k - 1]) if len(vals) >= k else float("nan")

    @staticmethod
    def gap23(A) -> float:
        """
        Compute the spectral gap lambda_3 - lambda_2 of the signed Laplacian.

        Parameters
        ----------
        A : scipy.sparse.spmatrix
            Adjacency matrix.

        Returns
        -------
        float
            The gap lambda_3 - lambda_2, or nan if unavailable.
        """
        vals = Metrics._small_eigenvalues(A, k_small=3)
        return float(vals[2] - vals[1]) if len(vals) > 2 else float("nan")

    @staticmethod
    def lmax(A) -> float:
        """
        Compute the largest eigenvalue of the signed Laplacian.

        Parameters
        ----------
        A : scipy.sparse.spmatrix
            Adjacency matrix.

        Returns
        -------
        float
            Largest eigenvalue.
        """
        return Metrics._largest_eigenvalue(A)
    

    