import numpy as np
import scipy.sparse as sp
from al_graph_research.graphs.utils import degree_abs, to_sparse
from typing import Union

class SignedLaplacian:
    """
    Construct the signed graph Laplacian from an adjacency matrix.

    The signed Laplacian is defined as:

        L_s = D_{|A|} - A

    where:
        - A is the (possibly signed) adjacency matrix
        - D_{|A|} is the diagonal degree matrix computed from the absolute
          values of A, i.e., (D_{|A|})_{ii} = sum_j |A_{ij}|

    This formulation preserves negative edge information while maintaining
    a Laplacian-like structure.

    Parameters
    ----------
    A : array-like or scipy.sparse.spmatrix of shape (n, n)
        Adjacency matrix of the graph. Can be dense or sparse. If dense,
        it will be converted to CSR sparse format.

    Attributes
    ----------
    A : scipy.sparse.csr_matrix
        Sparse adjacency matrix in CSR format.
    L_signed : scipy.sparse.csr_matrix
        Signed Laplacian matrix defined as D_{|A|} - A.
    D_abs : scipy.sparse.csr_matrix
        Diagonal matrix of absolute row sums of A.
    """

    def __init__(self, A: Union[np.ndarray, sp.spmatrix]):
        """
        Initialize the signed Laplacian from an adjacency matrix.

        Parameters
        ----------
        A : array-like or scipy.sparse.spmatrix
            Input adjacency matrix. Will be converted to CSR format if needed.
        """
        self.A = to_sparse(A)
        self.L_signed, self.D_abs = self._signed_laplacian()
    
    def _signed_laplacian(self):
        """
        Compute the signed Laplacian and absolute degree matrix.

        Returns
        -------
        L_signed : scipy.sparse.csr_matrix
            Signed Laplacian matrix defined as D_{|A|} - A.
        D_abs : scipy.sparse.csr_matrix
            Diagonal matrix of absolute degrees.
        """
        d = degree_abs(self.A)
        D_abs = sp.diags(d, dtype=float)        
        
        L_signed = D_abs - self.A
        
        return L_signed, D_abs