import numpy as np
import scipy.sparse as sp
from al_graph_research.graphs.utils import degree_abs, to_sparse
from typing import Union

class SignedLaplacian:
    def __init__(self, A: Union[np.ndarray, sp.spmatrix]):
        self.A = to_sparse(A)
        self.L_signed, self.D_abs = self._signed_laplacian()
    
    def _signed_laplacian(self):
        
        d = degree_abs(self.A)
        D_abs = sp.diags(d, dtype=float)        
        
        L_signed = D_abs - self.A
        
        return L_signed, D_abs