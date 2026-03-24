import numpy as np
import scipy.sparse as sp


def to_sparse(A):
    if sp.issparse(A):
        return A.tocsr()
    return sp.csr_matrix(A)


def degree_abs(A):
    A = to_sparse(A)
    A_abs = A.copy()
    A_abs.data = np.abs(A_abs.data)
    d = np.asarray(A_abs.sum(axis=1)).ravel().astype(float)
    return d