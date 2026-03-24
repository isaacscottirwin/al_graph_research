import numpy as np
from al_graph_research.graphs.signed_laplacian import SignedLaplacian

def setup():
    return np.array([[0, 2, -1], 
                  [2, 0, 3], 
                  [-1, 3, 0]])

def test_signed_laplacian():
    A = setup()
    test_laplacian = SignedLaplacian(A)

    actual_D = test_laplacian.D_abs.toarray()
    actual_L = test_laplacian.L_signed.toarray()

    correct_D = np.array([[3,0,0],
                          [0,5,0],
                          [0,0,4]])
    
    correct_L = np.array([[3, -2, 1], 
                          [-2, 5, -3],
                          [1, -3, 4]])

    assert np.array_equal(actual_D, correct_D)
    assert np.array_equal(actual_L, correct_L)

    