from graphlearning.ssl import ssl
from scipy import sparse
import graphlearning as gl
import graphlearning.utils as utils
import numpy as np

class AlteredPoisson(ssl):
    def __init__(self, W=None, class_priors=None, tol=1e-3):
        """Poisson Learning
        ===================

        Semi-supervised learning via the solution of the Poisson equation 
        \\[L^p u = \\sum_{j=1}^m \\delta_j(y_j - \\overline{y})^T,\\]
        where \\(L=D-W\\) is the combinatorial graph Laplacian, 
        \\(y_j\\) are the label vectors, \\(\\overline{y} = \\frac{1}{m}\\sum_{i=1}^m y_j\\) 
        is the average label vector, \\(m\\) is the number of training points, and 
        \\(\\delta_j\\) are standard basis vectors. See the reference for more details.

        Implements 3 different solvers, spectral, gradient_descent, and conjugate_gradient. 
        GPU acceleration is available for gradient descent. See [1] for details.

        Parameters
        ----------
        W : numpy array, scipy sparse matrix, or graphlearning graph object (optional), default=None
            Weight matrix representing the graph.
        class_priors : numpy array (optional), default=None
            Class priors (fraction of data belonging to each class). If provided, the predict function
            will attempt to automatic balance the label predictions to predict the correct number of
            nodes in each class.
        min_iter : int (optional), default=50
            Minimum number of iterations of gradient descent before checking stopping condition.
        max_iter : int (optional), default=1000
            Maximum number of iterations of gradient descent.
        tol : float (optional), default=1e-3
            Tolerance for conjugate gradient solver.

        Examples
        --------
        Poisson learning works on directed (i.e., nonsymmetric) graphs with the gradient descent solver: [poisson_directed.py](https://github.com/jwcalder/GraphLearning/blob/master/examples/poisson_directed.py).
        ```py
        import numpy as np
        import graphlearning as gl
        import matplotlib.pyplot as plt
        import sklearn.datasets as datasets

        X,labels = datasets.make_moons(n_samples=500,noise=0.1)
        W = gl.weightmatrix.knn(X,10,symmetrize=False)

        train_ind = gl.trainsets.generate(labels, rate=5)
        train_labels = labels[train_ind]

        model = gl.ssl.poisson(W, solver='gradient_descent')
        pred_labels = model.fit_predict(train_ind, train_labels)

        accuracy = gl.ssl.ssl_accuracy(pred_labels, labels, train_ind)   
        print("Accuracy: %.2f%%"%accuracy)

        plt.scatter(X[:,0],X[:,1], c=pred_labels)
        plt.scatter(X[train_ind,0],X[train_ind,1], c='r')
        plt.show()
        ```

        Reference
        ---------
        [1] J. Calder, B. Cook, M. Thorpe, D. Slepcev. [Poisson Learning: Graph Based Semi-Supervised
        Learning at Very Low Label Rates.](http://proceedings.mlr.press/v119/calder20a.html), 
        Proceedings of the 37th International Conference on Machine Learning, PMLR 119:1306-1316, 2020.
        """
        super().__init__(W, class_priors)
        # Only keep conjugate gradient solver as the other solvers are more difficult to compute for signed graphs.
        self.tol = tol

        #Setup accuracy filename
        fname = '_poisson' 
        self.accuracy_filename = fname

        #Setup Algorithm name
        self.name = 'Poisson Learning'
    
    @staticmethod
    def signed_laplacian(G, normalization="combinatorial"):
        if hasattr(G, "W") and G.W is not None:
            W = G.W
        elif hasattr(G, "A") and G.A is not None:
            W = G.A
        elif hasattr(G, "weight_matrix") and G.weight_matrix is not None:
            W = G.weight_matrix
        else:
            raise AttributeError(
                "Graph object has no adjacency matrix (expected .W, .A, or .weight_matrix)"
            )
        W = W.tocsr()
        abs_deg = np.asarray(np.abs(W).sum(axis=1)).ravel()
        D_abs = sparse.diags(abs_deg)
        L = D_abs - W
        if normalization == "normalized":
            D_inv_sqrt = sparse.diags(1.0 / np.sqrt(abs_deg + 1e-10))
            L = D_inv_sqrt @ L @ D_inv_sqrt
        return L
    
    @staticmethod
    def signed_degree_matrix(G, p=1.0):
        if hasattr(G, "W") and G.W is not None:
            W = G.W
        elif hasattr(G, "A") and G.A is not None:
            W = G.A
        elif hasattr(G, "weight_matrix") and G.weight_matrix is not None:
            W = G.weight_matrix
        else:
            raise AttributeError(
                "Graph object has no adjacency matrix (expected .W, .A, or .weight_matrix)"
            )

        W = W.tocsr()
        abs_deg = np.asarray(np.abs(W).sum(axis=1)).ravel()

        return sparse.diags((abs_deg + 1e-10) ** p)

    def _fit(self, train_ind, train_labels, all_labels=None):
        assert np.isin(train_labels, [-1, 1]).all()

        n = self.graph.num_nodes # type: ignore
        W = self.graph.weight_matrix # type: ignore
        #Zero out diagonal for faster convergence
        W = W - sparse.spdiags(W.diagonal(),0,n,n)
        G = gl.graph(W)

        #Poisson source term
        onehot = np.zeros((len(train_labels), 2))
        onehot[train_labels == -1, 0] = 1
        onehot[train_labels == 1, 1] = 1
        source = np.zeros((n, 2))
        source[train_ind] = onehot - np.mean(onehot, axis=0)
                
        #Conjugate gradient solver
        #Use signed Laplacian as graph contains negative weights.

        
        L = self.signed_laplacian(G, normalization='combinatorial')
        L = L + 1e-6 * sparse.eye(n)
        # compute abs degree matrix for preconditioning with D^-0.5
        D = self.signed_degree_matrix(G, p=-0.5) 
        u = utils.conjgrad(L, D@source, tol=self.tol)
        u = D@u

        #Took out spectral and gredient descent solvers as they are more difficult to compute for signed graphss

        # convert to (n, 1) vector of entries in [-1, 1] by taking difference of max two entries and normalizing
        # if entry is < 0, predict -1, if entry is > 0, predict +1
        score = u[:, 1] - u[:, 0]
        max_abs = np.max(np.abs(score))
        if max_abs > 0:
            score = score / max_abs
        return score
