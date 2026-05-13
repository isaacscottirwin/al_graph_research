from graphlearning.ssl import ssl
from scipy import sparse
import graphlearning as gl
import graphlearning.utils as utils
import numpy as np

class AlteredPoisson(ssl):
    def __init__(self, W=None, class_priors=None, tol=1e-3, normalization="normalized"):
        """    
        Signed Poisson Learning
        =======================

        Semi-supervised learning on a signed graph using a signed graph Laplacian.

        This model solves a Poisson-type graph equation using the signed Laplacian

            L_signed = D_abs - W,

        where W is a possibly signed adjacency matrix and D_abs is the diagonal
        matrix of absolute degrees,

            D_abs[i, i] = sum_j |W[i, j]|.

        The label information is encoded as a source term supported only on the
        labeled nodes. For binary labels {-1, +1}, labels are converted to a
        two-column one-hot representation, centered by subtracting the average
        labeled one-hot vector, and then propagated through the signed graph.

        This implementation only uses a conjugate-gradient solve. The spectral
        and gradient-descent solvers from the original unsigned Poisson learning
        implementation are intentionally omitted because they require additional
        care for signed graphs.

        After solving for the two-column Poisson solution u, the output is converted
        to a scalar score vector by taking

            score = u[:, 1] - u[:, 0].

        The score is then normalized to lie in [-1, 1]. Positive scores correspond
        to class +1, and negative scores correspond to class -1.

        Parameters
        ----------
        W : numpy array, scipy sparse matrix, or graphlearning graph object, optional
            Weight matrix representing the graph. The matrix may contain positive
            and negative edge weights.

        class_priors : numpy array, optional
            Class priors. Passed to the parent graphlearning SSL class.

        normalization : {"combinatorial", "normalized"}, default="normalized"
            Which signed Laplacian normalization to use.

            If "combinatorial", the model solves approximately
                L_signed u = source.
            A small diagonal regularization is added to improve numerical stability.

            If "normalized", the model uses
                L_norm = D_abs^{-1/2} L_signed D_abs^{-1/2}
            and solves the normalized system following the structure of the original
            graphlearning Poisson conjugate-gradient solver.

        tol : float, default=1e-3
            Tolerance for the conjugate-gradient solver.

        
        Reference
        ---------
        [1] J. Calder, B. Cook, M. Thorpe, D. Slepcev. [Poisson Learning: Graph Based Semi-Supervised
        Learning at Very Low Label Rates.](http://proceedings.mlr.press/v119/calder20a.html), 
        Proceedings of the 37th International Conference on Machine Learning, PMLR 119:1306-1316, 2020.
        """
        super().__init__(W, class_priors)
        self.tol = tol
        if normalization not in ["combinatorial", "normalized"]:
            raise ValueError("Normalization must be 'combinatorial' or 'normalized'")
        self.normalization = normalization
        self.accuracy_filename = f"_poisson_{normalization}"
        self.name = f"Signed Poisson Learning ({normalization})"

    @staticmethod
    def signed_laplacian(G, normalization="combinatorial"):
        """
        Construct the signed graph Laplacian.
        The signed Laplacian is defined as
            L_signed = D_abs - W,
        where W is the signed adjacency matrix and D_abs is the diagonal matrix
        of absolute degrees,
            D_abs[i, i] = sum_j |W[i, j]|.
        If normalization="normalized", this returns the normalized signed Laplacian
            L_norm = D_abs^{-1/2} L_signed D_abs^{-1/2}.

        Parameters
        ----------
        G : graph-like object
            Graph object containing an adjacency matrix. The method checks for
            one of the attributes .W, .A, or .weight_matrix.

        normalization : {"combinatorial", "normalized"}, default="combinatorial"
            Type of signed Laplacian to return.

        Returns
        -------
        scipy sparse matrix
            The signed graph Laplacian.
        """
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
        """
        Construct a power of the signed absolute degree matrix.
        The signed absolute degree matrix is
            D_abs[i, i] = sum_j |W[i, j]|.
        This method returns
            D_abs^p.
        For example, p=-0.5 gives D_abs^{-1/2}, which is used with the normalized
        signed Laplacian.

        Parameters
        ----------
        G : graph-like object
            Graph object containing an adjacency matrix. The method checks for
            one of the attributes .W, .A, or .weight_matrix.

        p : float, default=1.0
            Power applied to the absolute degree entries.

        Returns
        -------
        scipy sparse diagonal matrix
            The diagonal matrix D_abs^p.
        """
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

    def _fit(self, train_ind, train_labels):
        """
        Fit signed Poisson learning scores from labeled training nodes.
        The method assumes binary labels in {-1, +1}. Labels are converted into
        a two-column one-hot source term, centered by subtracting the average
        labeled one-hot vector. The resulting source is then propagated through
        either the combinatorial or normalized signed Laplacian.

        For normalized signed Poisson, this solves
            L_norm v = D_abs^{-1/2} source,
        then converts back using
            u = D_abs^{-1/2} v.
        For combinatorial signed Poisson, this solves
            (L_signed + eps I) u = source,
        where eps is a small regularization parameter used to improve numerical
        stability.
        The two-column output is converted to a scalar score by
            score = u[:, 1] - u[:, 0],
        and normalized to lie in [-1, 1].

        Parameters
        ----------
        train_ind : array-like
            Indices of labeled training nodes.

        train_labels : array-like
            Binary labels for the training nodes. Must contain only -1 and +1.

        Returns
        -------
        numpy.ndarray
            Score vector of shape (n,). Positive scores predict class +1 and
            negative scores predict class -1.
        """
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

        L = self.signed_laplacian(G, normalization=self.normalization)

        if self.normalization == "normalized":
            D = self.signed_degree_matrix(G, p=-0.5)
            u = utils.conjgrad(L, D @ source, tol=self.tol)
            u = D @ u
        elif self.normalization == "combinatorial":
            L = L + 1e-6 * sparse.eye(n)
            u = utils.conjgrad(L, source, tol=self.tol)

        #Took out spectral and gredient descent solvers as they are more difficult to compute for signed graphss

        # convert to (n, 1) vector of entries in [-1, 1] by taking difference of max two entries and normalizing
        # if entry is < 0, predict -1, if entry is > 0, predict +1
        score = u[:, 1] - u[:, 0]
        max_abs = np.max(np.abs(score))
        if max_abs > 0:
            score = score / max_abs
        return score
