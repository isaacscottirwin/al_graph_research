from graphlearning.ssl import ssl
from scipy import sparse
from scipy.sparse.linalg import spsolve
import graphlearning.utils as utils
import graphlearning.graph as graph
import numpy as np


class AlteredLaplace(ssl):
    def __init__(self, W=None, class_priors=None, X=None, reweighting="none", normalization="combinatorial", tau=0, 
                 order=1, mean_shift=False, tol=1e-5, alpha=2, zeta=1e7, r=0.1):
        """
        Signed Laplace Learning
        =======================

        Semi-supervised learning on a signed graph using a signed graph Laplacian.

        This model solves a Laplace-type boundary value problem where the labeled
        nodes are fixed to their observed labels and the unlabeled nodes are inferred
        by solving a linear system on the unlabeled subgraph.

        The signed graph Laplacian is defined as

            L_signed = D_abs - W,

        where W is a possibly signed adjacency matrix and D_abs is the diagonal matrix
        of absolute degrees,

            D_abs[i, i] = sum_j |W[i, j]|.

        For binary labels {-1, +1}, the method fixes the labeled nodes to their given
        values and solves for the remaining unlabeled values. Positive output values
        predict class +1 and negative output values predict class -1.

        Parameters
        ----------
        W : numpy array, scipy sparse matrix, or graphlearning graph object, optional
            Weight matrix representing the graph. The matrix may contain positive and
            negative edge weights.

        class_priors : numpy array, optional
            Class priors. Passed to the parent graphlearning SSL class.

        X : numpy array, optional
            Data features. Used only when a reweighting method requires access to the
            original data features.

        reweighting : {"none", "wnll", "poisson", "properly"}, default="none"
            Reweighting scheme applied before solving the Laplace problem. If "none",
            the original graph is used. Other options call the graphlearning graph
            reweighting routine.

        normalization : {"combinatorial", "normalized"}, default="combinatorial"
            Type of signed Laplacian to use.

            If "combinatorial", the method uses

                L_signed = D_abs - W.

            If "normalized", the method uses

                L_norm = D_abs^{-1/2} L_signed D_abs^{-1/2}.

        tau : float or numpy.ndarray, default=0
            Zeroth-order parameter retained for compatibility with the original
            Laplace-learning interface. In this implementation, tau is stored and used
            for naming, but it is not currently added to the solved linear system.

        order : int, default=1
            Power of the signed Laplacian used in the boundary value problem. If
            order > 1, the method replaces L by L^order.

        mean_shift : bool, default=False
            If True, subtracts the mean of the final score vector.

        tol : float, default=1e-5
            Tolerance for the conjugate-gradient solver.

        alpha : float, default=2
            Parameter retained for compatibility with properly-weighted graph
            reweighting.

        zeta : float, default=1e7
            Parameter retained for compatibility with properly-weighted graph
            reweighting.

        r : float, default=0.1
            Parameter retained for compatibility with properly-weighted graph
            reweighting.

        Notes
        -----
        This class modifies standard Laplace learning by replacing the ordinary graph
        Laplacian D - W with the signed graph Laplacian D_abs - W. This allows negative
        edges to encourage opposite signs between neighboring nodes.

        The method enforces the labels as hard boundary conditions. Therefore, changes
        to edges only between labeled nodes may have little or no effect on the
        unlabeled solution, since labeled values are fixed during the solve.


        References
        ---------
        [1] X. Zhu, Z. Ghahramani, and J. D. Lafferty. Semi-supervised learning using gaussian fields
        and harmonic functions. Proceedings of the 20th International Conference on Machine Learning
        (ICML-03), 2003.

        [2] J. Calder, B. Cook, M. Thorpe, D. Slepcev. Poisson Learning: Graph Based Semi-Supervised
        Learning at Very Low Label Rates. Proceedings of the 37th International Conference on Machine
        Learning, PMLR 119:1306-1316, 2020.

        [3] Z. Shi, S. Osher, and W. Zhu. Weighted nonlocal laplacian on interpolation from sparse data.
        Journal of Scientific Computing 73.2 (2017): 1164-1177.

        [4] J. Calder, D. Slepčev. Properly-weighted graph Laplacian for semi-supervised learning.
        Applied mathematics & optimization (2019): 1-49.
        """
        super().__init__(W, class_priors)

        self.reweighting = reweighting
        self.normalization = normalization
        self.mean_shift = mean_shift
        self.tol = tol
        self.order = order
        self.X = X

        if type(tau) in [float, int]:
            self.tau = np.ones(self.graph.num_nodes) * tau # type: ignore
        elif type(tau) is np.ndarray:
            self.tau = tau

        fname = "_laplace"
        self.name = "Laplace Learning"
        if self.reweighting != "none":
            fname += "_" + self.reweighting
            self.name += ": " + self.reweighting + " reweighted"
        if self.normalization != "combinatorial":
            fname += "_" + self.normalization
            self.name += " " + self.normalization
        if self.mean_shift:
            fname += "_meanshift"
            self.name += " with meanshift"
        if self.order > 1:
            fname += "_order%d" % int(self.order)
            self.name += " order %d" % int(self.order)
        if np.max(self.tau) > 0:
            fname += "_tau_%.3f" % np.max(self.tau)
            self.name += " tau=%.3f" % np.max(self.tau)

        self.accuracy_filename = fname

    @staticmethod
    def signed_laplacian(G, normalization="combinatorial"):
        """
        Construct the signed graph Laplacian.

        The signed Laplacian is defined as

            L_signed = D_abs - W,

        where W is the signed adjacency matrix and D_abs is the diagonal matrix of
        absolute degrees,

            D_abs[i, i] = sum_j |W[i, j]|.

        If normalization="normalized", this returns

            L_norm = D_abs^{-1/2} L_signed D_abs^{-1/2}.

        Parameters
        ----------
        G : graph-like object
            Graph object containing an adjacency matrix. The method checks for one of
            the attributes .W, .A, or .weight_matrix.

        normalization : {"combinatorial", "normalized"}, default="combinatorial"
            Type of signed Laplacian to return.

        Returns
        -------
        scipy sparse matrix
            Signed graph Laplacian.
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
        abs_deg = np.array(np.abs(W).sum(axis=1)).flatten()
        D = sparse.spdiags(abs_deg, 0, W.shape[0], W.shape[1])
        L = D - W

        if normalization == "normalized":
            D_inv_sqrt = sparse.diags(1.0 / np.sqrt(abs_deg + 1e-10))
            L = D_inv_sqrt @ L @ D_inv_sqrt

        return L

    def _fit(self, train_ind, train_labels, all_labels=None):
        """
        Fit signed Laplace learning scores from labeled training nodes.

        The method assumes binary labels in {-1, +1}. Labeled nodes are treated as hard
        boundary conditions, while unlabeled nodes are solved from the signed Laplace
        equation.

        The signed Laplacian is first constructed from the current graph. If order > 1,
        the operator is replaced by a higher-order power of the signed Laplacian.

        The unknown values on the unlabeled nodes are found by solving

            L_UU u_U = -L_UL u_L,

        where U is the set of unlabeled nodes and L is the set of labeled nodes.

        A diagonal preconditioned conjugate-gradient solve is used by default. If the
        system is tiny, degenerate, or the CG solution contains non-finite values, the
        method falls back to a direct sparse solve.

        Parameters
        ----------
        train_ind : array-like
            Indices of labeled training nodes.

        train_labels : array-like
            Binary labels for the training nodes. Must contain only -1 and +1.

        all_labels : array-like, optional
            Optional full label vector. Included for interface compatibility; not used
            by this implementation.

        Returns
        -------
        numpy.ndarray
            Score vector of shape (n,). Positive scores predict class +1 and negative
            scores predict class -1.
        """
        assert np.isin(train_labels, [-1, 1]).all()

        if self.reweighting == "none":
            G = self.graph
        else:
            W = self.graph.reweight( # type: ignore
                train_ind,
                method=self.reweighting,
                normalization=self.normalization,
                X=self.X,
            )
            G = graph.graph(W)

        n = G.num_nodes # type: ignore
        L = self.signed_laplacian(G, normalization=self.normalization)

        if self.order > 1:
            Lpow = L * L
            if self.order > 2:
                for _ in range(2, self.order):
                    Lpow = L * Lpow
            L = Lpow

        F = np.zeros(n)
        F[train_ind] = train_labels

        idx = np.full(n, True, dtype=bool)
        idx[train_ind] = False

        b = -L[:, train_ind].dot(F[train_ind])
        b = b[idx]

        A = L[idx, :][:, idx]

        m = A.shape[0]
        if m == 0:
            u = np.zeros(n)
            u[train_ind] = F[train_ind]
            if self.mean_shift:
                u -= np.mean(u)
            return u

        b_vec = np.asarray(b).ravel()
        if np.allclose(b_vec, 0.0):
            v = np.zeros(m)
        elif m == 1:
            a00 = float(A[0, 0])
            v = np.array([0.0]) if abs(a00) < 1e-14 else np.array([float(b_vec[0]) / a00])
        else:
            # Conjugate gradient is fast, but it can produce NaNs on some tiny/degenerate systems.
            # I fall back to a direct sparse solve if CG fails or returns non-finite values.
            M = A.diagonal()
            M = sparse.spdiags(1.0 / np.sqrt(M + 1e-10), 0, m, m).tocsr()
            rhs = M @ b
            try:
                v = utils.conjgrad(M @ A @ M, rhs, tol=self.tol)
                v = M @ v
                if not np.all(np.isfinite(v)):
                    raise FloatingPointError("Non-finite CG solution")
            except Exception:
                v = spsolve(A.tocsr(), b)
                v = np.asarray(v).ravel()

        u = np.zeros(n)
        u[idx] = v
        u[train_ind] = F[train_ind]

        if self.mean_shift:
            u -= np.mean(u)

        return u

