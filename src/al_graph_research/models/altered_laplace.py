from graphlearning.ssl import ssl
from scipy import sparse
from scipy.sparse.linalg import spsolve
import graphlearning.utils as utils
import graphlearning.graph as graph
import numpy as np


class AlteredLaplace(ssl):
    def __init__(
        self,
        W=None,
        class_priors=None,
        X=None,
        reweighting="none",
        normalization="combinatorial",
        tau=0,
        order=1,
        mean_shift=False,
        tol=1e-5,
        alpha=2,
        zeta=1e7,
        r=0.1,
    ):
        """Laplace Learning
        ===================

        Semi-supervised learning via the solution of the Laplace equation
        \\[\\tau u_j + L^m u_j = 0, \\ \\ j \\geq m+1,\\]
        subject to \\(u_j = y_j\\) for \\(j=1,\\dots,m\\), where \\(L=D-W\\) is the
        combinatorial graph Laplacian and \\(y_j\\) for \\(j=1,\\dots,m\\) are the
        label vectors. Default order is m=1, and m > 1 corresponds to higher order Laplace Learning.

        The original method was introduced in [1]. This class also implements reweighting
        schemes `poisson` proposed in [2], `wnll` proposed in [3], and `properly`, proposed in [4].
        If `properly` is selected, the user must additionally provide the data features `X`.

        Parameters
        ----------
        W : numpy array, scipy sparse matrix, or graphlearning graph object (optional), default=None
            Weight matrix representing the graph.
        class_priors : numpy array (optional), default=None
            Class priors (fraction of data belonging to each class). If provided, the predict function
            will attempt to automatic balance the label predictions to predict the correct number of
            nodes in each class.
        X : numpy array (optional)
            Data features, used to construct the graph. This is required for the `properly` weighted
            graph Laplacian method.
        normalization : {'combinatorial','randomwalk','normalized'} (optional), defualt='combinatorial'
            Normalization for the graph Laplacian.
        reweighting : {'none', 'wnll', 'poisson', 'properly'} (optional), default='none'
            Reweighting scheme for low label rate problems. If 'properly' is selected, the user
            must supply the data features `X`.
        tau : float or numpy array (optional), default=0
            Zeroth order term in Laplace equation. Can be a scalar or vector.
        order : integer (optional), default=1
            Power m for higher order Laplace learning. Currently only integers are allowed.
        mean_shift : bool (optional), default=False
            Whether to shift output to mean zero.
        tol : float (optional), default=1e-5
            Tolerance for conjugate gradient solver.
        alpha : float (optional), default=2
            Parameter for `properly` reweighting.
        zeta : float (optional), default=1e7
            Parameter for `properly` reweighting.
        r : float (optional), default=0.1
            Radius for `properly` reweighting.

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

