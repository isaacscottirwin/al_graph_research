import numpy as np
from al_graph_research.graphs.utils import to_sparse
import scipy.sparse as sp # type: ignore

class GraphAnalysis:
    @staticmethod
    def adjacency_block(A, labels):
        """
        Reorder an adjacency matrix so nodes are grouped by label, then extract the
        cross-label block and its positive-weight edges.

        Parameters
        ----------
        A : scipy.sparse.spmatrix or array-like of shape (n, n)
            Adjacency matrix of the graph.
        labels : array-like of shape (n,)
            Binary node labels. The actual label values do not matter, but there
            must be exactly two distinct labels.

        Returns
        -------
        A_ordered : scipy.sparse.spmatrix
            Adjacency matrix reordered so all nodes from one label group come first
            and all nodes from the other label group come second.
        order : ndarray of shape (n,)
            Permutation array giving the reordered node indices.
        split : int
            Index where the reordered matrix switches from the first label group to
            the second label group.
        cross_edges : list[tuple[int, int]]
            List of positive-weight cross-label edges in the form
            (original_row_index, original_col_index).
        """
        labels = np.asarray(labels)

        if labels.ndim != 1:
            raise ValueError("labels must be a 1D array.")

        A = to_sparse(A)

        if A.shape[0] != A.shape[1]: # type: ignore
            raise ValueError("A must be square.")
        if A.shape[0] != len(labels): # type: ignore
            raise ValueError("labels must have the same length as the size of A.")

        unique_labels = np.unique(labels)
        if len(unique_labels) != 2:
            raise ValueError("labels must contain exactly two distinct values.")

        order = np.argsort(labels)
        labels_sorted = labels[order]
        split = np.sum(labels_sorted == unique_labels[0])

        A_ordered = A[order][:, order]

        block = A_ordered[:split, split:].tocoo()

        cross_edges = [
            (int(order[i]), int(order[split + j]))
            for i, j, w in zip(block.row, block.col, block.data)
            if w > 0
        ]

        return A_ordered, order, split, cross_edges

    @staticmethod
    def neighbors_in_other_class(A, labels, index):
        """
        Return the neighbors of a node whose labels differ from that node's label.

        Parameters
        ----------
        A : scipy.sparse.spmatrix
            Sparse adjacency matrix of shape (n, n).
        labels : array-like of shape (n,)
            Class labels for the nodes.
        index : int
            Node index whose neighbors are to be checked.

        Returns
        -------
        list[int]
            Indices of neighbors of `index` whose labels differ from
            `labels[index]`.

        Notes
        -----
        This function only inspects the nonzero entries in row `index`, so it is
        efficient for sparse graphs.
        """
        labels = np.asarray(labels)

        if A.shape[0] != A.shape[1]:
            raise ValueError("A must be square.")
        if A.shape[0] != len(labels):
            raise ValueError("labels must have the same length as the size of A.")
        if not (0 <= index < A.shape[0]):
            raise IndexError("index is out of bounds.")
        if not sp.issparse(A):
            A = sp.csr_matrix(A)
        else:
            A = A.tocsr()

        row = A.getrow(index)
        neighbor_indices = row.indices
        pt_label = labels[index]

        return neighbor_indices[labels[neighbor_indices] != pt_label].tolist()
    
    # modularity and community detection methods, I don't know how relevant these are to the rest of the project 
    # but they are useful for graph analysis and testing, so I included them here. I learned about 
    # modularity in my graph data science class and thought it would be interesting to implement. 
    @staticmethod
    def _validate_partition_vector(A, labels):
        """
        Validate and convert a binary partition label vector into a sign vector.

        Parameters
        ----------
        A : scipy.sparse.spmatrix
            Sparse adjacency matrix of shape (n, n).
        labels : array-like of shape (n,)
            Binary partition labels. 
            - a sign vector with entries in {-1, 1}.

        Returns
        -------
        s : ndarray of shape (n,)
            Sign vector with entries in {-1.0, 1.0}.

        Raises
        ------
        ValueError
            If the adjacency matrix is not initialized, if the label vector has the
            wrong length, or if it is not binary.
        """
        if A is None:
            raise ValueError("Adjacency matrix is not initialized.")

        n = A.shape[0]
        labels = np.asarray(labels).ravel()

        if labels.shape[0] != n:
            raise ValueError("Label vector must have length equal to the number of nodes.")

        unique = set(np.unique(labels))

        if unique.issubset({-1, 1}):
            s = labels.astype(float)
        elif unique.issubset({0, 1}):
            s = np.where(labels == 1, 1.0, -1.0)
        else:
            raise ValueError("Labels must be binary: either in {0,1} or in {-1,1}.")

        return s

    @staticmethod
    def modularity_score_for_partition(A, labels):
        """
        Compute the standard unsigned modularity score for a given binary partition.

        Mathematical background
        -----------------------
        Let A be the adjacency matrix of an undirected graph, let

            k_i = sum_j A_ij

        be the degree of node i, and let

            2m = sum_i k_i

        be twice the total edge weight.

        For a bipartition encoded by a sign vector s in {-1, 1}^n, the modularity
        score is

            Q = (1 / 4m) s^T B s,

        where

            B = A - (k k^T) / (2m).

        Expanding this gives

            Q = (1 / 4m) [ s^T A s - (s^T k)^2 / (2m) ].

        This method computes the modularity score of a user-supplied partition
        rather than the partition induced by the leading eigenvector of B.

        Parameters
        ----------
        A : scipy.sparse.spmatrix
            Sparse adjacency matrix of shape (n, n).
        labels : array-like of shape (n,)
            Binary node labels defining the partition. 
            - entries in {-1, 1}.

        Returns
        -------
        Q : float
            Modularity score of the given partition.

        Raises
        ------
        ValueError
            If the graph is invalid, if the adjacency matrix is not symmetric, if
            the graph has no edges, or if the labels are invalid.
        """
        if A is None:
            raise ValueError("Adjacency matrix is not initialized.")

        A = A.tocsr().astype(float)

        if (A != A.T).nnz != 0:
            raise ValueError("Adjacency matrix must be symmetric.")

        s = GraphAnalysis._validate_partition_vector(A, labels)

        k = np.asarray(A.sum(axis=1)).ravel()
        m = float(k.sum() / 2.0)

        if m <= 0:
            raise ValueError("Modularity is undefined for a graph with no edges.")

        As = A @ s
        sAs = float(s @ As)
        stk = float(s @ k)

        Q = (sAs - (stk ** 2) / (2.0 * m)) / (4.0 * m)
        return float(Q)

    @staticmethod
    def signed_modularity_score_for_partition(A, labels):
        """
        Compute a signed modularity score for a given binary partition.

        Mathematical background
        -----------------------
        For an undirected signed graph with adjacency matrix A, write

            A = A^+ - A^-,

        where

            A^+_ij = max(A_ij, 0),
            A^-_ij = max(-A_ij, 0).

        Define the positive and negative degree vectors by

            k_i^+ = sum_j A^+_ij,
            k_i^- = sum_j A^-_ij,

        and let

            2m^+ = sum_i k_i^+,
            2m^- = sum_i k_i^-.

        A standard signed modularity matrix is

            B_s
            =
            (A^+ - (k^+ (k^+)^T) / (2m^+))
            -
            (A^- - (k^- (k^-)^T) / (2m^-)).

        Equivalently,

            B_s
            =
            A
            - (k^+ (k^+)^T) / (2m^+)
            + (k^- (k^-)^T) / (2m^-).

        For a bipartition encoded by a sign vector s in {-1,1}^n, this method
        computes

            Q = (1 / 4) s^T B_s s.

        Expanding gives

            Q = (1 / 4) [
                    s^T A s
                    - (s^T k^+)^2 / (2m^+)
                    + (s^T k^-)^2 / (2m^-)
                ],

        where the positive and negative correction terms are included only when
        m^+ > 0 and m^- > 0, respectively.

        Parameters
        ----------
        A : scipy.sparse.spmatrix
            Sparse adjacency matrix of shape (n, n).
        labels : array-like of shape (n,)
            Binary node labels defining the partition. 
            - entries in {-1, 1}.

        Returns
        -------
        Q : float
            Signed modularity score of the given partition.

        Raises
        ------
        ValueError
            If the graph is invalid, if the adjacency matrix is not symmetric, if
            the graph has no edges, or if the labels are invalid.
        """
        
        if A is None:
            raise ValueError("Adjacency matrix is not initialized.")

        A = A.tocsr().astype(float)

        if (A != A.T).nnz != 0:
            raise ValueError("Adjacency matrix must be symmetric.")

        s = GraphAnalysis._validate_partition_vector(A, labels)

        A_pos = A.copy()
        A_pos.data = np.maximum(A_pos.data, 0.0)
        A_pos.eliminate_zeros()

        A_neg = A.copy()
        A_neg.data = np.maximum(-A_neg.data, 0.0)
        A_neg.eliminate_zeros()

        k_pos = np.asarray(A_pos.sum(axis=1)).ravel()
        k_neg = np.asarray(A_neg.sum(axis=1)).ravel()

        m_pos = float(k_pos.sum() / 2.0)
        m_neg = float(k_neg.sum() / 2.0)

        if m_pos <= 0 and m_neg <= 0:
            raise ValueError("Signed modularity is undefined for a graph with no edges.")

        As = A @ s
        sAs = float(s @ As)

        Q = sAs

        if m_pos > 0:
            stk_pos = float(s @ k_pos)
            Q -= (stk_pos ** 2) / (2.0 * m_pos)

        if m_neg > 0:
            stk_neg = float(s @ k_neg)
            Q += (stk_neg ** 2) / (2.0 * m_neg)

        Q /= 4.0
        return float(Q)


    @staticmethod
    def communities_from_labels(A, labels):
        """
        Convert a binary node label vector into two communities.

        Parameters
        ----------
        A : scipy.sparse.spmatrix
            Sparse adjacency matrix of shape (n, n).
        labels : array-like of shape (n,)
            Binary node labels. 
            - entries in {-1, 1}.

        Returns
        -------
        community1 : set[int]
            Nodes whose sign is +1.
        community2 : set[int]
            Nodes whose sign is -1.
        """
        s = GraphAnalysis._validate_partition_vector(A, labels)
        community1 = set(np.flatnonzero(s == 1.0))
        community2 = set(np.flatnonzero(s == -1.0))
        return community1, community2