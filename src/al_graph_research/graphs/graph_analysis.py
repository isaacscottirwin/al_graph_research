import numpy as np
from al_graph_research.graphs.utils import to_sparse
import scipy.sparse as sp # type: ignore

class GraphAnalysis:
    @staticmethod
    def adjacency_block(A, labels):
        """
        Reorder an adjacency matrix so nodes are grouped by class label, then
        extract the cross-class block and its positive-weight edges.

        Parameters
        ----------
        A : scipy.sparse.spmatrix
            Sparse adjacency matrix of shape (n, n).
        labels : array-like of shape (n,)
            Binary class labels for the nodes. This function assumes the first
            group is label 0 and the second group is label 1.

        Returns
        -------
        A_ordered : scipy.sparse.spmatrix
            Adjacency matrix reordered so all class-0 nodes come first and all
            class-1 nodes come second.
        order : ndarray of shape (n,)
            Permutation array giving the reordered node indices.
        split : int
            Index where the reordered matrix switches from class 0 to class 1.
        cross_edges : list[tuple[int, int, float]]
            List of positive-weight cross-class edges in the form
            (original_row_index, original_col_index, weight).

        Notes
        -----
        The cross block is the upper-right block of the reordered adjacency
        matrix, corresponding to edges from class 0 to class 1.
        """
        labels = np.asarray(labels)

        if A.shape[0] != A.shape[1]:
            raise ValueError("A must be square.")
        if A.shape[0] != len(labels):
            raise ValueError("labels must have the same length as the size of A.")
        # make sure A is sparse
        A = to_sparse(A)

        class0 = np.flatnonzero(labels == 0)
        class1 = np.flatnonzero(labels == 1)

        if len(class0) + len(class1) != len(labels):
            raise ValueError("labels must be binary with values 0 and 1.")

        order = np.concatenate((class0, class1))
        split = len(class0)

        A_ordered = A[order][:, order]

        block = A_ordered[:split, split:].tocoo()

        cross_edges = [
            (order[i], order[split + j], w)
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