class EdgeModification:
        """
        This module contains functions for modifying graphs, such as adding or removing edges, changing edge weights, etc.
        """
        @staticmethod
        def set_edge_zero(A, i, j):
                """
                Set the weight of the edge between nodes i and j to zero.

                Parameters
                ----------
                A : scipy.sparse.spmatrix
                    The adjacency matrix of the graph in sparse format.
                i : int
                    The index of the first node.
                j : int
                    The index of the second node.

                Returns
                -------
                A_new : scipy.sparse.spmatrix
                    A new adjacency matrix with the specified edge weight set to zero.
                """
                A = A.tolil() 
                A[i, j] = 0 
                A[j, i] = 0 
                return A.tocsr() 
        
        @staticmethod
        def negate_edge(A, i, j):
                """
                Negate the weight of the edge between nodes i and j.

                Parameters
                ----------
                A : scipy.sparse.spmatrix
                    The adjacency matrix of the graph in sparse format.
                i : int
                    The index of the first node.
                j : int
                    The index of the second node.

                Returns
                -------
                A_new : scipy.sparse.spmatrix
                    A new adjacency matrix with the specified edge weight negated.
                """
                A = A.tolil() 
                A[i, j] = -A[i, j]
                A[j, i] = -A[j, i] 
                return A.tocsr() 