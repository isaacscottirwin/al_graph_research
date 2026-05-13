from __future__ import annotations
from dataclasses import dataclass, field
import scipy.sparse as sp
import numpy as np
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from al_graph_research.graphs.knn_graph import KNNGraph
@dataclass
class RunState:
    """
    Container for the full state of a single active querying experiment run.

    This dataclass stores the current graph objects, current training labels,
    current embedding, and all metric histories generated throughout one run of an
    active learning experiment.

    The histories stored in this object allow later visualization and analysis of
    how the graph structure, embeddings, eigenvalues, eigenvectors, predictions,
    and accuracy evolve over time as additional labels are queried and graph edges
    are altered.

    Attributes
    ----------
    run_id : int
        Identifier for the run.

    graph : KNNGraph
        Graph object used during the run.

    adjacency : scipy.sparse.spmatrix
        Current adjacency matrix of the graph.

    laplacian : scipy.sparse.spmatrix
        Current signed graph Laplacian.

    embedding : numpy.ndarray
        Current spectral embedding of the graph nodes.

    y_train : numpy.ndarray
        Current training label vector.

        Labeled nodes contain labels in {-1, +1}, while unlabeled nodes contain
        the experiment's empty-value marker, typically 0.

    labeled_indices : list[int]
        Current indices of labeled nodes.

    labeled_indices_history : list[list[int]]
        History of labeled-node sets over time.

    accuracy_history : list[float]
        Classification accuracy history over the experiment rounds.

    lam1_history : list[float]
        History of the first tracked eigenvalue.

    lam2_history : list[float]
        History of the second tracked eigenvalue.

    lam3_history : list[float]
        History of the third tracked eigenvalue.

    lam4_history : list[float]
        History of the fourth tracked eigenvalue.

    vec1_history : list[numpy.ndarray]
        History of the first tracked eigenvector.

    vec2_history : list[numpy.ndarray]
        History of the second tracked eigenvector.

    vec3_history : list[numpy.ndarray]
        History of the third tracked eigenvector.

    vec4_history : list[numpy.ndarray]
        History of the fourth tracked eigenvector.

    embedding_history : list[numpy.ndarray]
        History of graph embeddings over time.

    prediction_history : list[numpy.ndarray]
        History of predicted label vectors over time.
    """
    run_id: int
    graph: KNNGraph
    adjacency: sp.spmatrix
    laplacian: sp.spmatrix
    embedding: np.ndarray
    y_train: np.ndarray
    labeled_indices: list[int]
    labeled_indices_history: list[list[int]] = field(default_factory=list)
    accuracy_history: list[float] = field(default_factory=list)
    lam1_history: list[float] = field(default_factory=list)
    lam2_history: list[float] = field(default_factory=list)
    lam3_history: list[float] = field(default_factory=list)
    lam4_history: list[float] = field(default_factory=list)
    vec1_history: list[np.ndarray] = field(default_factory=list)
    vec2_history: list[np.ndarray] = field(default_factory=list)
    vec3_history: list[np.ndarray] = field(default_factory=list)
    vec4_history: list[np.ndarray] = field(default_factory=list)
    embedding_history: list[np.ndarray] = field(default_factory=list)
    prediction_history: list[np.ndarray] = field(default_factory=list)