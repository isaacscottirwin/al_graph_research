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
    Container for all state tracked during a single experiment run.

    This dataclass stores the graph objects, current labeled set, current spectral
    objects, altered edges, and all per-step histories produced during one run of
    an active learning or graph alteration experiment.

    Attributes
    ----------
    run_id : int
        Identifier for the run.

    graph : KNNGraph
        Graph object used in the run.

    adjacency : scipy.sparse.spmatrix
        Current adjacency matrix.

    laplacian : scipy.sparse.spmatrix
        Current graph Laplacian.

    embedding : numpy.ndarray
        Current embedding of the graph nodes.

    altered_edges : set[tuple[int, int]]
        Set of graph edges altered during the run.

    y_train : numpy.ndarray
        Training label vector.

    labeled_indices : list[int]
        Current list of labeled node indices.

    labeled_indices_history : list[list[int]]
        History of labeled node sets over time.

    accuracy_history : list[float]
        Accuracy recorded at each step.

    lam1_history, lam2_history, lam3_history, lam4_history : list[float]
        Histories of the first four tracked eigenvalues.

    vec1_history, vec2_history, vec3_history, vec4_history : list[numpy.ndarray]
        Histories of the first four tracked eigenvectors.

    embedding_history : list[numpy.ndarray]
        History of node embeddings over time.

    prediction_history : list[numpy.ndarray]
        History of predicted labels over time.
    """
    run_id: int
    graph: KNNGraph
    adjacency: sp.spmatrix
    laplacian: sp.spmatrix
    embedding: np.ndarray
    altered_edges: set[tuple[int, int]]
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