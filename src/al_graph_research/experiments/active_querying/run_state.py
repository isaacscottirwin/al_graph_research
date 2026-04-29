from __future__ import annotations
from dataclasses import dataclass, field
import scipy.sparse as sp
import numpy as np
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from al_graph_research.graphs.knn_graph import KNNGraph
@dataclass
class RunState:
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