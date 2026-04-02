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
    y_train: np.ndarray
    labeled_indices: list[int]
    accuracy_history: list[float] = field(default_factory=list)
    margin_history: list[float] = field(default_factory=list)
    delta_l2_history: list[float] = field(default_factory=list)
    lam2_history: list[float] = field(default_factory=list)
    gap23_history: list[float] = field(default_factory=list)
    kappa_history: list[float] = field(default_factory=list)
    embedding_history: list[np.ndarray] = field(default_factory=list)
    prediction_history: list[np.ndarray] = field(default_factory=list)