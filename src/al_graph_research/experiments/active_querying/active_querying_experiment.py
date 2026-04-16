import numpy as np
import scipy.sparse as sp
from al_graph_research.experiments.active_querying.run_state import RunState
from al_graph_research.experiments.active_querying.experiment_result import ExperimentResult
from al_graph_research.graphs.knn_graph import KNNGraph
from al_graph_research.graphs.signed_laplacian import SignedLaplacian
from al_graph_research.active_learning.laplace_labels import LaplaceLabels
from al_graph_research.graphs.graph_modifications import EdgeModification
from al_graph_research.graphs.graph_analysis import GraphAnalysis
from al_graph_research.experiments.only_edges.metrics import Metrics
from typing import Literal

class ActiveQueryingExperiment:
    def __init__(
        self,
        n_runs: int,
        num_rounds: int | Literal["max"],
        number_neighbors: int,
        kernel: str,
        alteration_strategy: str, # "zero" or "negate"
        graph_construction_method: str = "partner", # "legacy" or "partner"
        starting_idx = 1,
        ending_idx = 3,
        empty_val: int = 0,
        rng: np.random.Generator | None = None,
    ) -> None:
        self.n_runs = n_runs
        self.num_rounds = num_rounds
        self.number_neighbors = number_neighbors
        self.kernel = kernel
        if alteration_strategy not in {"zero", "negate"}:
            raise ValueError("alteration_strategy must be 'zero' or 'negate'.")
        else:
            self.alteration_strategy = alteration_strategy
        self.graph_construction_method = graph_construction_method
        self.starting_idx = starting_idx
        self.ending_idx = ending_idx
        self.empty_val = empty_val
        self.rng = rng if rng is not None else np.random.default_rng()

    def _initialize_run(self, dataset, run_id) -> RunState:
        data = dataset.data
        labels = dataset.labels
        N = len(labels)

        graph = KNNGraph(self.number_neighbors, self.kernel, data, construction_method=self.graph_construction_method)
        adjacency = graph.A_sp
        laplacian = SignedLaplacian(adjacency).L_signed
        _, embedding = graph.eigv_nd(laplacian, starting_idx=self.starting_idx, ending_idx=self.ending_idx)

        labeled_indices = self._select_initial_indices(dataset)
        
        y = np.full(N, int(self.empty_val), dtype=int)
        y[labeled_indices] = self._select_initial_indices(dataset)

        prediction = self._predict(adjacency, y, labels)
        acc = self._compute_accuracy(prediction, labels, y)

        run = RunState(
            run_id=run_id,
            graph=graph,
            adjacency=adjacency,
            laplacian=laplacian,
            embedding=embedding,
            y_train=y,
            labeled_indices=labeled_indices,
            accuracy_history=[acc],
            margin_history=[],
            delta_l2_history=[],
            lam1_history=[],
            lam2_history=[],
            embedding_history=[embedding.copy()],
            prediction_history=[prediction.copy()]
        )
        self._update_metrics(run, adjacency, prediction)
        return run

       

    
    def _predict(self, adjacency, y_train, true_labels) -> np.ndarray:
        scores = LaplaceLabels.labels_propagation(adjacency, y_train, self.empty_val)
        prediction = LaplaceLabels.laplaceClassifierWithVec(scores)

        acc_norm = np.mean(prediction == true_labels)
        acc_flip = np.mean(-prediction == true_labels)

        if acc_flip > acc_norm:
            prediction = -prediction

        return prediction

    def _compute_accuracy(self, prediction, true_labels, y_train) -> float:
        return LaplaceLabels.classifierAccuracy_Laplace_Vec(
            prediction,
            true_labels,
            method="unlabeled_only",
            labels=y_train,
            empty_val=self.empty_val,
        )
    
    def _update_metrics(self, run_state: RunState, adjacency, prediction=None) -> None:
        run_state.lam2_history.append(Metrics.lam2(adjacency))
        run_state.lam1_history.append(Metrics.lam1(adjacency))



    def _select_initial_indices(self, dataset) -> list[int]:
        if hasattr(dataset, "cluster_id") and dataset.cluster_id is not None:
            return self._select_gaussian_initial_indices(dataset)

        return self._select_generic_initial_indices(dataset)
    
    def _select_gaussian_initial_indices(self, dataset) -> list[int]:
        """
        Select one initial labeled point from each cluster in a Gaussian dataset.

        Assumes dataset.cluster_id exists and contains integer cluster labels.
        """
        cluster_ids = np.asarray(dataset.cluster_id)

        unique_clusters = np.unique(cluster_ids)
        if len(unique_clusters) < 2:
            raise ValueError("Expected at least 2 clusters.")

        selected_indices = []

        for cid in unique_clusters:
            cluster_indices = np.where(cluster_ids == cid)[0]
            chosen = int(self.rng.choice(cluster_indices))
            selected_indices.append(chosen)

        return selected_indices

    def _select_generic_initial_indices(self, dataset) -> list[int]:
        """
        Select one initial labeled point per class for a binary dataset.

        Assumes dataset.labels contains exactly two unique values.
        """
        labels = np.asarray(dataset.labels)
        unique_labels = np.unique(labels)

        if len(unique_labels) != 2:
            raise ValueError("Expected exactly 2 classes for generic selection.")

        selected_indices = []

        for lab in unique_labels:
            class_indices = np.where(labels == lab)[0]
            chosen = int(self.rng.choice(class_indices))
            selected_indices.append(chosen)

        return selected_indices

    def _run_round(self, run_state:RunState, round_idx: int, dataset)-> None:
        raise NotImplementedError("something")
