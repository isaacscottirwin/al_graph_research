import numpy as np
import scipy.sparse as sp
from al_graph_research.experiments.only_edges.run_state import RunState
from al_graph_research.experiments.only_edges.experiment_result import ExperimentResult
from al_graph_research.graphs.knn_graph import KNNGraph
from al_graph_research.graphs.signed_laplacian import SignedLaplacian
from al_graph_research.active_learning.laplace_labels import LaplaceLabels
from al_graph_research.graphs.graph_modifications import EdgeModification
from al_graph_research.graphs.graph_analysis import GraphAnalysis
from al_graph_research.experiments.only_edges.batch_sequences import BatchSequences
from al_graph_research.experiments.only_edges.metrics import Metrics
from typing import Literal

class EdgeAlterationExperiment:
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

    

    def run(self, dataset) -> ExperimentResult:
        self.batch_sequences, self.actual_num_rounds = self._generate_batch_sequences(dataset)
        runs = [self._initialize_run(dataset, run_id) for run_id in range(self.n_runs)]
        for round_idx in range(self.actual_num_rounds):
            for run_state in runs:
                self._run_round(run_state, round_idx, dataset)
        return ExperimentResult(runs=runs)
        
    def _generate_batch_sequences(self, dataset) -> tuple[list[list[list[tuple[int, int]]]], int]:
        graph = KNNGraph(self.number_neighbors, self.kernel, dataset.data, construction_method=self.graph_construction_method)
        adjacency = graph.A_sp
        _, _, _, cross_edges = GraphAnalysis.adjacency_block(adjacency, dataset.labels)

        if len(cross_edges) == 0:
            raise ValueError("No positive cross-label edges found, so no edge batches can be generated.")

        batch_gen = BatchSequences(
            cross_edges,
            self.n_runs,
            self.num_rounds,
            seed=int(self.rng.integers(1_000_000_000)),
        )
        batch_sequences, _, actual_num_rounds, _ = batch_gen.generate()

        return batch_sequences, actual_num_rounds
        
    def _initialize_run(self, dataset, run_id: int) -> RunState:
        data = dataset.data
        labels = dataset.labels
        N = len(labels)

        graph = KNNGraph(self.number_neighbors, self.kernel, data, construction_method=self.graph_construction_method)
        adjacency = graph.A_sp
        laplacian = SignedLaplacian(adjacency).L_signed
        _, embedding = graph.eigv_nd(laplacian, starting_idx=self.starting_idx, ending_idx=self.ending_idx)

        labeled_indices = self._select_initial_indices(dataset)

        y = np.full(N, int(self.empty_val), dtype=int)
        y[labeled_indices] = labels[labeled_indices]

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
            lam3_history=[],
            lam4_history=[],
            gap23_history=[],
            kappa_history=[],
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
        run_state.lam2_history.append(Metrics.lam_k(adjacency, k=2))
        run_state.lam1_history.append(Metrics.lam_k(adjacency, k=1))
        run_state.lam3_history.append(Metrics.lam_k(adjacency, k=3))
        run_state.lam4_history.append(Metrics.lam_k(adjacency, k=4))
        run_state.gap23_history.append(Metrics.gap23(adjacency))

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

    def _run_round(self, run_state: RunState, round_idx: int, dataset) -> None:
        labels = dataset.labels
        edges = self.batch_sequences[run_state.run_id][round_idx]
        run_state.adjacency = self._apply_edge_alterations(run_state.adjacency, edges)
        run_state.laplacian = SignedLaplacian(run_state.adjacency).L_signed
        _, run_state.embedding = run_state.graph.eigv_nd(run_state.laplacian, starting_idx=self.starting_idx, ending_idx=self.ending_idx)
        run_state.embedding_history.append(run_state.embedding.copy())
        prediction = self._predict(run_state.adjacency, run_state.y_train, labels)
        run_state.prediction_history.append(prediction.copy())
        acc = self._compute_accuracy(prediction, labels, run_state.y_train)
        run_state.accuracy_history.append(acc)
        self._update_metrics(run_state, run_state.adjacency, prediction)
    
    def _apply_edge_alterations(self, adjacency, edges) -> sp.spmatrix:
        A = adjacency.copy()
        for i, j in edges:
            if self.alteration_strategy == "zero":
                A = EdgeModification.set_edge_zero(A, i, j)
            elif self.alteration_strategy == "negate":
                A = EdgeModification.negate_edge(A, i, j)
            else:
                raise ValueError(f"Unknown alteration strategy: {self.alteration_strategy}")
        return A

