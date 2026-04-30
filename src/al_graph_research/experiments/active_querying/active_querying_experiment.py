import numpy as np
import scipy.sparse as sp
from al_graph_research.experiments.active_querying.run_state import RunState
from al_graph_research.experiments.active_querying.experiment_result import ExperimentResult
from al_graph_research.graphs.knn_graph import KNNGraph
from al_graph_research.graphs.signed_laplacian import SignedLaplacian
from al_graph_research.active_learning.laplace_labels import LaplaceLabels
from al_graph_research.graphs.graph_modifications import EdgeModification
from al_graph_research.experiments.metrics import Metrics

class ActiveQueryingExperiment:
    def __init__(
        self,
        n_runs: int,
        accuracy_level: float,
        number_neighbors: int,
        num_starting_labels_per_class: int,
        num_queries_per_round: int,
        kernel: str,
        alteration_strategy: str, # "zero" or "negate", "baseline"
        specified_starting_seeds: list[list[int]] | None = None, # List of lists of indices for each run, or None to use random selection
        graph_construction_method: str = "partner", # "legacy" or "partner"
        max_iterations: int = 500,
        starting_idx = 1,
        ending_idx = 3,
        empty_val: int = 0,
        rng: np.random.Generator | None = None,
    ) -> None:
        self.n_runs = n_runs
        self.accuracy_level = accuracy_level
        self.number_neighbors = number_neighbors
        self.num_starting_labels_per_class = num_starting_labels_per_class
        self.num_queries_per_round = num_queries_per_round
        self.kernel = kernel
        if alteration_strategy not in {"zero", "negate", "baseline"}:
            raise ValueError("alteration_strategy must be 'zero', 'negate' or 'baseline'.")
        else:
            self.alteration_strategy = alteration_strategy

        self.specified_starting_seeds = specified_starting_seeds
        self.graph_construction_method = graph_construction_method
        self.max_iterations = max_iterations
        self.starting_idx = starting_idx
        self.ending_idx = ending_idx
        self.empty_val = empty_val
        self.rng = rng if rng is not None else np.random.default_rng()


    def run(self, dataset) -> ExperimentResult:
        if self.specified_starting_seeds is not None:
            if len(self.specified_starting_seeds) != self.n_runs:
                raise ValueError("Length of specified_starting_seeds must match n_runs.")

            if hasattr(dataset, "cluster_id") and dataset.cluster_id is not None:
                expected = len(np.unique(dataset.cluster_id))
            else:
                expected = len(np.unique(dataset.labels))

            if any(len(seeds) != expected for seeds in self.specified_starting_seeds):
                raise ValueError(f"Each seed list must have {expected} indices.")

        runs = []
        for run_id in range(self.n_runs):
            run_state = self._initialize_run(dataset, run_id)
            num_rounds = 0
            while (run_state.accuracy_history[-1] < self.accuracy_level
            and num_rounds < self.max_iterations):
                self._run_round(run_state, dataset)
                num_rounds += 1
            runs.append(run_state)
        return ExperimentResult(runs=runs)


    def _initialize_run(self, dataset, run_id: int) -> RunState:
        data = dataset.data
        labels = dataset.labels
        N = len(labels)

        graph = KNNGraph(self.number_neighbors, self.kernel, data, construction_method=self.graph_construction_method)
        adjacency = graph.A_sp
        laplacian = SignedLaplacian(adjacency).L_signed
        _, embedding = graph.eigv_nd(laplacian, starting_idx=self.starting_idx, ending_idx=self.ending_idx)

        vals, vecs = Metrics.eig_k(adjacency, k=4)
        lam1 = vals[0]
        lam2 = vals[1]
        lam3 = vals[2]
        lam4 = vals[3]
        vec1 = vecs[:, 0]
        vec2 = vecs[:, 1]
        vec3 = vecs[:, 2]
        vec4 = vecs[:, 3]
        if self.specified_starting_seeds is not None:
            labeled_indices = self.specified_starting_seeds[run_id].copy()
        else:
            labeled_indices = self._select_initial_indices(dataset, self.num_starting_labels_per_class)
        labeled_indices_history = [labeled_indices.copy()]

        y = np.full(N, int(self.empty_val), dtype=int)
        y[labeled_indices] = labels[labeled_indices]

        prediction, _ = self._predict(adjacency, y, labels)
        acc = self._compute_accuracy(prediction, labels, y)

        run = RunState(
            run_id=run_id,
            graph=graph,
            adjacency=adjacency,
            laplacian=laplacian,
            embedding=embedding,
            altered_edges=set(),
            y_train=y,
            labeled_indices=labeled_indices,
            labeled_indices_history=labeled_indices_history,
            accuracy_history=[acc],
            lam1_history=[lam1],
            lam2_history=[lam2],
            lam3_history=[lam3],
            lam4_history=[lam4],
            vec1_history=[vec1],
            vec2_history=[vec2],
            vec3_history=[vec3],
            vec4_history=[vec4],
            embedding_history=[embedding.copy()],
            prediction_history=[prediction.copy()] # type: ignore
        )
        return run

       
    def _predict(self, adjacency, y_train, true_labels) -> tuple[np.ndarray, np.ndarray]:
        scores = LaplaceLabels.labels_propagation(adjacency, y_train, self.empty_val)
        prediction = LaplaceLabels.laplaceClassifierWithVec(scores)

        acc_norm = np.mean(prediction == true_labels)
        acc_flip = np.mean(-prediction == true_labels)

        if acc_flip > acc_norm:
            prediction = -prediction
            scores = -scores

        return prediction, scores

    def _compute_accuracy(self, prediction, true_labels, y_train) -> float:
        return LaplaceLabels.classifierAccuracy_Laplace_Vec(
            prediction,
            true_labels,
            method="unlabeled_only",
            labels=y_train,
            empty_val=self.empty_val,
        )
    
    def _update_metrics(self, run_state: RunState, adjacency, labels) -> None:
        vals, vecs = Metrics.eig_k(adjacency, k=4)

        run_state.lam1_history.append(vals[0])
        run_state.lam2_history.append(vals[1])
        run_state.lam3_history.append(vals[2])
        run_state.lam4_history.append(vals[3])
        run_state.vec1_history.append(vecs[:, 0])
        run_state.vec2_history.append(vecs[:, 1])
        run_state.vec3_history.append(vecs[:, 2])
        run_state.vec4_history.append(vecs[:, 3])



    def _select_initial_indices(self, dataset, k: int = 1) -> list[int]:
        if hasattr(dataset, "cluster_id") and dataset.cluster_id is not None:
            return self._select_gaussian_initial_indices(dataset, k)

        return self._select_generic_initial_indices(dataset, k)
    
    def _select_gaussian_initial_indices(self, dataset, k) -> list[int]:
        """
        Select k initial labeled points from each cluster in a Gaussian dataset.

        Assumes dataset.cluster_id exists and contains integer cluster labels.
        """
        cluster_ids = np.asarray(dataset.cluster_id)

        unique_clusters = np.unique(cluster_ids)
        if len(unique_clusters) < 2:
            raise ValueError("Expected at least 2 clusters.")

        selected_indices: list[int] = []

        for cid in unique_clusters:
            cluster_indices = np.where(cluster_ids == cid)[0]

            if len(cluster_indices) < k:
                raise ValueError(
                    f"Cluster {cid} has only {len(cluster_indices)} points; cannot select k={k}."
                )

            chosen = self.rng.choice(cluster_indices, size=k, replace=False)
            selected_indices.extend(chosen.astype(int).tolist())

        return selected_indices


    def _select_generic_initial_indices(self, dataset, k) -> list[int]:
        """
        Select k initial labeled points per class for a binary dataset.

        Assumes dataset.labels contains exactly two unique values.
        """
        labels = np.asarray(dataset.labels)
        unique_labels = np.unique(labels)

        if len(unique_labels) != 2:
            raise ValueError("Expected exactly 2 classes for generic selection.")

        selected_indices: list[int] = []

        for lab in unique_labels:
            class_indices = np.where(labels == lab)[0]

            if len(class_indices) < k:
                raise ValueError(
                    f"Class {lab} has only {len(class_indices)} points; cannot select k={k}."
                )

            chosen = self.rng.choice(class_indices, size=k, replace=False)
            selected_indices.extend(chosen.astype(int).tolist())

        return selected_indices



    def _run_round(self, run_state:RunState, dataset)-> None:
        labels = dataset.labels
        labeled_indices = run_state.labeled_indices
        edges = self.check_labeled_nodes(labeled_indices, labels, run_state.altered_edges)
        run_state.adjacency = self._apply_edge_alterations(run_state.adjacency, edges)
        run_state.laplacian = SignedLaplacian(run_state.adjacency).L_signed
        _, run_state.embedding = run_state.graph.eigv_nd(run_state.laplacian, starting_idx=self.starting_idx, ending_idx=self.ending_idx)
        run_state.embedding_history.append(run_state.embedding.copy())
        prediction, scores = self._predict(run_state.adjacency, run_state.y_train, labels)
        run_state.prediction_history.append(prediction.copy())
        acc = self._compute_accuracy(prediction, labels, run_state.y_train)
        run_state.accuracy_history.append(acc)
        uncertain_nodes = self._get_uncertain_nodes(scores, run_state.labeled_indices)
        self._label_uncertain_nodes(run_state, uncertain_nodes, labels)
        run_state.labeled_indices_history.append(run_state.labeled_indices.copy())
        self._update_metrics(run_state, run_state.adjacency, labels)


    def check_labeled_nodes( self, labeled_indices: list[int], 
                            labels: np.ndarray, altered_edges: set[tuple[int, int]]) -> list[tuple[int, int]]:
        edges: list[tuple[int, int]] = []
        for pos1, idx1 in enumerate(labeled_indices):
            for idx2 in labeled_indices[pos1 + 1:]:
                a = idx1
                b = idx2
                edge = (min(a, b), max(a, b)) # Ensure only one ordering of the edge is used

                if edge in altered_edges:
                    continue

                if labels[a] != labels[b]:
                    edges.append(edge)
                    altered_edges.add(edge)

        return edges

                    
    def _apply_edge_alterations(self, adjacency, edges) -> sp.spmatrix:
        A = adjacency.copy()
        for i, j in edges:
            if self.alteration_strategy == "zero":
                A = EdgeModification.set_edge_zero(A, i, j)
            elif self.alteration_strategy == "negate":
                A = EdgeModification.negate_edge(A, i, j)
            elif self.alteration_strategy == "baseline":
                pass  # No alteration for baseline
            else:
                raise ValueError(f"Unknown alteration strategy: {self.alteration_strategy}")
        return A
    
    def _get_uncertain_nodes(self, scores: np.ndarray, labeled_indices: list[int]) -> list[int]:
        score_values = np.asarray(scores).ravel()
        n = score_values.shape[0]

        unlabeled_mask = np.ones(n, dtype=bool)
        unlabeled_mask[labeled_indices] = False

        order = np.argsort(np.abs(score_values))
        order = order[unlabeled_mask[order]]

        return order[:self.num_queries_per_round].astype(int).tolist()
    
    def _label_uncertain_nodes(self, run_state: RunState, uncertain_nodes: list[int], true_labels: np.ndarray) -> None:
        for idx in uncertain_nodes:
            run_state.y_train[idx] = true_labels[idx]
            run_state.labeled_indices.append(idx)



        

