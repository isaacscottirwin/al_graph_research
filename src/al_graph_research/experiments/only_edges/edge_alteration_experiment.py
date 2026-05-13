import numpy as np
import scipy.sparse as sp
from al_graph_research.experiments.only_edges.run_state import RunState
from al_graph_research.experiments.only_edges.experiment_result import ExperimentResult
from al_graph_research.graphs.knn_graph import KNNGraph
from al_graph_research.graphs.signed_laplacian import SignedLaplacian
from al_graph_research.active_learning.label_propagation import LabelPropagation
from al_graph_research.graphs.graph_modifications import EdgeModification
from al_graph_research.graphs.graph_analysis import GraphAnalysis
from al_graph_research.experiments.only_edges.batch_sequences import BatchSequences
from al_graph_research.experiments.metrics import Metrics
from typing import Literal

class EdgeAlterationExperiment:
    """
    Run an edge-only signed graph alteration experiment.

    This class manages repeated experiments for binary graph-based semi-supervised
    learning where the labeled set is fixed and only the graph structure changes
    over time. Each run constructs a kNN graph, selects an initial labeled set,
    predicts labels using either signed Laplace or signed Poisson learning, and
    then progressively alters positive cross-label edges in randomized batches.

    This experiment does not query new labels.
    The same initial labeled set is used for the entire run. The purpose of the
    experiment is to isolate the effect of edge alterations on the graph, the
    signed Laplacian, the spectral embedding, and the resulting SSL predictions.

    Before any run is initialized, the experiment constructs a graph from the
    dataset and identifies positive edges connecting nodes with opposite true
    labels. These cross-label edges are then shuffled into randomized batch
    sequences. Each run receives its own randomized sequence of edge batches.

    In each round, the experiment reads the next edge batch for the current run,
    alters those edges according to the selected strategy, recomputes the signed
    Laplacian and embedding, predicts labels using the selected SSL model, computes
    accuracy using the fixed labeled set, and updates the run histories.

    The per-round order is:

        1. Read the pre-generated edge batch for the current run and round.
        2. Apply the selected edge alteration strategy to that batch.
        3. Recompute the signed Laplacian from the updated adjacency matrix.
        4. Recompute and store the spectral embedding.
        5. Predict labels using the selected SSL model on the updated graph.
        6. Store predictions.
        7. Compute and store accuracy using the fixed labeled set.
        8. Store the current labeled-index history.
        9. Update eigenvalue and eigenvector histories.

    The labels are assumed to be binary and encoded as {-1, +1}. Unlabeled entries
    in the training vector are represented by `empty_val`, usually 0.

    Parameters
    ----------
    n_runs : int
        Number of independent randomized edge-alteration runs to perform.

    num_rounds : int or "max"
        Desired number of edge-alteration rounds.

        If "max", each round alters exactly one positive cross-label edge, so the
        number of rounds equals the number of available positive cross-label edges.

        If an integer, the available positive cross-label edges are divided into
        approximately that many batches.

    number_neighbors : int
        Number of neighbors used to construct the kNN graph.

    kernel : str
        Kernel name passed to `KNNGraph` when constructing the graph.

    alteration_strategy : {"zero", "negate"}
        Edge alteration rule applied to selected positive cross-label edges.

        - "zero":
            Delete the selected cross-label edge by setting its weight to zero.

        - "negate":
            Negate the selected cross-label edge weight, turning a positive
            cross-label edge into a negative signed edge.

    model_type : {"laplace", "poisson"}
        Semi-supervised learning model used to compute prediction scores.

        - "laplace":
            Uses signed Laplace learning.

        - "poisson":
            Uses signed Poisson learning.

    graph_construction_method : {"legacy", "partner"}, default="partner"
        Graph construction method passed to `KNNGraph`.

    starting_idx : int, default=1
        Starting eigenvector index used when constructing the spectral embedding.

    ending_idx : int, default=3
        Ending eigenvector index used when constructing the spectral embedding.

    empty_val : int, default=0
        Value used in `y_train` to mark unlabeled nodes.

    rng : numpy.random.Generator, optional
        Random number generator used for initial label selection and for generating
        randomized edge-batch seeds. If not provided, a new default generator is
        created.

    Attributes
    ----------
    n_runs : int
        Number of independent randomized runs.

    num_rounds : int or "max"
        Requested number of edge-alteration rounds.

    number_neighbors : int
        Number of neighbors in the kNN graph.

    kernel : str
        Kernel used for graph construction.

    alteration_strategy : str
        Selected edge alteration strategy.

    model_type : str
        Selected graph SSL model.

    graph_construction_method : str
        Graph construction method.

    starting_idx : int
        First embedding eigenvector index.

    ending_idx : int
        Last embedding eigenvector index.

    empty_val : int
        Unlabeled marker value.

    rng : numpy.random.Generator
        Random number generator used by the experiment.

    batch_sequences : list[list[list[tuple[int, int]]]]
        Generated edge batches after `run(dataset)` is called.

        The structure is:

            batch_sequences[run_id][round_idx] = list of edges altered that round.

    actual_num_rounds : int
        Actual number of edge-alteration rounds generated by the batching procedure.

    Returns
    -------
    None
        The constructor initializes the experiment configuration. Use `run(dataset)`
        to execute the experiment.
    """
    def __init__(
        self,
        n_runs: int,
        num_rounds: int | Literal["max"],
        number_neighbors: int,
        kernel: str,
        alteration_strategy: str, # "zero" or "negate"
        model_type: str, # "laplace" or "poisson"
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
        self.model_type = model_type
        self.graph_construction_method = graph_construction_method
        self.starting_idx = starting_idx
        self.ending_idx = ending_idx
        self.empty_val = empty_val
        self.rng = rng if rng is not None else np.random.default_rng()

    

    def run(self, dataset) -> ExperimentResult:
        """
        Execute the edge-only alteration experiment on a dataset.

        This method first generates randomized edge-batch sequences from the positive
        cross-label edges in the original graph. It then initializes each run and
        applies every edge-alteration round to every run.

        Parameters
        ----------
        dataset : object
            Dataset object containing at least:

            - data : array-like
                Feature matrix used to construct the kNN graph.

            - labels : array-like
                Binary labels in {-1, +1}.

            Optionally, the dataset may contain:

            - cluster_id : array-like
                Cluster identifiers used for cluster-balanced initial label selection.

        Returns
        -------
        ExperimentResult
            Object containing the RunState for each completed run.
        """
        self.batch_sequences, self.actual_num_rounds = self._generate_batch_sequences(dataset)
        runs = [self.initialize_run(dataset, run_id) for run_id in range(self.n_runs)]
        for round_idx in range(self.actual_num_rounds):
            for run_state in runs:
                self._run_round(run_state, round_idx, dataset)
        return ExperimentResult(runs=runs)
        
    def _generate_batch_sequences(self, dataset) -> tuple[list[list[list[tuple[int, int]]]], int]:
        """
        Generate randomized batches of positive cross-label edges.

        This method constructs the original kNN graph, extracts the positive
        cross-label edges using `GraphAnalysis.adjacency_block`, and then passes those
        edges to `BatchSequences` to create a randomized edge-batch schedule for each
        run.

        Parameters
        ----------
        dataset : object
            Dataset containing `data` and `labels`.

        Returns
        -------
        tuple[list[list[list[tuple[int, int]]]], int]
            batch_sequences :
                Randomized edge batches with structure

                    batch_sequences[run_id][round_idx] = list of edges.

            actual_num_rounds :
                Number of rounds produced by the batching procedure.

        Raises
        ------
        ValueError
            If no positive cross-label edges are found.
        """
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
        
    def initialize_run(self, dataset, run_id: int) -> RunState:
        """
        Initialize a single edge alteration run.

        This method constructs the initial graph, signed Laplacian, spectral embedding,
        eigenvalue/eigenvector metrics, fixed initial labeled set, training label
        vector, initial prediction, and initial accuracy.

        The labeled set is fixed for the entire edge-only experiment. Later rounds
        alter edges but do not add new labeled nodes.

        Parameters
        ----------
        dataset : object
            Dataset object with `data` and `labels` attributes. May also include
            `cluster_id`.

        run_id : int
            Identifier for the run being initialized.

        Returns
        -------
        RunState
            Initialized run state containing the graph, adjacency, Laplacian,
            embedding, training labels, labeled indices, and initial histories.
        """
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
        labeled_indices = self._select_initial_indices(dataset)
        labeled_indicies_history = labeled_indices.copy()

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
            labeled_indices_history=[labeled_indicies_history],
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
            prediction_history=[prediction.copy()]
        )
        return run

    def _predict(self, adjacency, y_train, true_labels) -> np.ndarray:
        """
        Predict labels for the current graph state.

        This method computes model scores using `LabelPropagation.labels_propagation`
        with the selected model type, converts the scores to binary predictions, and
        then resolves the global sign ambiguity by comparing the prediction to its
        global sign flip.

        Parameters
        ----------
        adjacency : scipy.sparse.spmatrix
            Current graph adjacency matrix.

        y_train : numpy.ndarray
            Fixed training label vector.

        true_labels : numpy.ndarray
            Ground-truth labels for all nodes.

        Returns
        -------
        numpy.ndarray
            Binary predicted labels in {-1, +1}.
        """
        scores = LabelPropagation.labels_propagation(adjacency, y_train, self.empty_val, model_type=self.model_type)
        prediction = LabelPropagation.classifier_with_vec(scores)

        acc_norm = np.mean(prediction == true_labels)
        acc_flip = np.mean(-prediction == true_labels)

        if acc_flip > acc_norm:
            prediction = -prediction

        return prediction

    def _compute_accuracy(self, prediction, true_labels, y_train) -> float:
        """
        Compute classification accuracy over unlabeled nodes.

        The fixed labeled nodes are excluded from the accuracy computation by using
        `LabelPropagation.classifier_accuracy_vec` with method="unlabeled_only".

        Parameters
        ----------
        prediction : numpy.ndarray
            Binary predicted labels.

        true_labels : numpy.ndarray
            Ground-truth labels.

        y_train : numpy.ndarray
            Training label vector containing labels on the fixed labeled set and
            `empty_val` elsewhere.

        Returns
        -------
        float
            Accuracy over unlabeled nodes.
        """
        return LabelPropagation.classifier_accuracy_vec(
            prediction,
            true_labels,
            method="unlabeled_only",
            labels=y_train,
            empty_val=self.empty_val,
        )
    
    def _update_metrics(self, run_state: RunState, adjacency) -> None:
        """
        Update eigenvalue and eigenvector histories for a run.

        This method computes the first four tracked eigenvalues and eigenvectors from
        the current adjacency matrix and appends them to the corresponding histories
        in the RunState.

        Parameters
        ----------
        run_state : RunState
            Run state whose spectral histories are updated.

        adjacency : scipy.sparse.spmatrix
            Current adjacency matrix.

        Returns
        -------
        None
            Updates the RunState in place.
        """
        vals, vecs = Metrics.eig_k(adjacency, k=4)

        run_state.lam1_history.append(vals[0])
        run_state.lam2_history.append(vals[1])
        run_state.lam3_history.append(vals[2])
        run_state.lam4_history.append(vals[3])
        run_state.vec1_history.append(vecs[:, 0])
        run_state.vec2_history.append(vecs[:, 1])
        run_state.vec3_history.append(vecs[:, 2])
        run_state.vec4_history.append(vecs[:, 3])

    def _select_initial_indices(self, dataset) -> list[int]:
        """
        Select the fixed initial labeled set for one run.

        If the dataset contains a non-None `cluster_id` attribute, this method selects
        one point from each cluster. Otherwise, it selects one point from each class.

        Parameters
        ----------
        dataset : object
            Dataset containing `labels` and optionally `cluster_id`.

        Returns
        -------
        list[int]
            Initial labeled node indices.
        """
        if hasattr(dataset, "cluster_id") and dataset.cluster_id is not None:
            return self._select_gaussian_initial_indices(dataset)

        return self._select_generic_initial_indices(dataset)

    def _select_gaussian_initial_indices(self, dataset) -> list[int]:
        """
        Select one initial labeled point from each cluster.

        This method is intended for Gaussian-style datasets with a `cluster_id`
        attribute. One point is sampled uniformly at random from each cluster.

        Parameters
        ----------
        dataset : object
            Dataset containing a `cluster_id` attribute.

        Returns
        -------
        list[int]
            Selected initial labeled indices.

        Raises
        ------
        ValueError
            If fewer than two clusters are present.
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
        Select one initial labeled point from each class.

        This method assumes the dataset has exactly two unique class labels. One point
        is sampled uniformly at random from each class.

        Parameters
        ----------
        dataset : object
            Dataset containing a `labels` attribute.

        Returns
        -------
        list[int]
            Selected initial labeled indices.

        Raises
        ------
        ValueError
            If the dataset does not contain exactly two classes.
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
        """
        Run one edge alteration round for a single RunState.

        This method updates the run state in place. It retrieves the pre-generated
        edge batch for the current run and round, applies the selected alteration
        strategy, recomputes graph-dependent quantities, evaluates the current model,
        and records all histories.

        No new labels are queried or added in this method. The labeled set remains
        fixed throughout the experiment.

        The order of operations is:

            1. Retrieve the current edge batch.
            2. Apply edge alterations to the current adjacency matrix.
            3. Recompute the signed Laplacian.
            4. Recompute and store the spectral embedding.
            5. Predict labels on the updated graph.
            6. Store the prediction history.
            7. Compute and store accuracy using the fixed labeled set.
            8. Store the current labeled-index history.
            9. Update eigenvalue and eigenvector histories.

        Parameters
        ----------
        run_state : RunState
            Current state of one experiment run. It is modified in place.

        round_idx : int
            Index of the current edge-alteration round.

        dataset : object
            Dataset containing the ground-truth labels.

        Returns
        -------
        None
            Updates the RunState in place.
        """
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
        run_state.labeled_indices_history.append(run_state.labeled_indices.copy())
        self._update_metrics(run_state, run_state.adjacency)

    def _apply_edge_alterations(self, adjacency, edges) -> sp.spmatrix:
        """
        Apply the selected edge alteration strategy to a batch of edges.

        The adjacency matrix is copied before modification. Each edge in the batch is
        then either zeroed out or negated, depending on `self.alteration_strategy`.

        Parameters
        ----------
        adjacency : scipy.sparse.spmatrix
            Current adjacency matrix.

        edges : list[tuple[int, int]]
            Edge batch to alter.

        Returns
        -------
        scipy.sparse.spmatrix
            Modified adjacency matrix.

        Raises
        ------
        ValueError
            If the alteration strategy is unknown.
        """
        A = adjacency.copy()
        for i, j in edges:
            if self.alteration_strategy == "zero":
                A = EdgeModification.set_edge_zero(A, i, j)
            elif self.alteration_strategy == "negate":
                A = EdgeModification.negate_edge(A, i, j)
            else:
                raise ValueError(f"Unknown alteration strategy: {self.alteration_strategy}")
        return A

