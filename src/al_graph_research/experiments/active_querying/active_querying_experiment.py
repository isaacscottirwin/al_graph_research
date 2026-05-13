import numpy as np
import scipy.sparse as sp
from al_graph_research.experiments.active_querying.run_state import RunState
from al_graph_research.experiments.active_querying.experiment_result import ExperimentResult
from al_graph_research.graphs.knn_graph import KNNGraph
from al_graph_research.graphs.signed_laplacian import SignedLaplacian
from al_graph_research.active_learning.label_propagation import LabelPropagation
from al_graph_research.graphs.graph_modifications import EdgeModification
from al_graph_research.experiments.metrics import Metrics

class ActiveQueryingExperiment:
    """
    Run an active querying experiment on a signed k-nearest-neighbor graph.

    This class manages repeated active learning experiments for binary graph-based
    semi-supervised learning. Each run constructs a kNN graph, selects an initial
    labeled set, predicts labels using either signed Laplace or signed Poisson
    learning, queries new nodes, and optionally alters labeled-labeled cross-class
    edges.

    In each round, the experiment first uses the current labeled set to identify
    new labeled-labeled cross-class edges. Those edges are altered according to the
    selected strategy. The signed Laplacian and embedding are then recomputed from
    the updated graph. Next, the selected SSL model predicts labels and scores.
    Accuracy is computed before the newly queried labels are added. Finally, the
    query rule selects new unlabeled nodes, those nodes are labeled, and the run
    histories are updated.

    The per-round order is:

        1. Find new cross-class edges among the currently labeled nodes.
        2. Apply the selected edge alteration strategy.
        3. Recompute the signed Laplacian.
        4. Recompute and store the spectral embedding.
        5. Predict labels and scores using the selected SSL model.
        6. Store predictions.
        7. Compute and store accuracy before adding new queried labels.
        8. Select new unlabeled nodes using the query rule.
        9. Add queried labels to the training set.
        10. Store the updated labeled-index history.
        11. Update eigenvalue and eigenvector histories.


    The labels are assumed to be binary and encoded as {-1, +1}. Unlabeled entries
    in the training vector are represented by `empty_val`, usually 0.

    Parameters
    ----------
    n_runs : int
        Number of independent experiment runs to perform.

    n_rounds : int
        Number of active querying rounds per run.

    number_neighbors : int
        Number of neighbors used to construct the kNN graph.

    num_starting_labels_per_class : int
        Number of initial labeled points to select from each class or cluster.

    num_queries_per_round : int
        Number of new points queried and labeled at each round.

    kernel : str
        Kernel name passed to `KNNGraph` when constructing the graph.

    alteration_strategy : {"zero", "negate", "baseline"}
        Edge alteration rule applied to labeled-labeled cross-class edges.

        - "zero":
            Delete the discovered cross-class edge by setting its weight to zero.

        - "negate":
            Negate the discovered cross-class edge weight, turning a positive
            cross-class edge into a negative signed edge.

        - "baseline":
            Do not alter edges. This is used as the control condition.

    query_method : {"frustration", "uncertainty"}
        Rule used to choose which unlabeled nodes to query.

        - "uncertainty":
            Select unlabeled nodes whose prediction scores are closest to zero.

        - "frustration":
            Select unlabeled nodes incident to the largest number of frustrated
            positive edges under the current predicted sign vector.

    model_type : {"laplace", "poisson"}
        Semi-supervised learning model used to compute prediction scores.

        - "laplace":
            Uses signed Laplace learning.

        - "poisson":
            Uses signed Poisson learning.

    specified_starting_seeds : list[list[int]], optional
        Explicit initial labeled indices for each run. If provided, the outer list
        must have length `n_runs`. Each inner list gives the starting labeled
        indices for that run.

        If the dataset has `cluster_id`, each seed list must contain one entry per
        cluster. Otherwise, each seed list must contain one entry per class.

    graph_construction_method : {"legacy", "partner"}, default="partner"
        Graph construction method passed to `KNNGraph`.

    starting_idx : int, default=1
        Starting eigenvector index used when constructing the embedding.

    ending_idx : int, default=3
        Ending eigenvector index used when constructing the embedding.

    empty_val : int, default=0
        Value used in `y_train` to mark unlabeled nodes.

    rng : numpy.random.Generator, optional
        Random number generator used for initial label selection. If not provided,
        a new default generator is created.

    Attributes
    ----------
    n_runs : int
        Number of runs.

    n_rounds : int
        Number of querying rounds.

    number_neighbors : int
        Number of neighbors in the kNN graph.

    num_starting_labels_per_class : int
        Number of initially labeled points per class or cluster.

    num_queries_per_round : int
        Number of queried labels per round.

    kernel : str
        Kernel used for graph construction.

    alteration_strategy : str
        Selected edge alteration strategy.

    query_method : str
        Selected query method.

    model_type : str
        Selected graph SSL model.

    specified_starting_seeds : list[list[int]] or None
        Optional fixed starting seeds.

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

    Returns
    -------
    None
        The constructor initializes the experiment configuration. Use `run(dataset)`
        to execute the experiment.
    """
    def __init__(
        self,
        n_runs: int,
        n_rounds: int,
        number_neighbors: int,
        num_starting_labels_per_class: int,
        num_queries_per_round: int,
        kernel: str,
        alteration_strategy: str, # "zero" or "negate", "baseline"
        query_method: str, # "frustration" or "Uncertainty"
        model_type: str, # "laplace" or "poisson"
        specified_starting_seeds: list[list[int]] | None = None, # List of lists of indices for each run, or None to use random selection
        graph_construction_method: str = "partner", # "legacy" or "partner"
        starting_idx = 1,
        ending_idx = 3,
        empty_val: int = 0,
        rng: np.random.Generator | None = None,
    ) -> None:
        self.n_runs = n_runs
        self.n_rounds = n_rounds
        self.number_neighbors = number_neighbors
        self.num_starting_labels_per_class = num_starting_labels_per_class
        self.num_queries_per_round = num_queries_per_round
        self.kernel = kernel
        if alteration_strategy not in {"zero", "negate", "baseline"}:
            raise ValueError("alteration_strategy must be 'zero', 'negate' or 'baseline'.")
        else:
            self.alteration_strategy = alteration_strategy
        if query_method not in {"frustration", "uncertainty"}:
            raise ValueError("query_method must be 'frustration' or 'uncertainty'.")
        else:
            self.query_method = query_method
        self.model_type = model_type
        self.specified_starting_seeds = specified_starting_seeds
        self.graph_construction_method = graph_construction_method
        self.starting_idx = starting_idx
        self.ending_idx = ending_idx
        self.empty_val = empty_val
        self.rng = rng if rng is not None else np.random.default_rng()


    def run(self, dataset) -> ExperimentResult:
        """
        Execute the active querying experiment on a dataset.

        This method initializes all runs, repeatedly applies one active querying round
        to each run, and returns an ExperimentResult containing the full collection of
        RunState objects.

        If explicit starting seeds were provided, this method validates that the number
        of seed lists matches `n_runs` and that each seed list has the expected length.
        For Gaussian-style datasets with `cluster_id`, the expected length is the
        number of clusters. Otherwise, the expected length is the number of unique
        class labels.

        Parameters
        ----------
        dataset : object
            Dataset object containing at least:

            - data : array-like
                Feature matrix used to construct the graph.

            - labels : array-like
                Binary labels in {-1, +1}.

            Optionally, the dataset may contain:

            - cluster_id : array-like
                Cluster identifiers used for cluster-balanced initial label selection.

        Returns
        -------
        ExperimentResult
            Object containing the RunState for each completed run.

        Raises
        ------
        ValueError
            If `specified_starting_seeds` is provided but has invalid length or invalid
            per-run seed sizes.
        """
        if self.specified_starting_seeds is not None:
            if len(self.specified_starting_seeds) != self.n_runs:
                raise ValueError("Length of specified_starting_seeds must match n_runs.")

            if hasattr(dataset, "cluster_id") and dataset.cluster_id is not None:
                expected = len(np.unique(dataset.cluster_id))
            else:
                expected = len(np.unique(dataset.labels))

            if any(len(seeds) != expected for seeds in self.specified_starting_seeds):
                raise ValueError(f"Each seed list must have {expected} indices.")

        runs = [self.initialize_run(dataset, run_id) for run_id in range(self.n_runs)]
        for _ in range(self.n_rounds):
            for run_state in runs:
                self._run_round(run_state, dataset)
        return ExperimentResult(runs=runs)


    def initialize_run(self, dataset, run_id: int) -> RunState:
        """
        Initialize a single experiment run.

        This method constructs the initial kNN graph, signed Laplacian, spectral
        embedding, eigenvalue/eigenvector metrics, initial labeled set, training label
        vector, first prediction, and first accuracy value.

        The initial labeled set is either taken from `specified_starting_seeds` or
        sampled using `_select_initial_indices`.

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
            Fully initialized run state containing the graph, adjacency, Laplacian,
            embedding, training labels, labeled indices, and initial metric histories.
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
        """
        Predict labels and return continuous scores for the current graph state.

        This method calls `LabelPropagation.labels_propagation` using the selected
        model type. It then converts the returned continuous scores into binary
        predictions using `LabelPropagation.classifier_with_vec`.

        Because eigenvector-based and graph-based binary classifiers can have a global
        sign ambiguity, this method compares the accuracy of the prediction and its
        global sign flip against the true labels. If the flipped prediction has higher
        accuracy, both the predictions and scores are multiplied by -1.

        Parameters
        ----------
        adjacency : scipy.sparse.spmatrix
            Current graph adjacency matrix.

        y_train : numpy.ndarray
            Training label vector. Labeled nodes contain their true labels and
            unlabeled nodes contain `empty_val`.

        true_labels : numpy.ndarray
            Ground-truth labels for all nodes.

        Returns
        -------
        tuple[numpy.ndarray, numpy.ndarray]
            prediction :
                Binary predicted labels in {-1, +1}.

            scores :
                Continuous score vector returned by the selected graph SSL model,
                possibly sign-flipped to match the true-label orientation.
        """
        scores = LabelPropagation.labels_propagation(adjacency, y_train, self.empty_val, model_type=self.model_type)
        prediction = LabelPropagation.classifier_with_vec(scores)

        acc_norm = np.mean(prediction == true_labels)
        acc_flip = np.mean(-prediction == true_labels)

        if acc_flip > acc_norm:
            prediction = -prediction
            scores = -scores

        return prediction, scores

    def _compute_accuracy(self, prediction, true_labels, y_train) -> float:
        """
        Compute classification accuracy for the current predictions.

        Accuracy is computed using `LabelPropagation.classifier_accuracy_vec` with
        method="unlabeled_only", so labeled training nodes are excluded from the
        reported accuracy.

        Parameters
        ----------
        prediction : numpy.ndarray
            Binary predicted labels.

        true_labels : numpy.ndarray
            Ground-truth labels.

        y_train : numpy.ndarray
            Training label vector containing true labels on labeled nodes and
            `empty_val` on unlabeled nodes.

        Returns
        -------
        float
            Accuracy over currently unlabeled nodes.
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
        Update spectral metric histories for a run.

        This method computes the first four tracked eigenvalues and eigenvectors from
        the current adjacency matrix and appends them to the corresponding histories
        stored in the RunState.

        Parameters
        ----------
        run_state : RunState
            Run state whose metric histories will be updated.

        adjacency : scipy.sparse.spmatrix
            Current adjacency matrix used for the eigenvalue computation.

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



    def _select_initial_indices(self, dataset, k: int = 1) -> list[int]:
        """
        Select initial labeled indices for a dataset.

        If the dataset contains a non-None `cluster_id` attribute, this method samples
        initial labels from each cluster. Otherwise, it samples from each class.

        Parameters
        ----------
        dataset : object
            Dataset containing `labels` and optionally `cluster_id`.

        k : int, default=1
            Number of initial points selected from each class or cluster.

        Returns
        -------
        list[int]
            Initial labeled node indices.
        """
        if hasattr(dataset, "cluster_id") and dataset.cluster_id is not None:
            return self._select_gaussian_initial_indices(dataset, k)

        return self._select_generic_initial_indices(dataset, k)
    
    def _select_gaussian_initial_indices(self, dataset, k) -> list[int]:
        """
        Select initial labeled points from each cluster in a Gaussian-style dataset.

        This method assumes the dataset has a `cluster_id` attribute assigning each
        point to a cluster. It samples `k` points without replacement from every unique
        cluster.

        Parameters
        ----------
        dataset : object
            Dataset containing a `cluster_id` attribute.

        k : int
            Number of points to sample from each cluster.

        Returns
        -------
        list[int]
            Selected initial labeled indices.

        Raises
        ------
        ValueError
            If fewer than two clusters are present or if any cluster contains fewer
            than `k` points.
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
        Select initial labeled points from each class in a generic binary dataset.

        This method assumes the dataset has exactly two unique class labels. It samples
        `k` points without replacement from each class.

        Parameters
        ----------
        dataset : object
            Dataset containing a `labels` attribute.

        k : int
            Number of points to sample from each class.

        Returns
        -------
        list[int]
            Selected initial labeled indices.

        Raises
        ------
        ValueError
            If the dataset does not contain exactly two classes or if either class has
            fewer than `k` points.
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
        """
        Run one active querying round for a single RunState.

        This method updates the run state in place. It applies edge alterations using
        the labels available at the start of the round, recomputes graph-dependent
        objects, evaluates the current model, then queries and adds new labels for use
        in future rounds.

        Accuracy is computed before the newly queried nodes are added to the training
        set, so each recorded accuracy corresponds to the model state before that
        round's new labels are incorporated.

        Parameters
        ----------
        run_state : RunState
            Current state of one experiment run.

        dataset : object
            Dataset containing the ground-truth labels.

        Returns
        -------
        None
            Updates the RunState in place.
        """
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
        uncertain_nodes = self._get_uncertain_nodes(scores, run_state)
        self._label_uncertain_nodes(run_state, uncertain_nodes, labels)
        run_state.labeled_indices_history.append(run_state.labeled_indices.copy())
        self._update_metrics(run_state, run_state.adjacency)


    def check_labeled_nodes( self, labeled_indices: list[int], 
                            labels: np.ndarray, altered_edges: set[tuple[int, int]]) -> list[tuple[int, int]]:
        """
        Find newly discovered labeled-labeled cross-class edges.

        This method checks all pairs of currently labeled nodes. If two labeled nodes
        have opposite true labels and the edge has not already been altered, the edge
        is added to the returned list and recorded in `altered_edges`.

        Edges are stored in sorted order `(min(i, j), max(i, j))` so that each
        undirected edge has a unique representation.

        Parameters
        ----------
        labeled_indices : list[int]
            Current labeled node indices.

        labels : numpy.ndarray
            Ground-truth labels for all nodes.

        altered_edges : set[tuple[int, int]]
            Set of edges that have already been altered. This set is updated in place.

        Returns
        -------
        list[tuple[int, int]]
            Newly discovered cross-class labeled-labeled edges.
        """
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
        """
        Apply the selected edge alteration strategy to a list of edges.

        The input adjacency matrix is copied before modification. Each edge is altered
        according to `self.alteration_strategy`.

        Parameters
        ----------
        adjacency : scipy.sparse.spmatrix
            Current adjacency matrix.

        edges : list[tuple[int, int]]
            Edges to alter.

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
            elif self.alteration_strategy == "baseline":
                pass  # No alteration for baseline
            else:
                raise ValueError(f"Unknown alteration strategy: {self.alteration_strategy}")
        return A
    
    def _get_uncertain_nodes(self, scores: np.ndarray, run_state: RunState) -> list[int]:
        """
        Select the next unlabeled nodes to query.

        The query rule is selected by `self.query_method`.

        If `query_method == "uncertainty"`, this method selects nodes whose continuous
        model scores are closest to zero.

        If `query_method == "frustration"`, this method selects nodes with the largest
        frustration score, computed from the current predictions and current adjacency
        matrix.

        Parameters
        ----------
        scores : numpy.ndarray
            Continuous model score vector.

        run_state : RunState
            Current run state, including labeled indices and adjacency matrix.

        Returns
        -------
        list[int]
            Node indices selected for querying.
        """
        if self.query_method == "uncertainty":
            return LabelPropagation.uncertainty_score(scores, run_state.labeled_indices, self.num_queries_per_round)
        else:
            return LabelPropagation.frustration_score(scores, run_state.labeled_indices, self.num_queries_per_round, run_state.adjacency)
    
    
    def _label_uncertain_nodes(self, run_state: RunState, uncertain_nodes: list[int], true_labels: np.ndarray) -> None:
        """
        Add queried nodes to the labeled set.

        For each queried node, this method writes the true label into `y_train` and
        appends the node index to `labeled_indices`.

        Parameters
        ----------
        run_state : RunState
            Current run state. Its training label vector and labeled index list are
            modified in place.

        uncertain_nodes : list[int]
            Node indices selected for labeling.

        true_labels : numpy.ndarray
            Ground-truth labels for all nodes.

        Returns
        -------
        None
            Updates the RunState in place.
        """
        for idx in uncertain_nodes:
            run_state.y_train[idx] = true_labels[idx]
            run_state.labeled_indices.append(idx)



        

