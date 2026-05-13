import numpy as np

class BatchSequences:
    """
    Generate randomized edge-batch sequences for repeated graph alteration
    experiments.

    This class partitions a collection of cross-class edges into batches that are
    applied sequentially over multiple experiment rounds. Each run receives its own
    random permutation of the edge list, allowing repeated experiments with
    different edge-order realizations.

    The generated structure has the form

        batch_sequences[run][round] = list of edges altered in that round.

    This is primarily used for experiments where graph edges are progressively
    altered over time, such as edge deletion or edge negation studies on signed
    graphs.

    Parameters
    ----------
    cross_edges : list[tuple[int, int]]
        List of cross-class graph edges available for batching.

    n_runs : int
        Number of independent randomized edge sequences to generate.

    desired_rounds : int or {"max"}
        Controls how edges are partitioned into batches.

        If "max":
            Each round contains exactly one edge, so the number of rounds equals
            the total number of edges.

        If a positive integer:
            The edge list is partitioned into approximately that many rounds using
            evenly sized batches.

    seed : int, optional
        Random seed used to initialize the NumPy random number generator.

    Attributes
    ----------
    cross_edges : list[tuple[int, int]]
        Original list of cross-class edges.

    n_runs : int
        Number of randomized runs.

    desired_rounds : int or str
        Desired number of rounds or "max".

    seed : int or None
        Random seed used for reproducibility.

    rng : numpy.random.Generator
        Random number generator used for edge shuffling.
    """
    def __init__(
        self,
        cross_edges: list[tuple[int, int]],
        n_runs: int,
        desired_rounds: int | str, # "max" or positive integer
        seed: int | None = None,
    ) -> None:
        self.cross_edges = cross_edges
        self.n_runs = n_runs
        self.desired_rounds = desired_rounds
        self.seed = seed
        self.rng = np.random.default_rng(seed) if seed is not None else np.random.default_rng()

    def generate(self) -> tuple[list[list[list[tuple[int, int]]]], int, int, int]:
        """
        Generate randomized edge-batch sequences.

        For each run, the cross-class edge list is randomly shuffled and then divided
        into batches according to `desired_rounds`.

        If `desired_rounds == "max"`, each batch contains exactly one edge.

        If `desired_rounds` is a positive integer, the method computes an approximate
        batch size and partitions the shuffled edge list into consecutive chunks.

        Returns
        -------
        tuple
            batch_sequences : list[list[list[tuple[int, int]]]]
                Nested list of edge batches.

                Structure:

                    batch_sequences[run][round] = list of edges.

            total_edges : int
                Total number of cross-class edges.

            num_rounds : int
                Number of generated rounds per run.

            edges_per_round : int
                Approximate number of edges assigned to each round.

        Raises
        ------
        ValueError
            If `desired_rounds` is neither "max" nor a positive integer.

        Notes
        -----
        The final round may contain fewer edges than earlier rounds if the edge count
        is not perfectly divisible by the batch size.
        """
        total_edges = len(self.cross_edges)
        if total_edges == 0:
            return [], 0, 0, 0
        if isinstance(self.desired_rounds, str):
            if self.desired_rounds == "max": 
                edges_per_round = 1
                num_rounds = total_edges
                batch_sequences: list[list[list[tuple[int, int]]]] = []
                for _ in range(self.n_runs):
                    edges_shuffled = self.cross_edges.copy()
                    self.rng.shuffle(edges_shuffled)
                    batches = [[edge] for edge in edges_shuffled]
                    batch_sequences.append(batches)
            else:
                raise ValueError("Invalid value for desired_rounds. Must be 'max' or a positive integer.")
        elif isinstance(self.desired_rounds, int) and self.desired_rounds > 0:
            edges_per_round = max(1, total_edges // self.desired_rounds)
            batch_sequences: list[list[list[tuple[int, int]]]] = [] # type: ignore
            for _ in range(self.n_runs):
                edges_shuffled = self.cross_edges.copy()
                self.rng.shuffle(edges_shuffled)

                batches = [
                    edges_shuffled[i:i + edges_per_round]
                    for i in range(0, total_edges, edges_per_round)
                ]
                batch_sequences.append(batches)

            num_rounds = len(batch_sequences[0])
        return batch_sequences, total_edges, num_rounds, edges_per_round