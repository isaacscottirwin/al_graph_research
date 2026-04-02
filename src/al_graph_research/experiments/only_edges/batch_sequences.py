import numpy as np

class BatchSequences:
    def __init__(
        self,
        cross_edges: list[tuple[int, int]],
        n_runs: int,
        desired_rounds: int,
        seed: int | None = None,
    ) -> None:
        self.cross_edges = cross_edges
        self.n_runs = n_runs
        self.desired_rounds = desired_rounds
        self.seed = seed
        self.rng = np.random.default_rng(seed) if seed is not None else np.random.default_rng()

    def generate(self) -> tuple[list[list[list[tuple[int, int]]]], int, int, int]:
        total_edges = len(self.cross_edges)
        if total_edges == 0:
            return [], 0, 0, 0

        edges_per_round = max(1, total_edges // self.desired_rounds)

        batch_sequences: list[list[list[tuple[int, int]]]] = []

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