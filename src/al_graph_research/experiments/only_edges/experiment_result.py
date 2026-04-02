from dataclasses import dataclass
import numpy as np

from al_graph_research.experiments.only_edges.run_state import RunState


@dataclass
class ExperimentResult:
    runs: list[RunState]

    def stack_metric(self, attr_name: str) -> np.ndarray:
        if len(self.runs) == 0:
            raise ValueError("Cannot stack metric with no runs.")

        values = [getattr(run, attr_name) for run in self.runs]
        lengths = [len(v) for v in values]

        if len(set(lengths)) != 1:
            raise ValueError(f"All runs must have the same length for {attr_name}.")

        return np.array(values, dtype=float)

    def mean_metric(self, attr_name: str) -> np.ndarray:
        return np.mean(self.stack_metric(attr_name), axis=0)

    def std_metric(self, attr_name: str) -> np.ndarray:
        return np.std(self.stack_metric(attr_name), axis=0)

    def accuracy_matrix(self) -> np.ndarray:
        return self.stack_metric("accuracy_history")

    def mean_accuracy(self) -> np.ndarray:
        return self.mean_metric("accuracy_history")

    def std_accuracy(self) -> np.ndarray:
        return self.std_metric("accuracy_history")

# Example usage:   
# result.stack_metric("lam2_history")
# result.mean_metric("lam2_history")
# result.std_metric("lam2_history")