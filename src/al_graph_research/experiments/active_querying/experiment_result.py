from dataclasses import dataclass
import numpy as np
from al_graph_research.experiments.active_querying.run_state import RunState

@dataclass
class ExperimentResult:
    runs: list[RunState]

    def stack_metric(self, attr_name: str) -> np.ndarray:
        """
        Stack variable-length run histories by padding shorter runs with NaN.
        Shape: (num_runs, max_history_length)
        """
        if len(self.runs) == 0:
            raise ValueError("Cannot stack metric with no runs.")

        values = [
            np.asarray(getattr(run, attr_name), dtype=float)
            for run in self.runs
        ]

        max_len = max(len(v) for v in values)

        stacked = np.full((len(values), max_len), np.nan)

        for i, v in enumerate(values):
            stacked[i, :len(v)] = v

        return stacked

    def mean_metric(self, attr_name: str) -> np.ndarray:
        return np.nanmean(self.stack_metric(attr_name), axis=0)

    def std_metric(self, attr_name: str) -> np.ndarray:
        return np.nanstd(self.stack_metric(attr_name), axis=0)

    def count_metric(self, attr_name: str) -> np.ndarray:
        """
        Number of runs still contributing at each time step.
        """
        stacked = self.stack_metric(attr_name)
        return np.sum(~np.isnan(stacked), axis=0)

    def accuracy_matrix(self) -> np.ndarray:
        return self.stack_metric("accuracy_history")

    def mean_accuracy(self) -> np.ndarray:
        return self.mean_metric("accuracy_history")

    def std_accuracy(self) -> np.ndarray:
        return self.std_metric("accuracy_history")

    def count_accuracy(self) -> np.ndarray:
        return self.count_metric("accuracy_history")