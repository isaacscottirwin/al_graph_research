from dataclasses import dataclass
import numpy as np
from al_graph_research.experiments.active_querying.run_state import RunState

@dataclass
class ExperimentResult:
    """
    Container for the collection of RunState objects produced by an experiment.

    This class provides utilities for aggregating metric histories across multiple
    experiment runs. Since different runs may terminate at different times or have
    histories of different lengths, metric histories are stacked using NaN padding.
    Aggregation statistics such as the mean, standard deviation, and active-run
    counts are then computed using NaN-aware operations.

    Each metric is referenced by the name of the corresponding RunState history
    attribute, such as:

        - "accuracy_history"
        - "lam1_history"
        - "lam2_history"
        - "vec1_history"

    The aggregation utilities are primarily intended for plotting and statistical
    comparison across repeated experiment runs.

    Attributes
    ----------
    runs : list[RunState]
        Collection of completed experiment runs.
    """
    runs: list[RunState]

    def stack_metric(self, attr_name: str) -> np.ndarray:
        """
        Stack a metric history across runs using NaN padding.

        Runs may have histories of different lengths. This method pads shorter histories
        with NaN values so that all runs can be combined into a single rectangular
        matrix.

        The resulting array has shape

            (num_runs, max_history_length).

        Parameters
        ----------
        attr_name : str
            Name of the RunState history attribute to stack.

        Returns
        -------
        numpy.ndarray
            Matrix containing the stacked histories with NaN padding.

        Raises
        ------
        ValueError
            If there are no runs stored in the ExperimentResult.
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
        """
        Compute the mean history of a metric across runs.

        NaN padding values are ignored when computing the mean.

        Parameters
        ----------
        attr_name : str
            Name of the RunState history attribute.

        Returns
        -------
        numpy.ndarray
            Mean metric history across runs.
        """
        return np.nanmean(self.stack_metric(attr_name), axis=0)

    def std_metric(self, attr_name: str) -> np.ndarray:
        """
        Compute the standard deviation history of a metric across runs.

        NaN padding values are ignored when computing the standard deviation.

        Parameters
        ----------
        attr_name : str
            Name of the RunState history attribute.

        Returns
        -------
        numpy.ndarray
            Standard deviation history across runs.
        """
        return np.nanstd(self.stack_metric(attr_name), axis=0)

    def count_metric(self, attr_name: str) -> np.ndarray:
        """
        Count how many runs contribute to each time step of a metric history.

        Because histories may have different lengths, later time steps may contain
        fewer contributing runs. This method counts the number of non-NaN entries at
        each time step.

        Parameters
        ----------
        attr_name : str
            Name of the RunState history attribute.

        Returns
        -------
        numpy.ndarray
            Number of contributing runs at each time step.
        """
        stacked = self.stack_metric(attr_name)
        return np.sum(~np.isnan(stacked), axis=0)

    def accuracy_matrix(self) -> np.ndarray:
        """
        Return the stacked accuracy history matrix.

        Returns
        -------
        numpy.ndarray
            Accuracy histories stacked across runs with NaN padding.
        """
        return self.stack_metric("accuracy_history")

    def mean_accuracy(self) -> np.ndarray:
        """
        Compute the mean accuracy history across runs.

        Returns
        -------
        numpy.ndarray
            Mean accuracy at each time step.
        """
        return self.mean_metric("accuracy_history")

    def std_accuracy(self) -> np.ndarray:
        """
        Compute the standard deviation accuracy history across runs.

        Returns
        -------
        numpy.ndarray
            Standard deviation accuracy at each time step.
        """ 
        return self.std_metric("accuracy_history")

    def count_accuracy(self) -> np.ndarray:
        """
        Count how many runs contribute to each accuracy time step.

        Returns
        -------
        numpy.ndarray
            Number of runs contributing to each accuracy step.
        """
        return self.count_metric("accuracy_history")