from dataclasses import dataclass
import numpy as np

from al_graph_research.experiments.only_edges.run_state import RunState


@dataclass
class ExperimentResult:
    """
    Container for the collection of RunState objects produced by an
    EdgeAlterationExperiment.

    This class provides utilities for aggregating metric histories across multiple
    edge-alteration runs. Unlike the active querying experiment, all runs in the
    edge-only experiment are expected to have the same number of rounds because the
    edge batches are generated before the experiment begins. As a result, metric
    histories are required to have equal lengths and can be stacked directly
    without NaN padding.

    The aggregation utilities are primarily intended for statistical analysis and
    visualization of experiment behavior across repeated randomized runs.

    Each metric is referenced by the name of the corresponding RunState history
    attribute, such as:

        - "accuracy_history"
        - "lam1_history"
        - "lam2_history"
        - "vec1_history"

    Attributes
    ----------
    runs : list[RunState]
        Collection of completed experiment runs.
    """
    runs: list[RunState]

    def stack_metric(self, attr_name: str) -> np.ndarray:
        """
        Stack a metric history across runs.

        All runs are required to have histories of identical length. The histories are
        stacked into a NumPy array of shape

            (num_runs, history_length).

        Parameters
        ----------
        attr_name : str
            Name of the RunState history attribute to stack.

        Returns
        -------
        numpy.ndarray
            Matrix containing the stacked metric histories.

        Raises
        ------
        ValueError
            If there are no runs stored in the ExperimentResult.

        ValueError
            If the requested metric histories do not all have the same length.
        """
        if len(self.runs) == 0:
            raise ValueError("Cannot stack metric with no runs.")

        values = [getattr(run, attr_name) for run in self.runs]
        lengths = [len(v) for v in values]

        if len(set(lengths)) != 1:
            raise ValueError(f"All runs must have the same length for {attr_name}.")

        return np.array(values, dtype=float)

    def mean_metric(self, attr_name: str) -> np.ndarray:
        """
        Compute the mean history of a metric across runs.

        Parameters
        ----------
        attr_name : str
            Name of the RunState history attribute.

        Returns
        -------
        numpy.ndarray
            Mean metric history across runs.
        """
        return np.mean(self.stack_metric(attr_name), axis=0)

    def std_metric(self, attr_name: str) -> np.ndarray:
        """
        Compute the standard deviation history of a metric across runs.

        Parameters
        ----------
        attr_name : str
            Name of the RunState history attribute.

        Returns
        -------
        numpy.ndarray
            Standard deviation history across runs.
        """
        return np.std(self.stack_metric(attr_name), axis=0)

    def accuracy_matrix(self) -> np.ndarray:
        """
        Return the stacked accuracy history matrix.

        Returns
        -------
        numpy.ndarray
            Accuracy histories stacked across runs.
        """
        return self.stack_metric("accuracy_history")

    def mean_accuracy(self) -> np.ndarray:
        """
        Compute the mean accuracy history across runs.

        Returns
        -------
        numpy.ndarray
            Mean accuracy history across runs.
        """
        return self.mean_metric("accuracy_history")

    def std_accuracy(self) -> np.ndarray:
        """
        Compute the standard deviation accuracy history across runs.

        Returns
        -------
        numpy.ndarray
            Standard deviation accuracy history across runs.
        """
        return self.std_metric("accuracy_history")

