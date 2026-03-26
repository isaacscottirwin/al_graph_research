import numpy as np
from numpy.random import default_rng
import graphlearning.datasets as datasets

class GaussianDataset:
    """
    Generate a synthetic Gaussian mixture dataset for binary classification.

    Each cluster is sampled from a multivariate normal distribution with a
    shared covariance matrix. Cluster 0 is assigned label -1, and all other
    clusters are assigned label 1.

    Parameters
    ----------
    n_per_cluster : int
        Number of samples to generate in each cluster.
    cov : array-like of shape (d, d)
        Shared covariance matrix for all clusters.
    mean_arr : array-like of shape (k, d), optional
        Array of cluster means. If provided, `n_blobs` is ignored.
    seed : int, optional
        Random seed used to initialize the NumPy random number generator.
    n_blobs : int, optional
        Number of clusters to generate when `mean_arr` is not provided.
        Defaults to 3.

    Attributes
    ----------
    n_per_cluster : int
        Number of samples per cluster.
    cov : ndarray
        Covariance matrix used for all clusters.
    mean_arr : ndarray
        Array of cluster means.
    rng : numpy.random.Generator
        Random number generator used for sampling.
    data : ndarray or None
        Generated data matrix of shape (n_samples, d).
    labels : ndarray or None
        Binary label vector of shape (n_samples,), with values in {-1, 1}.
    cluster_id : ndarray or None
        Cluster membership indices of shape (n_samples,).
    """

    def __init__(self, n_per_cluster, cov, mean_arr=None, seed=None, n_blobs=None):
        """
        Initialize the Gaussian dataset and generate samples immediately.

        Parameters
        ----------
        n_per_cluster : int
            Number of samples to generate in each cluster.
        cov : array-like of shape (d, d)
            Shared covariance matrix for all clusters.
        mean_arr : array-like of shape (k, d), optional
            Explicit cluster means. If None, default means are generated.
        seed : int, optional
            Random seed for reproducibility.
        n_blobs : int, optional
            Number of clusters to generate if `mean_arr` is None.
        """
        self.n_per_cluster = n_per_cluster
        self.cov = np.asarray(cov)
        self.rng = default_rng(seed)

        if mean_arr is not None:
            self.mean_arr = np.asarray(mean_arr)
        else:
            if n_blobs is None:
                n_blobs = 3
            self.mean_arr = self._default_means(n_blobs)

        self.data = None
        self.labels = None
        self.cluster_id = None

        self._generate()

    @staticmethod
    def euclidean_dist(point1, point2):
        """
        Compute the Euclidean distance between two points.

        Parameters
        ----------
        point1 : array-like
            First point.
        point2 : array-like
            Second point.

        Returns
        -------
        float
            Euclidean distance between `point1` and `point2`.
        """
        point1 = np.asarray(point1)
        point2 = np.asarray(point2)
        return np.linalg.norm(point1 - point2)

    @staticmethod
    def nearest_mean(point, means):
        """
        Return the index of the mean vector closest to a given point.

        Parameters
        ----------
        point : array-like of shape (d,)
            Query point.
        means : array-like of shape (k, d)
            Collection of mean vectors.

        Returns
        -------
        int
            Index of the closest mean in `means`.
        """
        point = np.asarray(point)
        means = np.asarray(means)
        return np.argmin(np.linalg.norm(means - point, axis=1))

    @staticmethod
    def _default_means(n_blobs, radius=4.0):
        """
        Generate default 2D cluster means equally spaced on a circle.

        Parameters
        ----------
        n_blobs : int
            Number of cluster centers to generate.
        radius : float, optional
            Radius of the circle on which the means are placed. Default is 4.0.

        Returns
        -------
        ndarray of shape (n_blobs, 2)
            Array of 2D cluster means.
        """
        angles = np.linspace(0.0, 2.0 * np.pi, n_blobs, endpoint=False)
        return np.column_stack((radius * np.cos(angles), radius * np.sin(angles)))

    def _generate(self):
        """
        Generate the Gaussian mixture dataset.

        This method samples `n_per_cluster` points from each cluster mean using
        the shared covariance matrix `cov`. It stores the resulting data matrix,
        cluster membership indices, and binary labels.

        Labeling convention:
        - cluster 0 -> label -1
        - all other clusters -> label 1

        Returns
        -------
        None
        """
        n_clusters = len(self.mean_arr)

        clusters = [
            self.rng.multivariate_normal(mean, self.cov, size=self.n_per_cluster)
            for mean in self.mean_arr
        ]

        self.data = np.vstack(clusters)
        self.cluster_id = np.repeat(np.arange(n_clusters), self.n_per_cluster)
        self.labels = np.where(self.cluster_id == 0, -1, 1)

class MnistDataset:
    """
    Load MNIST from graphlearning.datasets and keep only two requested classes.

    Parameters
    ----------
    digit_a : int
        First MNIST class to keep.
    digit_b : int
        Second MNIST class to keep.
    metric : str, optional
        Metric modifier passed to graphlearning.datasets.load. Default is "raw".
    label_values : tuple[int, int], optional
        Output labels to assign to (digit_a, digit_b). Default is (-1, 1).

    Attributes
    ----------
    data : ndarray
        Filtered data matrix containing only samples from digit_a and digit_b.
    original_labels : ndarray
        Original MNIST labels restricted to the selected two classes.
    labels : ndarray
        Relabeled binary targets corresponding to label_values.
    class_map : dict
        Mapping from original class labels to binary output labels.
    """

    def __init__(self, digit_a: int, digit_b: int, metric: str = "raw",
                 label_values: tuple[int, int] = (-1, 1)) -> None:
        if digit_a == digit_b:
            raise ValueError("digit_a and digit_b must be different.")
        if len(label_values) != 2:
            raise ValueError("label_values must have length 2.")

        self.digit_a = digit_a
        self.digit_b = digit_b
        self.metric = metric
        self.label_values = label_values

        self.data: np.ndarray | None = None
        self.original_labels: np.ndarray | None = None
        self.labels: np.ndarray | None = None
        self.class_map = {
            digit_a: label_values[0],
            digit_b: label_values[1],
        }

        self._generate()

    @staticmethod
    def _filter_two_classes(data: np.ndarray, labels: np.ndarray,
                            class_a: int, class_b: int) -> tuple[np.ndarray, np.ndarray]:
        """
        Restrict a dataset to exactly two classes.

        Parameters
        ----------
        data : ndarray
            Full data matrix.
        labels : ndarray
            Full label vector.
        class_a : int
            First class to keep.
        class_b : int
            Second class to keep.

        Returns
        -------
        filtered_data : ndarray
            Data rows whose labels are class_a or class_b.
        filtered_labels : ndarray
            Original labels restricted to class_a or class_b.
        """
        labels = np.asarray(labels)
        mask = (labels == class_a) | (labels == class_b)
        return data[mask], labels[mask]

    @staticmethod
    def _relabel_binary(labels: np.ndarray, class_a: int, class_b: int,
                        label_values: tuple[int, int]) -> np.ndarray:
        """
        Map two original class labels to two user-specified binary labels.
        """
        out = np.empty_like(labels, dtype=int)
        out[labels == class_a] = label_values[0]
        out[labels == class_b] = label_values[1]
        return out

    def _generate(self) -> None:
        """
        Load MNIST, restrict to the selected two classes, and relabel them.
        """
        data, labels = datasets.load("mnist", metric=self.metric)
        filtered_data, filtered_labels = self._filter_two_classes(
            data, labels, self.digit_a, self.digit_b
        )

        self.data = filtered_data
        self.original_labels = filtered_labels
        self.labels = self._relabel_binary(
            filtered_labels, self.digit_a, self.digit_b, self.label_values
        )


class FashionMnistDataset:
    """
    Load Fashion-MNIST from graphlearning.datasets and keep only two requested classes.

    Parameters
    ----------
    class_a : int
        First Fashion-MNIST class to keep.
    class_b : int
        Second Fashion-MNIST class to keep.
    metric : str, optional
        Metric modifier passed to graphlearning.datasets.load. Default is "raw".
    label_values : tuple[int, int], optional
        Output labels to assign to (class_a, class_b). Default is (-1, 1).

    Attributes
    ----------
    data : ndarray
        Filtered data matrix containing only samples from class_a and class_b.
    original_labels : ndarray
        Original Fashion-MNIST labels restricted to the selected two classes.
    labels : ndarray
        Relabeled binary targets corresponding to label_values.
    class_map : dict
        Mapping from original class labels to binary output labels.
    """

    def __init__(self, class_a: int, class_b: int, metric: str = "raw",
                 label_values: tuple[int, int] = (-1, 1)) -> None:
        if class_a == class_b:
            raise ValueError("class_a and class_b must be different.")
        if len(label_values) != 2:
            raise ValueError("label_values must have length 2.")

        self.class_a = class_a
        self.class_b = class_b
        self.metric = metric
        self.label_values = label_values

        self.data: np.ndarray | None = None
        self.original_labels: np.ndarray | None = None
        self.labels: np.ndarray | None = None
        self.class_map = {
            class_a: label_values[0],
            class_b: label_values[1],
        }

        self._generate()

    @staticmethod
    def _filter_two_classes(data: np.ndarray, labels: np.ndarray,
                            class_a: int, class_b: int) -> tuple[np.ndarray, np.ndarray]:
        """
        Restrict a dataset to exactly two classes.

        Parameters
        ----------
        data : ndarray
            Full data matrix.
        labels : ndarray
            Full label vector.
        class_a : int
            First class to keep.
        class_b : int
            Second class to keep.

        Returns
        -------
        filtered_data : ndarray
            Data rows whose labels are class_a or class_b.
        filtered_labels : ndarray
            Original labels restricted to class_a or class_b.
        """
        labels = np.asarray(labels)
        mask = (labels == class_a) | (labels == class_b)
        return data[mask], labels[mask]

    @staticmethod
    def _relabel_binary(labels: np.ndarray, class_a: int, class_b: int,
                        label_values: tuple[int, int]) -> np.ndarray:
        """
        Map two original class labels to two user-specified binary labels.
        """
        out = np.empty_like(labels, dtype=int)
        out[labels == class_a] = label_values[0]
        out[labels == class_b] = label_values[1]
        return out

    def _generate(self) -> None:
        """
        Load Fashion-MNIST, restrict to the selected two classes, and relabel them.
        """
        data, labels = datasets.load("fashionmnist", metric=self.metric)
        filtered_data, filtered_labels = self._filter_two_classes(
            data, labels, self.class_a, self.class_b
        )

        self.data = filtered_data
        self.original_labels = filtered_labels
        self.labels = self._relabel_binary(
            filtered_labels, self.class_a, self.class_b, self.label_values
        )