import numpy as np
from al_graph_research.models.altered_laplace import AlteredLaplace

class LaplaceLabels:
    @staticmethod
    def labels_propagation(A, y, empty_val):
        """
        Perform Laplace label propagation on a graph.

        Parameters
        A : scipy.sparse matrix
            Adjacency matrix of the graph.
        y : array-like of shape (N,)
            Label vector where labeled points have values (e.g., ±1)
            and unlabeled points contain `empty_val`.
        empty_val : int or float
            Value used to mark unlabeled points.

        Returns
        scores : ndarray
            Continuous label scores returned by the Laplace model.
            The sign of the score determines the predicted class.
        """
        y = np.asarray(y)
        train_indices = np.flatnonzero(y != empty_val)
        train_labels = y[train_indices]

        n_classes = np.unique(train_labels).size
        class_priors = np.full(n_classes, 1 / n_classes)

        model = AlteredLaplace(A, class_priors=class_priors)
        return model._fit(train_indices, train_labels)


    @staticmethod
    def get_uncertain_indicies(scores, num_pts=1):
        """
        Identify the most uncertain points based on Laplace scores.

        Uncertainty is defined as having score values closest to zero.

        Parameters
        scores : array-like of shape (N,)
            Continuous label scores from Laplace label propagation.
        num_pts : int, default=1
            Number of uncertain points to return.

        Returns
        int or list[int]
            Index (or indices) of the most uncertain points.
            Returns a single integer if `num_pts == 1`, otherwise a list.
        """
        scores = np.asarray(scores).ravel()
        uncertain_indices = np.argsort(np.abs(scores))[:num_pts]

        if num_pts == 1:
            return int(uncertain_indices[0])
        return uncertain_indices.tolist()

    @staticmethod
    def laplaceClassifierWithVec(scores):
        """
        Assign a label of 1 if score > 0, otherwise assign -1.
        """
        scores = np.asarray(scores)
        return np.where(scores > 0, 1, -1)

    @staticmethod
    def classifierAccuracy_Laplace_Vec(predicted_labels, true_labels, method="best_flip", unlabeled_mask=None, labels=None, empty_val=None):
        """
        Compute classification accuracy between predicted and true labels.

        Parameters
        ----------
        predicted_labels : array-like of shape (N,)
            Predicted binary labels.
        true_labels : array-like of shape (N,)
            True binary labels.
        method : str, default="best_flip"
            Method used to compute accuracy.

            Options
            -------
            "standard"
                mean(predicted_labels == true_labels)

            "best_flip"
                max(accuracy, flipped accuracy)

            "unlabeled_only"
                Accuracy computed only on unlabeled points.
                Requires either:
                    - unlabeled_mask, or
                    - labels + empty_val

        unlabeled_mask : array-like of bool, optional
            Boolean mask indicating which points are unlabeled.

        labels : array-like, optional
            Original label vector containing labeled and unlabeled entries.

        empty_val : scalar, optional
            Value used to mark unlabeled entries in `labels`.

        Returns
        -------
        float
            Classification accuracy.

        Raises
        ------
        ValueError
            If required inputs are missing or method is invalid.
        """
        predicted_labels = np.asarray(predicted_labels)
        true_labels = np.asarray(true_labels)

        acc = np.mean(predicted_labels == true_labels)

        if method == "standard":
            return float(acc)

        if method == "best_flip":
            acc_flipped = np.mean(-predicted_labels == true_labels)
            return float(max(acc, acc_flipped))

        if method == "unlabeled_only":
            if unlabeled_mask is None:
                if labels is None or empty_val is None:
                    raise ValueError(
                        "For 'unlabeled_only', provide either unlabeled_mask or (labels and empty_val)."
                    )
                labels = np.asarray(labels)
                unlabeled_mask = (labels == empty_val)
            else:
                unlabeled_mask = np.asarray(unlabeled_mask)

            if unlabeled_mask.sum() == 0:
                raise ValueError("No unlabeled points found.")

            pred_u = predicted_labels[unlabeled_mask]
            true_u = true_labels[unlabeled_mask]

            acc_u = np.mean(pred_u == true_u)
            acc_u_flipped = np.mean(-pred_u == true_u)

            return float(max(acc_u, acc_u_flipped))

        raise ValueError("method must be 'standard', 'best_flip', or 'unlabeled_only'.")

    @staticmethod
    def check_ll_lists_unique(lists):
        """
        Returns True if no two accuracy lists are exactly identical.
        This is a sanity check to ensure that different active learning strategies
        are producing different sequences of accuracies.
        """
        seen = set()

        for lst in lists:
            arr = np.asarray(lst)
            key = (arr.shape, arr.dtype.str, arr.tobytes())

            if key in seen:
                return False
            seen.add(key)

        return True
