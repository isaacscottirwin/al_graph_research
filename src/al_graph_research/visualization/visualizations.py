import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation


def dataset_visualization(data, labels):
    """
    Visualize a two-dimensional dataset before and after labels are applied.

    The first plot shows all data points in gray, treating the dataset as
    unlabeled. The second plot colors the points according to their binary labels,
    using red for -1 and blue for +1.

    Parameters
    ----------
    data : array-like of shape (n_samples, 2)
        Two-dimensional data points to plot.

    labels : array-like of shape (n_samples,)
        Binary labels for the data points. Expected values are -1 and +1.

    Returns
    -------
    None
        Displays the plots directly.
    """
    plt.scatter(np.array(data)[:, 0], np.array(data)[:, 1], c="gray")
    plt.title("Unlabeled Data")
    plt.show()

    # Plot labeled data
    c_map = {-1: 'red', 1: 'blue'}
    color_arr = [c_map[label] for label in labels]
    plt.scatter(np.array(data)[:, 0], np.array(data)[:, 1], c=color_arr)
    plt.title("Labeled Data")
    plt.show()

def plot_embedding_with_labels(emb, labels, title="Embedding with True Labels"):
    """
    Plot a two-dimensional embedding colored by true binary labels.

    This function is useful for visualizing spectral embeddings, Laplacian
    eigenmap coordinates, or any other two-dimensional node embedding.

    Parameters
    ----------
    emb : numpy.ndarray of shape (n_samples, 2)
        Two-dimensional embedding coordinates.

    labels : array-like of shape (n_samples,)
        Binary labels for each embedded point. Expected values are -1 and +1.

    title : str, default="Embedding with True Labels"
        Title displayed above the plot.

    Returns
    -------
    None
        Displays the plot directly.
    """
    c_map = {-1: 'red', 1: 'blue'}
    color_arr = [c_map[label] for label in labels]
    plt.scatter(emb[:, 0], emb[:, 1], c=color_arr)
    plt.title(title)
    plt.xlabel("EV1")
    plt.ylabel("EV2")
    plt.show()

def plot_metric(result, attr_name, label=None, show_std=True):
    """
    Plot the mean history of a metric across experiment runs.

    The metric is read from an ExperimentResult object using its `mean_metric` and
    `std_metric` methods. Optionally, the plot includes a shaded band showing one
    standard deviation above and below the mean.

    Parameters
    ----------
    result : ExperimentResult
        Object storing metric histories across one or more experiment runs.

    attr_name : str
        Name of the metric history attribute to plot, such as
        "accuracy_history" or "lam2_history".

    label : str, optional
        Label used in the plot legend.

    show_std : bool, default=True
        If True, display mean ± standard deviation shading.

    Returns
    -------
    None
        Adds the metric plot to the current matplotlib axes.
    """
    mean = result.mean_metric(attr_name)
    std = result.std_metric(attr_name)

    x = np.arange(len(mean))

    plt.plot(x, mean, label=label)

    if show_std:
        plt.fill_between(x, mean - std, mean + std, alpha=0.2)

def plot_metric_comparison(results, attr_name, labels, title=None, show_std=True):
    """
    Compare the same metric across multiple experiment results.

    Each ExperimentResult is plotted as a separate curve. The metric is read using
    the provided attribute name, and optional standard deviation shading can be
    included for each experiment.

    Parameters
    ----------
    results : list[ExperimentResult]
        Experiment result objects to compare.

    attr_name : str
        Name of the metric history attribute to plot.

    labels : list[str]
        Legend labels corresponding to each experiment result.

    title : str, optional
        Title displayed above the plot.

    show_std : bool, default=True
        If True, display mean ± standard deviation shading.

    Returns
    -------
    None
        Displays the comparison plot directly.
    """
    plt.figure(figsize=(8, 5))

    for result, label in zip(results, labels):
        mean = result.mean_metric(attr_name)
        std = result.std_metric(attr_name)

        x = np.arange(len(mean))
        plt.plot(x, mean, label=label)

        if show_std:
            plt.fill_between(x, mean - std, mean + std, alpha=0.2)

    plt.xlabel("Step")
    plt.ylabel(attr_name)

    if title is not None:
        plt.title(title)

    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_same_metric_comparison(result, attr_names, title=None, show_std=True):
    """
    Compare multiple metric histories from the same experiment result.

    Each attribute name is plotted as a separate curve using the mean metric history
    stored in the ExperimentResult object. Optional standard deviation shading can
    be included for each metric.

    Parameters
    ----------
    result : ExperimentResult
        Experiment result object containing the metric histories.

    attr_names : list[str]
        Names of the metric history attributes to compare.

    title : str, optional
        Title displayed above the plot.

    show_std : bool, default=True
        If True, display mean ± standard deviation shading.

    Returns
    -------
    None
        Displays the comparison plot directly.
    """
    plt.figure(figsize=(8, 5))

    for attr_name in attr_names:
        mean = np.asarray(result.mean_metric(attr_name))
        std = np.asarray(result.std_metric(attr_name))

        x = np.arange(len(mean))
        plt.plot(x, mean, label=attr_name)

        if show_std:
            plt.fill_between(x, mean - std, mean + std, alpha=0.2)

    plt.xlabel("Step")
    plt.ylabel("Metric Value")

    if title is not None:
        plt.title(title)

    plt.legend()
    plt.tight_layout()
    plt.show()


def animate_embedding_history(run_state, true_labels, save_path=None, dims=2):
    """
    Animate an embedding history over time.

    Each frame shows the embedding at one experiment step. Points are colored by
    their true labels, and currently labeled points are highlighted with black
    outline markers. The animation supports both two-dimensional and
    three-dimensional embeddings.

    Parameters
    ----------
    run_state : RunState
        Object containing `embedding_history` and labeled index information.

    true_labels : array-like of shape (n_samples,)
        True labels used to color the embedded points.

    save_path : str, optional
        If provided, path where the animation should be saved.

    dims : {2, 3}, default=2
        Number of embedding dimensions to visualize.

    Returns
    -------
    matplotlib.animation.FuncAnimation
        Animation object showing the embedding evolution over time.

    Raises
    ------
    ValueError
        If `dims` is not 2 or 3, if the embedding history is empty, or if the
        embedding does not contain enough dimensions.
    """
    embeddings = run_state.embedding_history
    labels = np.asarray(true_labels)

    if dims not in {2, 3}:
        raise ValueError("dims must be 2 or 3.")

    if len(embeddings) == 0:
        raise ValueError("embedding_history is empty.")

    if embeddings[0].shape[1] < dims:
        raise ValueError(f"Embedding has only {embeddings[0].shape[1]} columns, cannot plot {dims}D.")

    if dims == 2:
        fig, ax = plt.subplots(figsize=(7, 6))
    else:
        fig = plt.figure(figsize=(8, 7))
        ax = fig.add_subplot(111, projection="3d")

    def update(frame):
        ax.clear()
        emb = embeddings[frame]

        labeled = get_labeled(frame, run_state)

        if dims == 2:
            ax.scatter(
                emb[:, 0],
                emb[:, 1],
                c=labels,
                s=15,
                alpha=0.7,
            )
            ax.scatter(
                emb[labeled, 0],
                emb[labeled, 1],
                s=80,
                facecolors="none",
                edgecolors="black",
                linewidths=1.5,
            )
            ax.set_xlabel("EV1")
            ax.set_ylabel("EV2")
        else:
            ax.scatter(
                emb[:, 0],
                emb[:, 1],
                emb[:, 2],
                c=labels,
                s=15, # type: ignore
                alpha=0.7,
            )
            ax.scatter(
                emb[labeled, 0],
                emb[labeled, 1],
                emb[labeled, 2],
                s=80, # type: ignore
                facecolors="none",
                edgecolors="black",
                linewidths=1.5,
            )
            ax.set_xlabel("EV0")
            ax.set_ylabel("EV1")
            ax.set_zlabel("EV2") # type: ignore

        ax.set_title(f"Embedding at step {frame}")

    anim = FuncAnimation(fig, update, frames=len(embeddings), interval=500) # type: ignore

    if save_path is not None:
        anim.save(save_path)

    plt.close(fig)
    return anim





def animate_embedding_true_vs_pred(run_state, true_labels, save_path=None, dims=2):
    """
    Animate true labels and predicted labels side by side over time.

    Each frame shows the current embedding in two panels. The left panel colors
    points by their true labels, while the right panel colors points by the model's
    predicted labels. Currently labeled points are highlighted with black outline
    markers. The animation supports both two-dimensional and three-dimensional
    embeddings.

    Parameters
    ----------
    run_state : RunState
        Object containing `embedding_history`, `prediction_history`, and labeled
        index information.

    true_labels : array-like of shape (n_samples,)
        True labels for all points.

    save_path : str, optional
        If provided, path where the animation should be saved.

    dims : {2, 3}, default=2
        Number of embedding dimensions to visualize.

    Returns
    -------
    matplotlib.animation.FuncAnimation
        Animation object comparing true and predicted labels over time.

    Raises
    ------
    ValueError
        If `dims` is not 2 or 3, if the embedding and prediction histories have
        different lengths, if the embedding history is empty, or if the embedding
        does not contain enough dimensions.
    """
    embeddings = run_state.embedding_history
    predictions = run_state.prediction_history
    true_labels = np.asarray(true_labels)

    if dims not in {2, 3}:
        raise ValueError("dims must be 2 or 3.")

    if len(embeddings) != len(predictions):
        raise ValueError("embedding_history and prediction_history must have the same length.")

    if len(embeddings) == 0:
        raise ValueError("embedding_history is empty.")

    if embeddings[0].shape[1] < dims:
        raise ValueError(f"Embedding has only {embeddings[0].shape[1]} columns, cannot plot {dims}D.")

    fig = plt.figure(figsize=(12, 6))

    if dims == 2:
        ax_true = fig.add_subplot(1, 2, 1)
        ax_pred = fig.add_subplot(1, 2, 2)
    else:
        ax_true = fig.add_subplot(1, 2, 1, projection="3d")
        ax_pred = fig.add_subplot(1, 2, 2, projection="3d")


    def update(frame):
        ax_true.clear()
        ax_pred.clear()

        emb = embeddings[frame]
        pred_labels = np.asarray(predictions[frame])

        labeled = get_labeled(frame, run_state)

        true_neg = true_labels == -1
        true_pos = true_labels == 1

        pred_neg = pred_labels == -1
        pred_pos = pred_labels == 1

        if dims == 2:
            # Left: true labels
            ax_true.scatter(
                emb[true_neg, 0],
                emb[true_neg, 1],
                color="red",
                s=15,
                alpha=0.7,
                label="True -1",
            )
            ax_true.scatter(
                emb[true_pos, 0],
                emb[true_pos, 1],
                color="blue",
                s=15,
                alpha=0.7,
                label="True +1",
            )
            ax_true.scatter(
                emb[labeled, 0],
                emb[labeled, 1],
                s=80,
                facecolors="none",
                edgecolors="black",
                linewidths=1.5,
                label="Labeled",
            )
            ax_true.set_xlabel("EV1")
            ax_true.set_ylabel("EV2")

            # Right: predicted labels
            ax_pred.scatter(
                emb[pred_neg, 0],
                emb[pred_neg, 1],
                color="lightcoral",
                s=15,
                alpha=0.7,
                label="Pred -1",
            )
            ax_pred.scatter(
                emb[pred_pos, 0],
                emb[pred_pos, 1],
                color="lightblue",
                s=15,
                alpha=0.7,
                label="Pred +1",
            )
            ax_pred.scatter(
                emb[labeled, 0],
                emb[labeled, 1],
                s=80,
                facecolors="none",
                edgecolors="black",
                linewidths=1.5,
                label="Labeled",
            )
            ax_pred.set_xlabel("EV1")
            ax_pred.set_ylabel("EV2")

        else:
            # Left: true labels
            ax_true.scatter(
                emb[true_neg, 0],
                emb[true_neg, 1],
                emb[true_neg, 2],
                color="red",
                s=15, # type: ignore
                alpha=0.7,
                label="True -1",
            )
            ax_true.scatter(
                emb[true_pos, 0],
                emb[true_pos, 1],
                emb[true_pos, 2],
                color="blue",
                s=15, # type: ignore
                alpha=0.7,
                label="True +1",
            )
            ax_true.scatter(
                emb[labeled, 0],
                emb[labeled, 1],
                emb[labeled, 2],
                s=80, # type: ignore
                facecolors="none",
                edgecolors="black",
                linewidths=1.5,
                label="Labeled",
            )
            ax_true.set_xlabel("EV0")
            ax_true.set_ylabel("EV1")
            ax_true.set_zlabel("EV2") # type: ignore

            # Right: predicted labels
            ax_pred.scatter(
                emb[pred_neg, 0],
                emb[pred_neg, 1],
                emb[pred_neg, 2],
                color="lightcoral",
                s=15, # type: ignore
                alpha=0.7,
                label="Pred -1",
            )
            ax_pred.scatter(
                emb[pred_pos, 0],
                emb[pred_pos, 1],
                emb[pred_pos, 2],
                color="lightblue",
                s=15, # type: ignore
                alpha=0.7,
                label="Pred +1",
            )
            ax_pred.scatter(
                emb[labeled, 0],
                emb[labeled, 1],
                emb[labeled, 2],
                s=80, # type: ignore
                facecolors="none",
                edgecolors="black",
                linewidths=1.5,
                label="Labeled",
            )
            ax_pred.set_xlabel("EV0")
            ax_pred.set_ylabel("EV1")
            ax_pred.set_zlabel("EV2") # type: ignore

        ax_true.set_title(f"True labels, step {frame}")
        ax_pred.set_title(f"Predicted labels, step {frame}")

    anim = FuncAnimation(fig, update, frames=len(embeddings), interval=500) # type: ignore

    if save_path is not None:
        anim.save(save_path)

    plt.close(fig)
    return anim

def get_labeled(frame, run_state):
    """
    Return the labeled node indices for a given animation frame.

    If the run state stores a full labeled-index history, this function returns the
    labeled indices from the requested frame. Otherwise, it returns the final/static
    labeled index set stored on the run state.

    Parameters
    ----------
    frame : int
        Animation frame or experiment step.

    run_state : RunState
        Object containing either `labeled_indices_history` or `labeled_indices`.

    Returns
    -------
    numpy.ndarray
        Array of labeled node indices for the requested frame.
    """
    if hasattr(run_state, "labeled_indices_history"):
        return np.asarray(run_state.labeled_indices_history[frame], dtype=int)
    return np.asarray(run_state.labeled_indices, dtype=int)

def animate_eigenvectors_over_time(eigenvector_one_history, eigenvector_two_history, eigenvector_three_history, 
                                   eigenvector_four_history, title="Eigenvectors over Time", save_path=None):
    """
    Animate the first four eigenvector histories over time.

    Each frame plots four eigenvectors as curves over sorted node index. The y-axis
    limits are fixed across the full animation so that changes in eigenvector shape
    can be compared consistently over time.

    Parameters
    ----------
    eigenvector_one_history : sequence of array-like
        History of the first eigenvector.

    eigenvector_two_history : sequence of array-like
        History of the second eigenvector.

    eigenvector_three_history : sequence of array-like
        History of the third eigenvector.

    eigenvector_four_history : sequence of array-like
        History of the fourth eigenvector.

    title : str, default="Eigenvectors over Time"
        Base title for the animation.

    save_path : str, optional
        If provided, path where the animation should be saved.

    Returns
    -------
    matplotlib.animation.FuncAnimation
        Animation object showing eigenvector evolution over time.

    Raises
    ------
    AssertionError
        If the eigenvector histories do not all have the same length.
    """
    assert (
        len(eigenvector_one_history)
        == len(eigenvector_two_history)
        == len(eigenvector_three_history)
        == len(eigenvector_four_history)
    ), "All eigenvector histories must have the same length."

    time = len(eigenvector_one_history)
    n = len(eigenvector_one_history[0])
    x = np.arange(n)

    fig = plt.figure(figsize=(12, 8))
    ax = plt.subplot(1, 1, 1)

    all_vals = np.concatenate([
        np.asarray(eigenvector_one_history).ravel(),
        np.asarray(eigenvector_two_history).ravel(),
        np.asarray(eigenvector_three_history).ravel(),
        np.asarray(eigenvector_four_history).ravel(),
    ])

    y_min = np.nanmin(all_vals)
    y_max = np.nanmax(all_vals)

    def update(frame):
        ax.clear()

        ax.plot(x, eigenvector_one_history[frame], label="Eigenvector 1")
        ax.plot(x, eigenvector_two_history[frame], label="Eigenvector 2")
        ax.plot(x, eigenvector_three_history[frame], label="Eigenvector 3")
        ax.plot(x, eigenvector_four_history[frame], label="Eigenvector 4")

        ax.set_xlabel("Sorted node index")
        ax.set_ylabel("Eigenvector value")
        ax.set_ylim(y_min, y_max)
        ax.set_title(f"{title} (Step {frame})")
        ax.legend()

    anim = FuncAnimation(fig, update, frames=time, interval=500) # type: ignore

    if save_path is not None:
        anim.save(save_path)

    plt.close(fig)
    return anim