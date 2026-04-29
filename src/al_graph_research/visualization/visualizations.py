import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D


def dataset_visualization(data, labels):
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
    c_map = {-1: 'red', 1: 'blue'}
    color_arr = [c_map[label] for label in labels]
    plt.scatter(emb[:, 0], emb[:, 1], c=color_arr)
    plt.title(title)
    plt.xlabel("EV1")
    plt.ylabel("EV2")
    plt.show()

def plot_metric(result, attr_name, label=None, show_std=True):
    """
    Plot a metric stored in RunState histories.

    Parameters
    ----------
    result : ExperimentResult
    attr_name : str
        Name of the history attribute (e.g. "accuracy_history", "lam2_history")
    label : str, optional
        Label for legend
    show_std : bool
        Whether to show standard deviation band
    """
    mean = result.mean_metric(attr_name)
    std = result.std_metric(attr_name)

    x = np.arange(len(mean))

    plt.plot(x, mean, label=label)

    if show_std:
        plt.fill_between(x, mean - std, mean + std, alpha=0.2)

def plot_metric_comparison(results, attr_name, labels, title=None, show_std=True):
    """
    Compare a metric across multiple experiments.

    Parameters
    ----------
    results : list[ExperimentResult]
    attr_name : str
    labels : list[str]
    title : str, optional
    show_std : bool
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
    Compare multiple metrics from the same ExperimentResult.

    Parameters
    ----------
    result : ExperimentResult
        The experiment result object.
    attr_names : list[str]
        List of metric attribute names to plot.
    title : str, optional
        Title for the plot.
    show_std : bool
        If True, plot mean +/- std shading.
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
    if hasattr(run_state, "labeled_indices_history"):
        return np.asarray(run_state.labeled_indices_history[frame], dtype=int)
    return np.asarray(run_state.labeled_indices, dtype=int)

def animate_eigenvectors_over_time(eigenvector_one_history, eigenvector_two_history, eigenvector_three_history, 
                                   eigenvector_four_history, title="Eigenvectors over Time", save_path=None):
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