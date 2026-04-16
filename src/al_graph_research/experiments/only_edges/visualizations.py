import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

YLABEL_MAP = {
    "accuracy_history": "Accuracy",
    "lam2_history": "lambda₂",
    "lam1_history": "lambda₁",
    "gap23_history": "lambda₃ − lambda₂",
    "kappa_history": "kappa"
}

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
    plt.ylabel(YLABEL_MAP.get(attr_name, attr_name)) # type: ignore

    if title is not None:
        plt.title(title)

    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_same_metric_comparison(result, attr_names, title=None, show_std=True):
    """
    Compare a metric across multiple experiments without legend.

    Parameters
    ----------
    results : ExperimentResult
    attr_name : list[str]
    title : str, optional
    show_std : bool
    """
    plt.figure(figsize=(8, 5))

    for attr_name in attr_names:  
        mean = result.mean_metric(attr_name)
        std = result.std_metric(attr_name)

        x = np.arange(len(mean))
        plt.plot(x, mean, label=attr_name)

        if show_std:
            plt.fill_between(x, mean - std, mean + std, alpha=0.2)

    plt.xlabel("Step")
    plt.ylabel(YLABEL_MAP.get(attr_names[0], attr_names[0])) # type: ignore


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

    labeled = np.asarray(run_state.labeled_indices, dtype=int)

    def update(frame):
        ax.clear()
        emb = embeddings[frame]

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

    labeled = np.asarray(run_state.labeled_indices, dtype=int)

    def update(frame):
        ax_true.clear()
        ax_pred.clear()

        emb = embeddings[frame]
        pred_labels = np.asarray(predictions[frame])

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