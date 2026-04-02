import matplotlib.pyplot as plt
import numpy as np
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
