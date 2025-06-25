import matplotlib.pyplot as plt
import numpy as np


def plot_stacked_bar(data: np.ndarray,
                     title: str = "Title",
                     ylabel: str = "Count",
                     group_labels: list | tuple | None = None,
                     category_labels: list | tuple | None = None,
                     width: float = 0.35,
                     figsize: tuple[int, int] = (10, 6),
                     rotate: int = 0):
    """
    Create a stacked bar chart with multiple categories stacked on top of each other.
    
    Args:
        data (ndarray): R x C matrix where R is the number of groups and C is the number of categories in each group
        title (str): Title of the plot
        ylabel (str): Label for y-axis
        group_labels (list | tuple | None): Labels of each group
        category_labels (list | tuple | None): Labels for each category
        width (float): Width of the bars
        figsize (tuple[int, int]): Size of the figure
        rotate (int): Rotation angle for x-axis labels
    """
    assert all(len(sublist) == len(data[0]) for sublist in data[1:]), "All rows in data must have the same length."

    plt.figure(figsize=figsize)

    n_categories = len(data[0])
    n_groups = len(data)
    ind = np.arange(n_categories)
    bottom = np.array([0] * n_categories)
    for i in range(n_groups):
        plt.bar(ind, data[i], width, bottom=bottom, label=None if category_labels is None else category_labels[i])
        bottom += data[i]
    
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xticks(ind, group_labels, rotation=rotate, ha="right" if rotate > 0 else "center")
    plt.legend()
    plt.show()


def plot_grouped_bar(data: np.ndarray, 
                     title: str = "Title",
                     ylabel: str = "Count",
                     group_labels: list | tuple | None = None,
                     category_labels: list | tuple | None = None):
    """
    Create a grouped bar chart where each group contains multiple categories side by side.
    
    Args:
        data (ndarray): R x C matrix where R is the number of groups and C is the number of categories in each group
        title (str): Title of the plot
        ylabel (str): Label for y-axis
        group_labels (list | tuple | None): Labels of each group
        category_labels (list | tuple | None): Labels for each category
    """
    assert all(len(sublist) == len(data[0]) for sublist in data[1:]), "All rows in data must have the same length."

    n_categories = len(data[0])
    n_groups = len(data)
    ind = np.arange(n_categories)
    width = 0.75 / n_groups
    for i in range(n_groups):
        xpos = np.arange(n_categories) + width * (i + 0.5 - n_groups / 2)
        plt.bar(xpos, data[i], width, label=None if category_labels is None else category_labels[i])
    
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xticks(ind, group_labels)
    plt.legend()
    plt.show()
