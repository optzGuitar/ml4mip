from matplotlib import cm
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
import matplotlib.gridspec as gridspec
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
from matplotlib import patches
from itertools import combinations
import torch.nn.functional as F
import torch


def compute_difference_map(prediction):
    # expected shape:  (batch, classes, height, width, depth)
    differences = []
    prediction = F.softmax(prediction, dim=1)
    _, max_index = prediction.max(dim=1, keepdims=True)
    prediction2 = prediction.scatter(1, max_index, 0.0)
    gathered = prediction.gather(1, max_index)
    _, max_index2 = prediction2.max(dim=1, keepdims=True)
    gathered -= prediction.gather(1, max_index2)

    for dim0, dim1 in combinations(range(prediction.shape[1] - 1), 2):
        dim0, dim1 = dim0+1, dim1+1
        difference = torch.abs(prediction[:, dim0] - prediction[:, dim1])
        difference = (1-difference)
        differences.append(difference)

    stacked = torch.stack(differences, dim=1)
    stacked = (stacked - stacked.min()) / stacked.max()

    return stacked, 1 - gathered


def plot_difference_hist(differences):
    sns.histplot(differences[differences != 0].flatten())
    plt.savefig("hist.png")


def plot_colorful_uncertainty_map(differences, uncertainty_mask):
    # expected shape:  (batch, classes, height, width, depth), output of compute_difference_map
    fig = plt.figure(figsize=(9, 8))
    fig.suptitle("Pair-Wise Uncertainty Map")

    gs1 = gridspec.GridSpec(11, 11)
    gs1.update(wspace=0, hspace=0)
    cmap_red = LinearSegmentedColormap.from_list(
        "br", [(0, 0, 0), (1, 0, 0)], N=512)
    cmap_green = LinearSegmentedColormap.from_list(
        "bg", [(0, 0, 0), (0, 1, 0)], N=512)
    cmap_blue = LinearSegmentedColormap.from_list(
        "bb", [(0, 0, 0), (0, 0, 1)], N=512)
    axes = []

    modulated_image = torch.cat([differences, uncertainty_mask], dim=1)

    for t in range(11*(11)):
        ax = plt.subplot(gs1[t])
        axes.append(ax)
        ax.imshow(torch.zeros_like(differences[0, :, :, t].T), origin='lower')
        m_ = ax.imshow(modulated_image[0, :, :, t].T,
                       origin='lower', vmin=0, vmax=1, alpha=1)
        rect = patches.Rectangle((38, 20), 2, 2, facecolor='none')
        ax.add_patch(rect)
        ax.axis("off")
        ax.tick_params(
            which='both',
            bottom=False,
            top=False,
            left=False,
            right=False,
            labelleft=False,
            labelright=False,
            labelbottom=False,
            labeltop=False,
        )
        ax.set_aspect('auto')

    fig.tight_layout()
    fig.subplots_adjust(hspace=0, wspace=0)
    fig.colorbar(cm.ScalarMappable(cmap=cmap_red), ax=axes,
                 location="right", shrink=0.6, ticks=[])

    width = 0.03
    prev_ax = fig.get_axes()[-1]
    pos = prev_ax.get_position()
    diff = pos.x1 - pos.x0
    pos.x0 += 0.026
    pos.x1 = pos.x0 + width
    ax = fig.add_axes(pos)
    fig.colorbar(cm.ScalarMappable(cmap=cmap_green), ax=axes,
                 cax=ax, location="right", shrink=0.6, ticks=[])

    prev_ax = fig.get_axes()[-1]
    pos = prev_ax.get_position()
    diff = pos.x1 - pos.x0
    pos.x0 += diff
    pos.x1 += diff
    ax = fig.add_axes(pos)
    fig.colorbar(cm.ScalarMappable(cmap=cmap_blue), ax=axes,
                 cax=ax, location="right", shrink=0.6, ticks=[0, 1])

    plt.savefig("uncertainty_colors.png")


def compute_difference_map_best_two(prediction, y_true, y_mask):
    prediction = F.softmax(prediction, dim=1)

    # Get the indices of the top two predictions directly without sorting
    _, top_indices = torch.topk(prediction, k=2, dim=1)

    # Extract the probabilities corresponding to the top two indices
    top_probs = torch.gather(prediction, 1, top_indices)

    # Compute the difference map
    difference_map = 1 - (top_probs[:, 0] - top_probs[:, 1])

    return difference_map


def show_uncertainty_map(uncertainty_mask):
    # expected output of compute_difference_map
    fig = plt.figure(figsize=(8, 8))
    fig.suptitle("Top-Two uncertainty map")

    gs1 = gridspec.GridSpec(11, 11)
    gs1.update(wspace=0, hspace=0)
    cmap = ListedColormap(
        ['black', 'yellow', 'red', 'green', 'blue', 'purple'])
    axes = []

    for t in range(11*(11)):
        ax = plt.subplot(gs1[t])
        axes.append(ax)

        m_ = ax.matshow(uncertainty_mask[0, :, :, t].T,
                        cmap='gray', origin='lower', vmin=0, vmax=1)
        rect = patches.Rectangle((38, 20), 2, 2, facecolor='none')
        ax.add_patch(rect)
        ax.axis("off")
        ax.tick_params(
            which='both',
            bottom=False,
            top=False,
            left=False,
            right=False,
            labelleft=False,
            labelright=False,
            labelbottom=False,
            labeltop=False,
        )
        ax.set_aspect('auto')

    fig.tight_layout()
    fig.subplots_adjust(hspace=0, wspace=0)
    fig.colorbar(m_, ax=axes, location="right", shrink=0.6, ticks=[0, 1])

    plt.savefig("map.png")
