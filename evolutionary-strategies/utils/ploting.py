from os import path, makedirs
from pickle import dump, load
from enum import Enum

import matplotlib.pyplot as plt
import numpy as np


from matplotlib import rc


class AssetType(Enum):
    graphic = 1
    data = 2


def save(type, name, data=None):
    if type == AssetType.graphic:
        filePath = f"Graphics/{name}.png"
    else:  # AssetType.data
        filePath = f"Data/{name}.pickle"
    if not path.exists(path.dirname(filePath)):
        makedirs(path.dirname(filePath))
    if type == AssetType.graphic:
        plt.savefig(filePath)
    else:  # AssetType.data
        dump(data, open(filePath, "wb"))


def get_convergence_point(fitness, max_iter):
    return max_iter - 1
    last_fitness = None
    for i, f in enumerate(fitness):
        if f == last_fitness:
            return i
        last_fitness = f

    return max_iter


def plot(fitness, title=None, name=None):
    worst = np.amax(fitness, axis=0)
    best = np.amin(fitness, axis=0)
    avg = np.mean(fitness, axis=0)
    median = np.median(fitness, axis=0)
    std = np.std(fitness, axis=0)

    max_iter = median.size

    convergence_point = get_convergence_point(median, max_iter)

    x = np.arange(0, max_iter)
    y = avg

    every = 10

    ls = 'dotted'

    fig, ax = plt.subplots()

    ax.errorbar(x[::every], worst[::every], yerr=std[::every],
                label='Worst', marker='o', markersize=8,
                linestyle=':', linewidth=3)
    # ax.errorbar(x[::every], best[::every], yerr=std[::every],
    #             label='Best', marker='o', markersize=8,
    #             linestyle=':', linewidth=3)
    # ax.errorbar(x[::every], avg[::every], yerr=std[::every], label='Mean',
    #             marker='o', markersize=8, linestyle=':', linewidth=3)
    ax.errorbar(x[::every], median[::every], yerr=std[::every],
                label='Median', marker='o', markersize=8,
                linestyle=':', linewidth=3)
    ax.legend(prop={'size': 10})

    fig.set_size_inches(8, 5)

    plt.yticks(fontsize=13)
    plt.xticks(fontsize=13)
    plt.ylabel('Fitness', fontsize=15)
    plt.xlabel('Iterations', fontsize=15)
    plt.axis(
        [-0, convergence_point, -1, np.amax(worst) + np.amax(std) * 1.2]
    )
    plt.text(0.5, 0.95,
             f'Mean Stable Fitness {median[convergence_point]:.2f}' + '$\pm$' + f'{std[convergence_point]:.2f}',
             horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
    if title:
        plt.title(title, fontsize=15)

    if name:
        save(AssetType.graphic, name)

    plt.show()
    plt.close()
