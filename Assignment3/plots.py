import matplotlib.pyplot as plt; plt.rcParams['figure.dpi'] = 200
import numpy as np


def plot_pca(pca_data, labels):
    fig, ax = plt.subplots()
    colors = {0: 'tab:pink', 1: 'tab:olive', 2: 'tab:cyan'}

    for label in np.unique(labels):
        idx = np.where(labels == label)
        ax.scatter(pca_data[idx, 0], pca_data[idx, 1], c=colors[label], label=label, s=100)
    ax.legend()
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.savefig('img/pca_2D_kmeans.png')


def plot_pca_3D(pca_data, labels):
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    colors = {0: 'tab:pink', 1: 'tab:olive', 2: 'tab:cyan'}

    for label in np.unique(labels):
        idx = np.where(labels == label)
        ax.scatter(pca_data[idx, 0], pca_data[idx, 1], pca_data[idx, 2], c=colors[label], label=label, s=100)
    fig.legend(loc='upper right')

    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.set_zlabel('PC3')
    plt.savefig('img/pca_3D_kmeans.png')
