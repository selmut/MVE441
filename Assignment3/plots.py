import matplotlib.pyplot as plt; plt.rcParams['figure.dpi'] = 200
import numpy as np
import seaborn as sns


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


def plot_scores(score, n_cluster, metric):
    plt.figure()
    plt.plot(n_cluster, score[0,:])
    plt.plot(n_cluster, score[1,:])
    plt.plot(n_cluster, score[2,:])
    plt.legend(['kMeans', 'GMM', 'Agglomerative'], loc='upper right')
    plt.xlabel('Number of clusters'), plt.ylabel(metric), plt.title(metric+' score vs. number of clusters')
    plt.grid(True)
    plt.savefig('img/'+metric+'_vs_clusters.png')
    plt.close()


def plot_scores_feature_selection(score, n_cluster, metric):
    plt.figure()
    plt.plot(n_cluster, score[0,:])
    plt.plot(n_cluster, score[1,:])
    plt.plot(n_cluster, score[2,:])
    plt.legend(['kMeans', 'GMM', 'Agglomerative'], loc='upper right')
    plt.xlabel('Number of clusters'), plt.ylabel(metric), plt.title(metric+' score vs. number of clusters for feature selection')
    plt.grid(True)
    plt.savefig('img/'+metric+'_vs_clusters_feat_selection.png')
    plt.close()
    
def plot_dim_vs_clusters(scores, n_axes, n_clusters, filename):
    plt.figure()
    sns.heatmap(scores, cmap='flare_r', yticklabels=n_axes, xticklabels=n_clusters, vmin=0, vmax=1)
    plt.xlabel('Clusters')
    plt.ylabel('PCA-dim')
    '''plt.xticks([])
    plt.yticks([])'''
    plt.savefig(f'img/heatmaps/'+filename)
    plt.close()


def plot_scores_vs_clusters(scores, n_clusters, filename, score_type):
    plt.figure()
    plt.plot(n_clusters, scores[0, :], c='tab:pink')
    plt.plot(n_clusters, scores[1, :], c='tab:olive')
    plt.plot(n_clusters, scores[2, :], c='tab:cyan')
    plt.legend(['kMeans', 'GMM', 'Agglomerative'])
    plt.xlabel('Number of clusters')
    plt.ylabel(score_type)
    plt.savefig(f'img/scores/'+filename)
    plt.close()


def plot_scores_hist(chosen_dim_scores, classifier, n_clusters, filename):
    plt.figure()
    plt.hist(chosen_dim_scores[:, classifier, n_clusters-1], bins=15)
    plt.savefig('img/histograms/'+filename)
    plt.close()

