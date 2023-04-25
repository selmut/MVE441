import matplotlib.pyplot as plt
import numpy as np
import os
from main_1b import run_classification_each_pca_dim


def plot_scores(scores, classifier):
    plt.figure()
    plt.plot(scores)
    plt.savefig(os.path.join(os.path.dirname(__file__), 'img/')+classifier+'_accuracy.png')
    plt.close()


def plot_picture(vectorized_picture):
    if type(vectorized_picture).__module__ == np.__name__:
        plt.imshow(vectorized_picture.reshape((64,64), order='F'), cmap='gray')
    else:
        plt.imshow(vectorized_picture.to_numpy().reshape((64,64), order='F'), cmap='gray')
    plt.show()
    plt.close()


def feat_selection_plot(df, labels):
    n_feats = 12
    scores_to_plot = np.zeros((3, n_feats))
    runs = 100
    for i in range(runs):
        print("feat. {} out of {}...".format(i,runs))
        multiple_scores = run_classification_each_pca_dim(df, n_feats, labels)
        scores_to_plot[0, :] += multiple_scores[0]/runs
        scores_to_plot[1, :] += multiple_scores[1]/runs
        scores_to_plot[2, :] += multiple_scores[2]/runs

    plot_scores(scores_to_plot[0,:], 'KNN')  # 3 feat.
    plot_scores(scores_to_plot[1,:], 'LDA')  # 3 feat.
    plot_scores(scores_to_plot[2,:], 'QDA')  # 3 feat.
