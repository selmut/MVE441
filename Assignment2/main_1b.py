import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from Classifiers.knn import KNN
from Classifiers.lda import LDA
from Classifiers.qda import QDA
from cross_valid import CV
from plots import *

from sklearn.preprocessing import StandardScaler, scale
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split


path = os.path.dirname(__file__)
path_cnd_data = os.path.join(os.path.dirname(__file__), 'data/CATSnDOGS.csv')
path_cnd_label = os.path.join(os.path.dirname(__file__), 'data/Labels.csv')

# 0 is a cat, 1 is a dog
labels = pd.read_csv(path_cnd_label)
labels_np = labels.to_numpy()
# convert to values in [0,1]?
data = pd.read_csv(path_cnd_data)
data_np = data.to_numpy()

df = pd.concat([data, labels], axis=1)


# normalizes and centers the data and then performs pca
def reduce_dim(df, n):
    std_data = StandardScaler().fit_transform(df)
    pca = PCA(n_components=n, svd_solver='full')
    return pca.fit_transform(std_data)


def run_classification_each_pca_dim(df, n_feats, labels):
    # constructs all classifiers
    n_list = list(range(1, n_feats))
    neighbors = 10
    knn = KNN(neighbors)
    lda = LDA()
    qda = QDA()

    # stores average scores for each classifier for each number of pca features
    avg_accuracy_knn = np.zeros(n_feats)
    avg_accuracy_lda = np.zeros(n_feats)
    avg_accuracy_qda = np.zeros(n_feats)

    # loops through all possible number of features of pca
    for n in n_list:
        print("feat. {} out of {}...".format(n, n_feats-1))

        # converts the data to categorical data and performs pca
        pca_feats = reduce_dim(df, n)

        # splits into test and train data
        pca_train_data, pca_test_data, pca_train_labels, pca_test_labels = train_test_split(pd.DataFrame(pca_feats),
                                                                                            pd.DataFrame(labels),
                                                                                            test_size=0.2,
                                                                                            stratify=labels)

        cv_knn = CV(pca_test_data, pca_test_labels, 7, knn)
        cv_lda = CV(pca_test_data, pca_test_labels, 7, lda)
        cv_qda = CV(pca_test_data, pca_test_labels, 7, qda)

        # stores the average score of CV
        avg_accuracy_knn[n] = cv_knn.run_cv()
        avg_accuracy_lda[n] = cv_lda.run_cv()
        avg_accuracy_qda[n] = cv_qda.run_cv()

    '''plot_scores(avg_scores_gmm, 'GMM')
    plot_scores(avg_scores_kmeans, 'KMeans')
    plot_scores(avg_accuracy_knn, 'KNN')
    plot_scores(avg_accuracy_lda, 'LDA')
    plot_scores(avg_accuracy_qda, 'QDA')'''
    return avg_accuracy_knn, avg_accuracy_lda, avg_accuracy_qda


def choose_n_pixels(n_components, data):
    data_scaled = pd.DataFrame(scale(data), columns=data.columns)

    index = [f'PC-{i+1}' for i in range(n_components)]

    pca = PCA(n_components=n_components)
    pca.fit_transform(data_scaled)
    reduced_dim_df = pd.DataFrame(pca.components_, columns=data_scaled.columns, index=index)

    for i in index:
        current_pca_ax = reduced_dim_df.loc[i].abs().sort_values(ascending=False)
        max_pixel = current_pca_ax.to_numpy()[0]

        plt.figure()
        plt.plot(current_pca_ax.to_numpy())
        plt.plot(np.linspace(0, 4096, num=4096), np.ones(4096)*max_pixel*0.5, 'k--')
        plt.xlabel("Pixels in descending order"), plt.ylabel("Dimension variance"), plt.title("Variance of each dimension for " + i)
        plt.savefig(f'img/{i}.png')
        plt.close()


choose_n_pixels(3, data)

