import numpy as np
import pandas as pd
from Classifiers.knn import KNN
from Classifiers.lda import LDA
from Classifiers.qda import QDA
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
from cross_valid import CV


def reduce_dim(df, n):
    std_data = StandardScaler().fit_transform(df)
    pca = PCA(n_components=n, svd_solver='full')
    return pca.fit_transform(std_data)


def run_classification_each_pca_dim(data, labels, n_feats):
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
        pca_feats = reduce_dim(data, n)

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

    # plot_scores(avg_scores_gmm, 'GMM')
    # plot_scores(avg_scores_kmeans, 'KMeans')
    #plot_scores(avg_accuracy_knn, 'KNN')
    #plot_scores(avg_accuracy_lda, 'LDA')
    #plot_scores(avg_accuracy_qda, 'QDA')
    return (avg_accuracy_knn, avg_accuracy_lda, avg_accuracy_qda)



def run_classification(train_data, test_data, train_labels, test_labels):
    neighbors = 10
    nRuns = 10
    knn = KNN(neighbors)
    lda = LDA()
    qda = QDA()
    avg_knn_scores = np.zeros(4)

    print('Starting KNN...\n')
    scores_matrix = np.zeros(nRuns)

    for n in range(nRuns):
        #print(f'KNN run nr. {n}...')
        knn_model = knn.fit_data(train_data, train_labels)
        knn_predictions = knn.predict(test_data, test_labels, knn_model)
        knn_accuracy = accuracy_score(knn_predictions, test_labels)

        scores_matrix[n] = knn_accuracy

    avg_knn_scores = np.sum(scores_matrix, axis=0) / nRuns

    print('Starting LDA...\n')
    scores_matrix = np.zeros(nRuns)

    for n in range(nRuns):
        #print(f'LDA run nr. {n}...')
        lda_model = lda.fit_data(train_data, train_labels)
        lda_predictions = lda.predict(test_data, test_labels, lda_model)
        lda_accuracy = accuracy_score(lda_predictions, test_labels)

        scores_matrix[n] = lda_accuracy

    avg_lda_scores = np.sum(scores_matrix, axis=0) / nRuns

    print('Starting QDA...\n')
    scores_matrix = np.zeros(nRuns)

    for n in range(nRuns):
        #print(f'QDA run nr. {n}...')
        qda_model = qda.fit_data(train_data, train_labels)
        qda_predictions = qda.predict(test_data, test_labels, qda_model)
        qda_scores = accuracy_score(qda_predictions, test_labels)

        scores_matrix[n] = qda_scores

    avg_qda_scores = np.sum(scores_matrix, axis=0) / nRuns

    # stores the average score of CV
    print("KNN scores on dataset: ", avg_knn_scores)
    print("\nLDA scores on dataset: ", avg_lda_scores)
    print("\nQDA scores on dataset: ", avg_qda_scores)

    return avg_lda_scores, avg_qda_scores, avg_knn_scores
