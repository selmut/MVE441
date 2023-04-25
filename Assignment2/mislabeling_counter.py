import numpy as np

from Classifiers.knn import KNN
from Classifiers.lda import LDA
from Classifiers.qda import QDA

from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


class MislabelingCounter:

    def __init__(self, data, labels, nRuns):
        self.data = data
        self.labels = labels
        self.nRuns = nRuns

    def count_picture_mislabel_frequency(self):
        # initializes all classifiers
        neighbors = 5
        knn = KNN(neighbors)
        lda = LDA()
        qda = QDA()
        num_picture_misclassified = np.zeros((3, self.data.shape[0]))
        avg_accuracy = np.zeros((1, 3))

        for n in range(self.nRuns):
            print("Run {} out of {}...".format(n, self.nRuns))
            # split data
            train_data, test_data, train_labels, test_labels = train_test_split(self.data, self.labels, test_size=0.1,
                                                                                shuffle=True)

            # train and predict all classifiers
            knn_model = knn.fit_data(train_data, train_labels)
            knn_predictions = np.asarray(knn.predict(test_data, test_labels, knn_model))

            lda_model = lda.fit_data(train_data, train_labels)
            lda_predictions = np.asarray(lda.predict(test_data, test_labels, lda_model))

            qda_model = qda.fit_data(train_data, train_labels)
            qda_predictions = np.asarray(qda.predict(test_data, test_labels, qda_model))

            test_labels_np = np.asarray(test_labels)
            num_false = np.zeros((1, 3))

            avg_accuracy[0, 0] += accuracy_score(knn_predictions, test_labels) / self.nRuns
            avg_accuracy[0, 1] += accuracy_score(lda_predictions, test_labels) / self.nRuns
            avg_accuracy[0, 2] += accuracy_score(qda_predictions, test_labels) / self.nRuns
            # counts number of times predicted wrong and stores index of mislabeled picture
            for i in range(test_data.shape[0]):
                # number of mislabels per classifier
                '''num_false[0,0] += abs(knn_predictions[i] - test_labels_np[i])
                num_false[0,1] += abs(lda_predictions[i] - test_labels_np[i])
                num_false[0,2] += abs(qda_predictions[i] - test_labels_np[i])'''

                # store index for each classifier
                if knn_predictions[i] != test_labels_np[i]:
                    num_picture_misclassified[0, test_labels.index[i]] += 1

                if lda_predictions[i] != test_labels_np[i]:
                    num_picture_misclassified[1, test_labels.index[i]] += 1

                if qda_predictions[i] != test_labels_np[i]:
                    num_picture_misclassified[2, test_labels.index[i]] += 1

        '''print("KNN {} misclassifications of out {} possible\n".format(int(num_false[0,0]), test_data.shape[0]))
        print("LDA {} misclassifications of out {} possible\n".format(int(num_false[0,1]), test_data.shape[0]))
        print("QDA {} misclassifications of out {} possible\n".format(int(num_false[0,2]), test_data.shape[0]))'''

        return num_picture_misclassified, avg_accuracy
