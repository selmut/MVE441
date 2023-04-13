from sklearn.neighbors import KNeighborsClassifier
import numpy as np


class KNN:
    def __init__(self, num_neighbors):
        self.num_neighbors = num_neighbors

    def fit_data(self, train_data, train_labels):
        return KNeighborsClassifier(n_neighbors=self.num_neighbors).fit(train_data, np.ravel(train_labels))

    def predict(self, test_data, test_labels, model):
        return model.predict(test_data)
