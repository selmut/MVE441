from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import numpy as np


class LDA:
    def __init__(self):
        pass

    def fit_data(self, train_data, train_labels):
        return LinearDiscriminantAnalysis().fit(train_data, np.ravel(train_labels))

    def predict(self, test_data, test_labels, model):
        return model.predict(test_data)
