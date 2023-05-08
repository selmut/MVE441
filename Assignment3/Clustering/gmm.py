from sklearn.mixture import GaussianMixture
import numpy as np
import pandas as pd


class GMM:
    def __init__(self, num_classes):
        self.num_classes = num_classes

    def fit_data(self, train_data):
        return GaussianMixture(n_components=self.num_classes).fit(train_data)

    def predict(self, test_data, model):
        return model.predict(test_data)
