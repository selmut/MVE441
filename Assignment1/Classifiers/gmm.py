from sklearn.mixture import GaussianMixture
import numpy as np
import pandas as pd


class GMM:
    def __init__(self, num_classes):
        self.pca_data = None
        self.num_classes = num_classes

    def read_pca(self, path):
        self.pca_data = pd.read_csv(path).to_numpy()

    def classify(self):
        gmm_classification = GaussianMixture(n_components=self.num_classes).fit_predict(self.pca_data)
        return gmm_classification
