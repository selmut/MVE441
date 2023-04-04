import pandas as pd
from sklearn.cluster import KMeans


class kMeans:
    def __init__(self, num_clusters):
        self.pca_data = None
        self.num_clusters = num_clusters

    def read_pca(self, path):
        self.pca_data = pd.read_csv(path).to_numpy()

    def classify(self):
        kmeans_classification = KMeans(n_clusters=self.num_clusters).fit_predict(self.pca_data)
        return kmeans_classification
