from sklearn.cluster import DBSCAN

class DensityBasedClustering:
    def __init__(self, n_min):
        self.n_min = n_min

    def fit_data(self, train_data):
        return DBSCAN(min_samples=self.n_min).fit(train_data)


