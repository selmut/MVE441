from sklearn.cluster import DBSCAN

class DensityBasedClustering:
    def __init__(self, max_dist,n_min):
        self.max_dist=max_dist
        self.n_min = n_min

    def fit_data(self, train_data):
        return DBSCAN(eps= self.max_dist,min_samples=self.n_min).fit_predict(train_data)

