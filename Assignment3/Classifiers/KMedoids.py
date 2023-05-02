from sklearn_extra.cluster import KMedoids

class KMedoids:

    def __init__(self, num_clusters):
        self.num_clusters = num_clusters

    def fit_data(self, train_data):
        return KMedoids(n_clusters=self.num_clusters).fit(train_data)

    def predict(self, test_data, model):
        return model.predict(test_data)