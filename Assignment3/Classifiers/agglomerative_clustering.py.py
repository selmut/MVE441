from sklearn.cluster import AgglomerativeClustering

class Agglomerative:
    def __init__(self, num_clusters):
        self.num_clusters = num_clusters

    def fit_data(self, train_data):
        return AgglomerativeClustering(n_clusters=self.num_clusters).fit(train_data)
    
    def fit_predict(self, data):
        return AgglomerativeClustering(n_clusters=self.num_clusters).fit_predict(data)