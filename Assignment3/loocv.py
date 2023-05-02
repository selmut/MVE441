import numpy as np
import pandas as pd
from sklearn.metrics import silhouette_score, normalized_mutual_info_score, fowlkes_mallows_score
from sklearn.model_selection import LeaveOneOut

class LOOCV:
    def __init__(self, df, model):
        self.df = df
        self.model = model

    def run(self):
        loocv = LeaveOneOut()
        splits = loocv.get_n_splits(self.df)

        for i, (train_index, test_index) in enumerate(loocv.split(self.df)):
            train_data = self.df.iloc[train_index].loc[:, 'Gene 1':'Gene 2999']
            train_labels = self.df.iloc[train_index].loc[:, 'label']

            test_data = self.df.iloc[test_index].loc[:, 'Gene 1':'Gene 2999']
            test_labels = self.df.iloc[test_index].loc[:, 'label']

            #print(train_data.head())
            #print(train_labels.head())
            prediction = self.model.fit_predict(train_data)
