import pandas as pd
from sklearn.model_selection import StratifiedKFold
import numpy as np


class CV:
    def __init__(self, test_data, test_labels, folds):
        self.test_data = test_data
        self.test_labels = test_labels
        self.folds = folds

    def split_data(self):
        skf = StratifiedKFold(n_splits=self.folds, shuffle=True)
        data = self.test_data
        labels = self.test_labels

        for train_index, test_index in skf.split(data, labels):
            train_data, test_data = data.iloc[train_index], data.iloc[test_index]
            # train_labels, test_labels = labels[train_index], labels[test_index]

            print(train_data)  #, train_labels)

    def run_cv(self):
        self.split_data()


print('Reading data...')
pca_test_labels = pd.read_csv('csv/test_labels.csv', header=None)
pca_test_data = pd.read_csv('csv/test_data.csv', header=None)

print('\nStarting cross-validation...')
cv = CV(pca_test_data, pca_test_labels, 10)
cv.run_cv()

