import pandas as pd
from sklearn.model_selection import StratifiedKFold
import numpy as np


class CV:
    def __init__(self, pca_test_data, pca_test_labels):
        self.test_data = pca_test_data
        self.test_labels = pca_test_labels

    def split_data(self):
        skf = StratifiedKFold(n_splits=10)

        for train, test in skf.split(self.test_data, self.test_labels):
            print('train -  {}   |   test -  {}'.format(np.bincount(self.test_labels[train]),
                                                        np.bincount(self.test_labels[test])))

    def run_cv(self):
        self.split_data()


print('Reading data...')
pca_feats = pd.read_csv('csv/pca_feats.csv').to_numpy()
labels = pd.read_csv('csv/labels.csv').to_numpy()

print('\nStarting cross-validation...')
cv = CV(pca_feats, labels)
cv.run_cv()

