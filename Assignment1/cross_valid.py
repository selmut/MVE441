import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, f1_score, roc_auc_score, roc_curve
from Classifiers.gmm import GMM
import numpy as np
from collections import Counter


class CV:
    def __init__(self, test_data, test_labels, folds, classifier):
        self.test_data = test_data
        self.test_labels = test_labels
        self.folds = folds
        self.classifier = classifier

    def metrics(self,predicted_labels, true_labels):
        conf_mat = confusion_matrix(true_labels, predicted_labels)
        acc = accuracy_score(true_labels, predicted_labels)
        f1 = f1_score(true_labels, predicted_labels) #good score to consider since it is adjusted to imbalanced data sets, which we have
        precision = precision_score(true_labels, predicted_labels)
        roc_auc = roc_auc_score(true_labels, predicted_labels)

        fpr, tpr, thresholds = roc_curve(true_labels, predicted_labels)
        return conf_mat, [acc, f1, precision, roc_auc], [fpr, tpr, thresholds]

    def run_cv(self):
        skf = StratifiedKFold(n_splits=self.folds, shuffle=True)
        data = self.test_data
        labels = self.test_labels
        skf.get_n_splits(data, labels)

        avg_scores = np.zeros([1,4])

        for i, (train_index, test_index) in enumerate(skf.split(data, labels)):
            current_test_fold = data.iloc[test_index]
            current_test_fold_labels = labels.iloc[test_index]

            current_train_fold_labels = labels.iloc[train_index]
            current_train_fold = data.iloc[train_index]
            
            #values, counts = np.unique(current_test_fold_labels, return_counts=True)
            #print(counts)
            
            model = self.classifier.fit_data(current_train_fold, current_train_fold_labels)
            predicted_labels = self.classifier.predict(current_test_fold, current_test_fold_labels, model)
            conf_mat, scores, rates = self.metrics(predicted_labels,current_test_fold_labels)
            avg_scores += scores
        
        return avg_scores/self.folds

            
"""
print('Reading data...')
pca_test_labels = pd.read_csv('csv/test_labels.csv', header=None)
pca_test_data = pd.read_csv('csv/test_data.csv', header=None)

print('\nStarting cross-validation...')
gmm = GMM(2)
cv = CV(pca_test_data, pca_test_labels, 10, gmm)
test = cv.run_cv()
print(test)
"""