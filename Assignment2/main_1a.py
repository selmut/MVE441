import os
import warnings
import numpy as np
import pandas as pd
from plots import *
from mislabeling_counter import MislabelingCounter

warnings.filterwarnings('ignore')

# paths for folder, and for the data to use
path = os.path.dirname(__file__)
path_cnd_data = os.path.join(os.path.dirname(__file__), 'data/CATSnDOGS.csv')
path_cnd_label = os.path.join(os.path.dirname(__file__), 'data/Labels.csv')

# 0 is a cat, 1 is a dog
labels = pd.read_csv(path_cnd_label)
labels_np = labels.to_numpy()
# convert to values in [0,1]?
data = pd.read_csv(path_cnd_data)
data_np = data.to_numpy()

df = pd.concat([data, labels], axis=1)

mc = MislabelingCounter(data, labels, 20)
mislabeled_freq, acc = mc.count_picture_mislabel_frequency()
print("\nAvg. accuracy knn: ", acc[0, 0])
print("Avg. accuracy lda: ", acc[0, 1])
print("Avg. accuracy qda: ", acc[0, 2])
print("\nmislabeled freq: \n", mislabeled_freq)





