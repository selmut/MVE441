import os
import math
import random
import pandas as pd
import numpy as np

from plots import *

path = os.path.dirname(__file__)
path_cnd_data = os.path.join(os.path.dirname(__file__), 'data/CATSnDOGS.csv')
path_cnd_label = os.path.join(os.path.dirname(__file__), 'data/Labels.csv')

# 0 is a cat, 1 is a dog
labels = pd.read_csv(path_cnd_label)
labels_np = labels.to_numpy()
# convert to values in [0,1]?
data = pd.read_csv(path_cnd_data)
data_np = data.to_numpy()

df = pd.concat([data,labels], axis=1)

def flip_picture(vectorized_picture):
    pic_dim = int(math.sqrt(vectorized_picture.shape[0]))
    reshaped = np.reshape(np.asarray(vectorized_picture), (pic_dim,pic_dim))
    flipped_picture = np.copy(reshaped)

    for col in range(pic_dim):
        flipped_picture[:,col] = reshaped[:,pic_dim-col-1]

    vectorized_flipped_picture = np.reshape(flipped_picture, (1, pic_dim**2))
    return vectorized_flipped_picture


def classify_with_flipped_pictures(all_vectorized_pictures, labels):
    all_vectorized_pictures_np = all_vectorized_pictures.copy().to_numpy()
    labels_np = labels.to_numpy()
    num_pictures = labels.shape[0]
    flip_index = random.sample(range(num_pictures), int(num_pictures/2))
    for i in flip_index:
        all_vectorized_pictures_np[i,:] = flip_picture(all_vectorized_pictures_np[i,:])

    all_vectorized_pictures_np = pd.DataFrame(all_vectorized_pictures_np)



classify_with_flipped_pictures(data, labels)