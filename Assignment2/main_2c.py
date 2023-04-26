import os
import math
import random
import warnings
import pandas as pd
import numpy as np

from plots import *
from mislabeling_counter import MislabelingCounter

from sklearn.preprocessing import StandardScaler, scale
from sklearn.decomposition import PCA

warnings.filterwarnings('ignore')

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
nRuns = 5000

def choose_n_pixels(n_components, data):
    data_scaled = pd.DataFrame(scale(data), columns=data.columns)

    index = [f'PC-{i+1}' for i in range(n_components)]

    pca = PCA(n_components=n_components)
    pca.fit_transform(data_scaled)
    reduced_dim_df = pd.DataFrame(pca.components_, columns=data_scaled.columns, index=index)

    out_df = pd.DataFrame(0, columns=data_scaled.columns, index=index)

    for i in index:
        current_pca_ax = reduced_dim_df.loc[i].abs().sort_values(ascending=False)
        max_pixel = current_pca_ax.to_numpy()[0]

        plt.figure()
        plt.plot(current_pca_ax.to_numpy())
        plt.plot(np.linspace(0, 4096, num=4096), np.ones(4096)*max_pixel*0.5, 'k--')
        plt.xlabel("Pixels in descending order"), plt.ylabel("Dimension variance"), plt.title("Variance of each dimension for " + i)
        plt.savefig(f'img/{i}_flipped.png')
        plt.close()

        idxs = np.where(current_pca_ax.to_numpy()>=max_pixel*0.5)
        important_pixels = current_pca_ax.iloc[idxs]
        out_df.loc[i][important_pixels.keys()] += 1
    
    out = np.sum(out_df.to_numpy(), axis=0)

    plt.figure()
    plt.imshow(np.reshape(out, (64, 64), order='F'), cmap='gray')
    plt.savefig('img/important_pixels_flipped_picture.png')


def flip_picture(vectorized_picture):
    pic_dim = int(math.sqrt(vectorized_picture.shape[0]))
    reshaped = np.reshape(np.asarray(vectorized_picture), (pic_dim, pic_dim))
    flipped_picture = np.copy(reshaped)

    for col in range(pic_dim):
        flipped_picture[:, col] = reshaped[:, pic_dim-col-1]

    vectorized_flipped_picture = np.reshape(flipped_picture, (1, pic_dim**2))
    return vectorized_flipped_picture


def classify_with_flipped_pictures(all_vectorized_pictures, labels):
    all_vectorized_pictures_np = all_vectorized_pictures.copy().to_numpy()
    labels_np = labels.to_numpy()
    num_pictures = labels.shape[0]
    flip_index = random.sample(range(num_pictures), int(num_pictures/2))
    for i in flip_index:
        all_vectorized_pictures_np[i, :] = flip_picture(all_vectorized_pictures_np[i, :])

    all_vectorized_pictures_np = pd.DataFrame(all_vectorized_pictures_np)
    choose_n_pixels(3, all_vectorized_pictures_np)

    mc = MislabelingCounter(all_vectorized_pictures_np, labels, nRuns)
    return mc.count_picture_mislabel_frequency()


num_picture_misclassified, avg_accuracy = classify_with_flipped_pictures(data, labels)
'''num_picture_misclassified, avg_accuracy = classify_with_flipped_pictures(data, labels)

percent_picture_misclassified_knn = num_picture_misclassified[0]/nRuns
percent_picture_misclassified_lda = num_picture_misclassified[1]/nRuns
percent_picture_misclassified_qda = num_picture_misclassified[2]/nRuns

knn_idxs = np.where(percent_picture_misclassified_knn >= 0.05)[0]
lda_idxs = np.where(percent_picture_misclassified_lda >= 0.05)[0]
qda_idxs = np.where(percent_picture_misclassified_qda >= 0.05)[0]

idxs = np.intersect1d(np.intersect1d(knn_idxs, lda_idxs), np.intersect1d(lda_idxs, qda_idxs))

print(idxs)
print(f'KNN: {num_picture_misclassified[0, idxs]}')
print(f'LDA: {num_picture_misclassified[1, idxs]}')
print(f'QDA: {num_picture_misclassified[2, idxs]}')


for i in idxs:
    vectorized_picture = data.iloc[i, :]
    if type(vectorized_picture).__module__ == np.__name__:
        plt.imshow(vectorized_picture.reshape((64, 64), order='F'), cmap='gray')
    else:
        plt.imshow(vectorized_picture.to_numpy().reshape((64, 64), order='F'), cmap='gray')
    plt.savefig(f'img/2c/{i}.png')'''

# computed from 5000 realisations
knn_idxs = np.array([488, 438, 497, 508, 514, 410, 475, 492, 510, 507, 515, 469, 397, 523, 524, 484, 408, 377, 416, 496, 499])
lda_idxs = np.array([484, 493, 495, 474, 517, 376, 394, 473, 424, 475, 513, 487, 524, 497, 524, 439, 475, 468, 393, 496, 383])
qda_idxs = np.array([260, 277, 270, 265, 316, 256, 253, 268, 316, 286, 282, 278, 277, 292, 272, 257, 258, 257, 287, 274, 274])
idxs_intersect = [8, 12, 20, 30, 37, 39, 41, 47, 71, 85, 88, 89, 98, 102, 103, 117, 120, 143, 169, 177, 178]


