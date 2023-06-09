import math
import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from Classifiers.knn import KNN
from Classifiers.lda import LDA
from Classifiers.qda import QDA
from plots import *

from sklearn.metrics import accuracy_score
from sklearn.preprocessing import scale
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")


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


# converts a picture to 16 blocks of size 256. Ordered by column then row
def convert_to_blocks(vectorized_picture):
    pic_dim = int(math.sqrt(vectorized_picture.shape[0]))
    squares_dim = int(pic_dim/4)
    reshaped = np.reshape(np.asarray(vectorized_picture), (pic_dim,pic_dim))
    blocks = np.zeros((16, squares_dim**2))

    for i in range(4):
        for j in range(4):
            blocks[4*i+j, :] = np.reshape(reshaped[squares_dim*i:squares_dim*(i+1), squares_dim*j:squares_dim*(j+1)], (1,squares_dim**2))

    return blocks


def run_classification(train_data, test_data, train_labels, test_labels):
    neighbors = 10
    nRuns = 10
    knn = KNN(neighbors)
    lda = LDA()
    qda = QDA()
    avg_knn_scores = np.zeros(4)

    # print('Starting KNN...\n')
    scores_matrix = np.zeros(nRuns)

    for n in range(nRuns):
        # print(f'KNN run nr. {n}...')
        knn_model = knn.fit_data(train_data, train_labels)
        knn_predictions = knn.predict(test_data, test_labels, knn_model)
        knn_accuracy = accuracy_score(knn_predictions, test_labels)

        scores_matrix[n] = knn_accuracy

    avg_knn_scores = np.sum(scores_matrix, axis=0) / nRuns

    # print('Starting LDA...\n')
    scores_matrix = np.zeros(nRuns)

    for n in range(nRuns):
        # print(f'LDA run nr. {n}...')
        lda_model = lda.fit_data(train_data, train_labels)
        lda_predictions = lda.predict(test_data, test_labels, lda_model)
        lda_accuracy = accuracy_score(lda_predictions, test_labels)

        scores_matrix[n] = lda_accuracy

    avg_lda_scores = np.sum(scores_matrix, axis=0) / nRuns

    # print('Starting QDA...\n')
    scores_matrix = np.zeros(nRuns)

    for n in range(nRuns):
        # print(f'QDA run nr. {n}...')
        qda_model = qda.fit_data(train_data, train_labels)
        qda_predictions = qda.predict(test_data, test_labels, qda_model)
        qda_scores = accuracy_score(qda_predictions, test_labels)

        scores_matrix[n] = qda_scores

    avg_qda_scores = np.sum(scores_matrix, axis=0) / nRuns

    return avg_lda_scores, avg_qda_scores, avg_knn_scores


data_scaled = pd.DataFrame(scale(data), columns=data.columns)
pca = PCA(n_components=3)
pca_feats = pca.fit_transform(data_scaled)
pca_train_data, pca_test_data, pca_train_labels, pca_test_labels = train_test_split(pd.DataFrame(pca_feats), pd.DataFrame(labels),
                                                                                    test_size=0.2, stratify=labels)


#def find_best_block(all_vectorized_pictures, labels):
#    num_blocks = 16
#    block_accuracy = np.zeros((num_blocks, 3))
#    block_train_labels = labels
#
#    block_test_data = all_vectorized_pictures
#    block_test_labels = labels
#
#    for block in range(num_blocks):
#        print(f'\nCurrent block: {block+1}')
#        block_train_data = np.zeros((all_vectorized_pictures.shape[0], all_vectorized_pictures.shape[1]))
#        for picture in range(all_vectorized_pictures.shape[0]):
#            blocks = convert_to_blocks(all_vectorized_pictures.iloc[picture, :])
#            block_train_data[picture, (block * 16 * 16):((block + 1) * 16 * 16)] = blocks[block, :]
#
#        block_accuracy[block, :] = run_classification(block_train_data, block_test_data, block_train_labels,
#                                                      block_test_labels)
#    return block_accuracy




def find_best_block(all_vectorized_pictures, labels):
    num_blocks = 16
    block_accuracy = np.zeros((num_blocks, 3))
    block_data = np.zeros((all_vectorized_pictures.shape[0], 16*16))
    for block in range(num_blocks):
        print(f'Current block: {block+1}')
        for picture in range(all_vectorized_pictures.shape[0]):
            blocks = convert_to_blocks(all_vectorized_pictures.iloc[picture, :])
            block_data[picture,:] = blocks[block, :]

        block_train_data, block_test_data, block_train_labels, block_test_labels = train_test_split(pd.DataFrame(block_data), pd.DataFrame(labels),
                                                                                                    test_size=0.2, stratify=labels)
        block_accuracy[block, :] = run_classification(block_train_data, block_test_data, block_train_labels,
                                                      block_test_labels)
    return block_accuracy



def get_block_importance_matrix(block_accuracy):
    n_blocks = 16
    n_rows = 4
    n_cols = 4

    knn_acc = block_accuracy[:, 0]
    lda_acc = block_accuracy[:, 1]
    qda_acc = block_accuracy[:, 2]

    knn_out = np.ones((64, 64))
    lda_out = np.ones((64, 64))
    qda_out = np.ones((64, 64))

    current_block = 0
    for row in range(n_rows):
        for col in range(n_cols):
            knn_tmp = np.ones((n_blocks, n_blocks))*knn_acc[current_block]
            knn_out[n_blocks*row:n_blocks*(row+1), n_blocks*col:n_blocks*(col+1)] = knn_tmp

            lda_tmp = np.ones((n_blocks, n_blocks))*lda_acc[current_block]
            lda_out[n_blocks*row:n_blocks*(row+1), n_blocks*col:n_blocks*(col+1)] = lda_tmp

            qda_tmp = np.ones((n_blocks, n_blocks))*qda_acc[current_block]
            qda_out[n_blocks*row:n_blocks*(row+1), n_blocks*col:n_blocks*(col+1)] = qda_tmp

            current_block += 1

    plt.figure()
    plt.imshow(np.rot90(knn_out,-1), cmap='gray')
    plt.savefig('img/knn_best_blocks.png')
    plt.show()
    plt.close()

    plt.figure()
    plt.imshow(np.rot90(lda_out,-1), cmap='gray')
    plt.savefig('img/lda_best_blocks.png')
    plt.show()
    plt.close()

    plt.figure()
    plt.imshow(np.rot90(qda_out,-1), cmap='gray')
    plt.savefig('img/qda_best_blocks.png')
    plt.show()
    plt.close()

    return knn_out, lda_out, qda_out

nRuns = 15
block_acc = np.zeros((16,3))
for i in range(nRuns):
    print(f'\nRun {i+1} out of {nRuns}...')
    block_acc += find_best_block(data, labels)

print(block_acc)
knn_mat, lda_mat, qda_mat = get_block_importance_matrix(block_acc)

'''print(block_acc)
print(np.max(block_acc, axis=0))
print(np.where(block_acc == np.max(block_acc, axis=0)))

max_rows, max_cols = np.where(block_acc == np.max(block_acc, axis=0))'''

