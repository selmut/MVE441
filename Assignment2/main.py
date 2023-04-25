import os
import pandas as pd
import warnings
import math
from sklearn import preprocessing
from sklearn.decomposition import PCA
import random
from sklearn.feature_selection import VarianceThreshold, SelectKBest, SequentialFeatureSelector
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, f1_score, roc_auc_score, roc_curve
import numpy as np
from Functions.mislabel_freq import count_picture_mislabel_frequency
from Functions.run_classifications import run_classification_each_pca_dim, run_classification
from sklearn.model_selection import train_test_split
warnings.filterwarnings("ignore")


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

df = pd.concat([data,labels], axis=1)


def plot_scores(scores, classifier):
    plt.figure()
    plt.plot(scores)
    plt.savefig(os.path.join(os.path.dirname(__file__), 'img/')+classifier+'_accuracy.png')
    plt.close()


def convert_to_blocks(vectorized_picture):
    pic_dim = int(math.sqrt(vectorized_picture.shape[0]))
    squares_dim = int(pic_dim/4)
    reshaped = np.reshape(np.asarray(vectorized_picture), (pic_dim,pic_dim))
    blocks = np.zeros((16,squares_dim**2))

    for i in range(4):
        for j in range(4):
            blocks[4*i+j,:] = np.reshape(reshaped[squares_dim*i:squares_dim*(i+1), squares_dim*j:squares_dim*(j+1)], (1,squares_dim**2))

    return blocks


def choose_n_pixels(n_components):
    data_scaled = pd.DataFrame(preprocessing.scale(data), columns=data.columns)

    index = [f'PC-{i+1}' for i in range(n_components)]

    pca = PCA(n_components=n_components)
    pca.fit_transform(data_scaled)
    reduced_dim_df = pd.DataFrame(pca.components_, columns=data_scaled.columns, index=index)

    for i in index:
        current_pca_ax = reduced_dim_df.loc[i].abs().sort_values(ascending=False)
        max_pixel = current_pca_ax.to_numpy()[0]

        plt.figure()
        plt.plot(current_pca_ax.to_numpy())
        plt.plot(np.linspace(0, 4096, num=4096), np.ones(4096)*max_pixel*0.5, 'k--')
        plt.xlabel("Pixels in descending order"), plt.ylabel("Dimension variance"), plt.title("Variance of each dimension for " + i)
        plt.savefig(f'img/{i}.png')
        plt.close()



def flip_picture(vectorized_picture):
    pic_dim = int(math.sqrt(vectorized_picture.shape[0]))
    reshaped = np.reshape(np.asarray(vectorized_picture), (pic_dim,pic_dim))
    flipped_picture = np.copy(reshaped)

    for col in range(pic_dim):
        flipped_picture[:,col] = reshaped[:,pic_dim-col-1]

    vectorized_flipped_picture = np.reshape(flipped_picture, (1, pic_dim**2))
    return vectorized_flipped_picture

def plot_picture(vectorized_picture):
    if type(vectorized_picture).__module__ == np.__name__:
        plt.imshow(vectorized_picture.reshape((64,64), order='F'), cmap='gray')
    else:
        plt.imshow(vectorized_picture.to_numpy().reshape((64,64), order='F'), cmap='gray')
    plt.show()
    plt.close()

# converts a picture to 16 blocks of size 256. Ordered by coloumn then row
#test_block = convert_to_blocks(data.iloc[0,:])
#imgplot = plt.imshow(np.rot90(data.iloc[12,:].to_numpy().reshape(64,64), 3), cmap='gray')

def feat_selection_plot():
    n_feats = 12
    scores_to_plot = np.zeros((3,n_feats))
    runs = 100
    for i in range(runs):
        print("feat. {} out of {}...".format(i,runs))
        multiple_scores = run_classification_each_pca_dim(df,n_feats)
        scores_to_plot[0,:] += multiple_scores[0]/runs
        scores_to_plot[1,:] += multiple_scores[1]/runs
        scores_to_plot[2,:] += multiple_scores[2]/runs

    plot_scores(scores_to_plot[0,:], 'KNN') # 3 feat.
    plot_scores(scores_to_plot[1,:], 'LDA') # 3 feat.
    plot_scores(scores_to_plot[2,:], 'QDA') # 3 feat.


data_scaled = pd.DataFrame(preprocessing.scale(data),columns = data.columns) 
pca = PCA(n_components=3)
pca_feats = pca.fit_transform(data_scaled)
pca_train_data, pca_test_data, pca_train_labels, pca_test_labels = train_test_split(pd.DataFrame(pca_feats),pd.DataFrame(labels), 
                                                                                    test_size=0.2, stratify=labels)

#run_classification(pca_train_data, pca_test_data, pca_train_labels, pca_test_labels)


def find_best_block(all_vectorized_pictures, labels):
    num_blocks = 16
    block_accuracy = np.zeros((num_blocks,3))
    block_train_labels = labels
    
    block_test_data = all_vectorized_pictures
    block_test_labels = labels

    for block in range(num_blocks):
        block_train_data = np.zeros((all_vectorized_pictures.shape[0], all_vectorized_pictures.shape[1]))
        for picture in range(all_vectorized_pictures.shape[0]):
            blocks = convert_to_blocks(all_vectorized_pictures.iloc[picture,:])
            block_train_data[picture,(block*16*16):((block+1)*16*16)] = blocks[block,:]

        block_accuracy[block,:] = run_classification(block_train_data, block_test_data, block_train_labels, block_test_labels)
    return block_accuracy




# TODO, perform cathund on this 
def classify_with_flipped_pictures(all_vectorized_pictures, labels):
    all_vectorized_pictures_np = all_vectorized_pictures.copy().to_numpy()
    labels_np = labels.to_numpy()
    num_pictures = labels.shape[0]
    flip_index = random.sample(range(num_pictures), int(num_pictures/2))
    for i in flip_index:
        all_vectorized_pictures_np[i,:] = flip_picture(all_vectorized_pictures_np[i,:])


choose_n_pixels(3)

# counts how many times each picture is mislabeled for each classifier
#count_picture_mislabel_frequency(data, labels, nRuns = 20)

# converts a picture to 16 blocks of size 256. Ordered by coloumn then row
convert_to_blocks(data.iloc[0,:])

run_classification_each_pca_dim(data, labels, n_feats=12)

print(find_best_block(data, labels))

classify_with_flipped_pictures(data, labels)
'''pd.DataFrame(pca_train_data).to_csv(csv_path+'train_data.csv', header=False, index=False)
pd.DataFrame(pca_train_labels).to_csv(csv_path+'train_labels.csv', header=False, index=False)
pd.DataFrame(pca_test_data).to_csv(csv_path+'test_data.csv', header=False, index=False)
pd.DataFrame(pca_test_labels).to_csv(csv_path+'test_labels.csv', header=False, index=False)

csv_path = os.path.join(os.path.dirname(__file__), 'csv/')

c = ['tab:blue', 'tab:orange']
plt.scatter(pca1, pca2, c=labels, s=1, cmap=colors.ListedColormap(c))
plt.savefig(os.path.join(os.path.dirname(__file__), 'img/')+'pca_plot.png')
plt.close()

pca_feats.tofile(csv_path+'pca_feats.csv', sep=',', format='%10.5f')

labels.to_numpy().tofile(csv_path+'labels.csv', sep=',', format='%10.5f')'''