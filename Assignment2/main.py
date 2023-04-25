import os
import pandas as pd
import warnings
import math
from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import random
from sklearn.feature_selection import VarianceThreshold, SelectKBest, SequentialFeatureSelector
from matplotlib import pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from matplotlib import colors
from cross_valid import CV
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, f1_score, roc_auc_score, roc_curve
import numpy as np
from Classifiers.knn import KNN
from Classifiers.lda import LDA
from Classifiers.qda import QDA
from sklearn.model_selection import train_test_split
import seaborn as sns
warnings.filterwarnings("ignore")

##################################################################
# methods for selecting features                                 #
# https://scikit-learn.org/stable/modules/feature_selection.html #
##################################################################

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

# counts how many times each picture is mislabeled for each classifier
def count_picture_mislabel_frequency(nRuns):
    # initalizes all class classifiers
    neighbors = 5
    knn = KNN(neighbors)
    lda = LDA()
    qda = QDA()
    num_picture_missclassified = np.zeros((3,data.shape[0]))
    avg_accuracy = np.zeros((1,3))
    for nruns in range(nRuns):
        print("Run {} out of {}...".format(nruns,nRuns))
        # split data
        train_data, test_data, train_labels, test_labels = train_test_split(data, labels, test_size=0.1, shuffle=True)

        # train and predict all classifiers
        knn_model = knn.fit_data(train_data, train_labels)
        knn_predictions = np.asarray(knn.predict(test_data,test_labels,knn_model))

        lda_model = lda.fit_data(train_data, train_labels)
        lda_predictions = np.asarray(lda.predict(test_data,test_labels,lda_model))

        qda_model = qda.fit_data(train_data, train_labels)
        qda_predictions = np.asarray(qda.predict(test_data,test_labels,qda_model))

        test_labels_np = np.asarray(test_labels)
        num_false = np.zeros((1,3))

        avg_accuracy[0,0] += accuracy_score(knn_predictions, test_labels)/nRuns
        avg_accuracy[0,1] += accuracy_score(lda_predictions, test_labels)/nRuns
        avg_accuracy[0,2] += accuracy_score(qda_predictions, test_labels)/nRuns
        # counts number of times predicted wrong and stores index of mislabeled picture
        for i in range(test_data.shape[0]):
            # number of mislabels per classifier
            #num_false[0,0] += abs(knn_predictions[i] - test_labels_np[i])
            #num_false[0,1] += abs(lda_predictions[i] - test_labels_np[i])
            #num_false[0,2] += abs(qda_predictions[i] - test_labels_np[i])

            # store index for each classifier
            if knn_predictions[i] != test_labels_np[i]:
                num_picture_missclassified[0,test_labels.index[i]] += 1

            if lda_predictions[i] != test_labels_np[i]:
                num_picture_missclassified[1,test_labels.index[i]] += 1

            if qda_predictions[i] != test_labels_np[i]:
                num_picture_missclassified[2,test_labels.index[i]] += 1

    #print("KNN {} missclassifications of out {} possible\n".format(int(num_false[0,0]), test_data.shape[0]))
    #print("LDA {} missclassifications of out {} possible\n".format(int(num_false[0,1]), test_data.shape[0]))
    #print("QDA {} missclassifications of out {} possible\n".format(int(num_false[0,2]), test_data.shape[0]))
    return(num_picture_missclassified, avg_accuracy)

#mislabeled_freq, acc = count_picture_mislabel_frequency(nRuns = 20)
#print("Avg. accuracy knn: ", acc[0,0])
#print("Avg. accuracy lda: ", acc[0,1])
#print("Avg. accuracy qda: ", acc[0,2])
#print("mislabeled freq: \n", mislabeled_freq)
#print(count_picture_mislabel_frequency(nRuns = 20))

# converts a picture to 16 blocks of size 256. Ordered by column then row
def convert_to_blocks(vectorized_picture):
    pic_dim = int(math.sqrt(vectorized_picture.shape[0]))
    squares_dim = int(pic_dim/4)
    reshaped = np.reshape(np.asarray(vectorized_picture), (pic_dim,pic_dim))
    blocks = np.zeros((16,squares_dim**2))

    for i in range(4):
        for j in range(4):
            blocks[4*i+j,:] = np.reshape(reshaped[squares_dim*i:squares_dim*(i+1), squares_dim*j:squares_dim*(j+1)], (1,squares_dim**2))

    return blocks


# normalizes and centers the data and then performs pca
def reduce_dim(df, n):
    std_data = StandardScaler().fit_transform(df)
    pca = PCA(n_components=n, svd_solver='full')
    return pca.fit_transform(std_data)

def plot_scores(scores, classifier):
    plt.figure()
    plt.plot(scores)
    plt.savefig(os.path.join(os.path.dirname(__file__), 'img/')+classifier+'_accuracy.png')
    plt.close()


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


choose_n_pixels(3)


def run_classification_each_pca_dim(df, n_feats):
    # constructs all classifiers
    n_list = list(range(1, n_feats))
    neighbors = 10
    knn = KNN(neighbors)
    lda = LDA()
    qda = QDA()

    # stores average scores for each classifier for each number of pca features 
    avg_accuracy_knn = np.zeros(n_feats)
    avg_accuracy_lda = np.zeros(n_feats)
    avg_accuracy_qda = np.zeros(n_feats)

    # loops through all possible number of features of pca
    for n in n_list:
        print("feat. {} out of {}...".format(n, n_feats-1))

        # converts the data to categorical data and performs pca 
        pca_feats = reduce_dim(df, n)

        # splits into test and train data
        pca_train_data, pca_test_data, pca_train_labels, pca_test_labels = train_test_split(pd.DataFrame(pca_feats),
                                                                                            pd.DataFrame(labels),
                                                                                            test_size=0.2,
                                                                                            stratify=labels)

        cv_knn = CV(pca_test_data, pca_test_labels, 7, knn)
        cv_lda = CV(pca_test_data, pca_test_labels, 7, lda)
        cv_qda = CV(pca_test_data, pca_test_labels, 7, qda)

        # stores the average score of CV
        avg_accuracy_knn[n] = cv_knn.run_cv()
        avg_accuracy_lda[n] = cv_lda.run_cv()
        avg_accuracy_qda[n] = cv_qda.run_cv()

    # plot_scores(avg_scores_gmm, 'GMM')
    # plot_scores(avg_scores_kmeans, 'KMeans')
    #plot_scores(avg_accuracy_knn, 'KNN')
    #plot_scores(avg_accuracy_lda, 'LDA')
    #plot_scores(avg_accuracy_qda, 'QDA')
    return (avg_accuracy_knn, avg_accuracy_lda, avg_accuracy_qda)


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


def run_classification(train_data, test_data, train_labels, test_labels):
    neighbors = 10
    nRuns = 10
    knn = KNN(neighbors)
    lda = LDA()
    qda = QDA()
    avg_knn_scores = np.zeros(4)

    print('Starting KNN...\n')
    scores_matrix = np.zeros(nRuns)

    for n in range(nRuns):
        #print(f'KNN run nr. {n}...')
        knn_model = knn.fit_data(train_data, train_labels)
        knn_predictions = knn.predict(test_data, test_labels, knn_model)
        knn_accuracy = accuracy_score(knn_predictions, test_labels)

        scores_matrix[n] = knn_accuracy

    avg_knn_scores = np.sum(scores_matrix, axis=0) / nRuns

    print('Starting LDA...\n')
    scores_matrix = np.zeros(nRuns)

    for n in range(nRuns):
        #print(f'LDA run nr. {n}...')
        lda_model = lda.fit_data(train_data, train_labels)
        lda_predictions = lda.predict(test_data, test_labels, lda_model)
        lda_accuracy = accuracy_score(lda_predictions, test_labels)

        scores_matrix[n] = lda_accuracy

    avg_lda_scores = np.sum(scores_matrix, axis=0) / nRuns

    print('Starting QDA...\n')
    scores_matrix = np.zeros(nRuns)

    for n in range(nRuns):
        #print(f'QDA run nr. {n}...')
        qda_model = qda.fit_data(train_data, train_labels)
        qda_predictions = qda.predict(test_data, test_labels, qda_model)
        qda_scores = accuracy_score(qda_predictions, test_labels)

        scores_matrix[n] = qda_scores

    avg_qda_scores = np.sum(scores_matrix, axis=0) / nRuns

    # stores the average score of CV
    print("KNN scores on dataset: ", avg_knn_scores)
    print("\nLDA scores on dataset: ", avg_lda_scores)
    print("\nQDA scores on dataset: ", avg_qda_scores)

    return avg_lda_scores, avg_qda_scores, avg_knn_scores



data_scaled = pd.DataFrame(preprocessing.scale(data),columns = data.columns) 
pca = PCA(n_components=3)
pca_feats = pca.fit_transform(data_scaled)
pca_train_data, pca_test_data, pca_train_labels, pca_test_labels = train_test_split(pd.DataFrame(pca_feats),pd.DataFrame(labels), 
                                                                                    test_size=0.2, stratify=labels)

run_classification(pca_train_data, pca_test_data, pca_train_labels, pca_test_labels)


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


print(find_best_block(data, labels))



# TODO, perform cathund on this 
def classify_with_flipped_pictures(all_vectorized_pictures, labels):
    all_vectorized_pictures_np = all_vectorized_pictures.copy().to_numpy()
    labels_np = labels.to_numpy()
    num_pictures = labels.shape[0]
    flip_index = random.sample(range(num_pictures), int(num_pictures/2))
    for i in flip_index:
        all_vectorized_pictures_np[i,:] = flip_picture(all_vectorized_pictures_np[i,:])

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