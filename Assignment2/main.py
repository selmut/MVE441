import os
import pandas as pd
import warnings
import math
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold, SelectKBest, SequentialFeatureSelector
from matplotlib import pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from matplotlib import colors
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
# convert to values in [0,1]?
data = pd.read_csv(path_cnd_data)

# counts how many times each picture is mislabeled for each classifier
def count_picture_mislabel_frequency(nRuns):
    # initalizes all class classifiers
    neighbors = 5
    knn = KNN(neighbors)
    lda = LDA()
    qda = QDA()
    num_picture_missclassified = np.zeros((3,data.shape[0]))
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
    return(num_picture_missclassified)



#print(count_picture_mislabel_frequency(nRuns = 100))

#knn = KNeighborsClassifier(n_neighbors=5)
#print("starting feature selection...")
#sfs = SequentialFeatureSelector(knn, n_features_to_select=4, direction="backward")
#print("fitting data accordingly...")
#sfs.fit(data.iloc[:, :], labels)
#print(sfs.get_support())
#print(sfs.transform(data.iloc[:, :]).shape)
#print(sfs.transform(data.iloc[:, 0:10]).shape)


# converts a picture to 16 blocks of size 256. Ordered by coloumn then row
def convert_to_blocks(vectorized_picture):
    pic_dim = int(math.sqrt(vectorized_picture.shape[0]))
    squares_dim = int(pic_dim/4)
    reshaped = np.reshape(np.asarray(vectorized_picture), (pic_dim,pic_dim))
    blocks = np.zeros((16,squares_dim**2))

    for i in range(4):
        for j in range(4):
            blocks[4*i+j,:] = np.reshape(reshaped[squares_dim*i:squares_dim*(i+1), squares_dim*j:squares_dim*(j+1)], (1,squares_dim**2))

    return blocks


def flip_picture(vectorized_picture):
    pic_dim = int(math.sqrt(vectorized_picture.shape[0]))
    reshaped = np.reshape(np.asarray(vectorized_picture), (pic_dim,pic_dim))
    flipped_picture = np.copy(reshaped)

    for row in range(pic_dim):
        flipped_picture[row,:] = reshaped[pic_dim-row-1,:]

    vectorized_flipped_picture = np.reshape(flipped_picture, (1, pic_dim**2))
    return vectorized_flipped_picture

# converts a picture to 16 blocks of size 256. Ordered by coloumn then row
#test_block = convert_to_blocks(data.iloc[0,:])

seq = np.zeros((16,1))
for i in range(seq.shape[0]):
    seq[i,0] = i

test_flip = flip_picture(data.iloc[0,:])
print("real first: \n{} \n{} \n {} \n{} \n".format(data.iloc[0,-16:-1],data.iloc[0,-32:-17],data.iloc[0,-48:-33],data.iloc[0,-64:-49]))
print("flipped: \n{} {} \n {} {} \n".format(test_flip[0,0:15],test_flip[0,16:31],test_flip[0,32:47],test_flip[0,48:63]))


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