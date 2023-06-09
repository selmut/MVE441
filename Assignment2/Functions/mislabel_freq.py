import numpy as np
from Classifiers.knn import KNN
from Classifiers.lda import LDA
from Classifiers.qda import QDA
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score

def count_picture_mislabel_frequency(data, labels, nRuns):
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