from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold
from feature_extraction import loadDataset
from knn import getAccuracy, nearestClass, getNeighbors

data, labels = loadDataset(filename="gtzan.dat")

# Stratified K-Fold cross-validation with 10 splits
skf = StratifiedKFold(n_splits = 10, shuffle = True, random_state = 1)
fold = 0
acc = 0
acc_list = []
# Performing cross-validation
for train_index, test_index in skf.split(data, labels):
    leng = len(test_index)
    predictions = []
    test_labels = []
    fold += 1
    data_train, data_test = data[train_index], data[test_index]
    labels_train, labels_test = labels[train_index], labels[test_index]   
    for x in range(leng):
        preds = nearestClass(getNeighbors(data_train, labels_train, data_test[x], 5))
        predictions.append(preds)
        test_labels.append(labels_test)    
    print(confusion_matrix(labels_test, predictions))
    accuracy1 = getAccuracy(labels_test, predictions)
    acc_list.append(accuracy1)
    print("Accuracy in fold "+ str(fold) + ": " + str(accuracy1 * 100))
print("Average accuracy: " + str(sum(acc_list) * 10))
