import numpy as np
import operator

# Accuracy from the testing predictions
def getAccuracy(testSet, predictions):
    correct = 0 
    for x in range(len(testSet)):
        if testSet[x] == predictions[x]:
            correct += 1
    return 1.0 * correct / len(testSet)

# A custom distance function for use with k-NN
def distance(instance1, instance2, k):
    mm1 = instance1[0] 
    cm1 = instance1[1:]
    dcm1 = np.linalg.det(cm1)
    mm2 = instance2[0]
    cm2 = instance2[1:]
    dcm2 = np.linalg.det(cm2)
    icm2 = np.linalg.inv(cm2)
    dmm = mm2 - mm1
    
    # 
    distance = np.trace(np.dot(icm2, cm1))
    # Mahalanobis distance between the two instances
    distance += np.sqrt(np.dot(np.dot((dmm).transpose(), icm2), dmm)) 
    # Difference in Differential entropy between instances
    # (measured indirectly as a property of Covariance matrices)
    distance += np.log(dcm2) - np.log(dcm1)
    # distance -= k
    return distance

# A function which finds k neighbours of the given instance in the training set
def getNeighbors(trainingSet, trainingLabels, instance, k):
    distances = []
    for x in range(len(trainingSet)):
        # Since the distance function is not symmetric, taking the distance in both directions
        dist = distance(trainingSet[x], instance, k) + distance(instance, trainingSet[x], k)
        distances.append((trainingLabels[x], dist))
    # sorting by distance in ascending order
    distances.sort(key=operator.itemgetter(1))
    neighbors = [d[0] for d in distances[:k]]
    return neighbors

# k-NN logic to find the nearest neighbour's class
def nearestClass(neighbors):
    classVote = {}
    for x in range(len(neighbors)):
        response = neighbors[x]
        if response in classVote:
            classVote[response] += 1 
        else:
            classVote[response] = 1
    sorter = sorted(classVote.items(), key=operator.itemgetter(1), reverse=True)
    return sorter[0][0]
