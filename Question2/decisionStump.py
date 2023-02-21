import csv
import numpy as np

testDataPath = "./data/testdata.csv"
trainDataPath = "./data/traindata.csv"

#Mapping from the value to its group
# trainData = {}
# testData = {}
trainData = []

#Parameters, need to rewrite these in accordance with what we see in documentation:
stumpBaseGini, stumpBaseError = 0, 0

GiniError, finalError = 0,0

directionBaseGini, directionBaseError = 0,0

featureThreshold, featureError = 0,0

features = [-1, 1]


def gini(df):
    gini = 0
    res_gini = 0
    for feature in features:
        pass

def classification_error(df):
    pass

if __name__ == "__main__":
    '''
    Given a hardcoded test and train data path, this program will train a decision stump
    using the algorithm outlined in the documentation. It will then output the
    feature index, threshold, the polarization parameter, and the training error.

    It will then test against the testdata and report the testing error
    '''
    
    #Reading in the training data and adding a column for estimated grouping
    with open(testDataPath, 'r') as testFile:
        csvreader = csv.reader(testFile)
        for row in csvreader:
            trainData.append(row + [-1])


    trainData = np.asarray(trainData, float)
    for feature in features:
        for idx, dataPoint in enumerate(trainData):
            #Identifying the feature that the datapoint belongs to
            #CHECK BOUNDS
            trainData[:idx][4] = -1
            trainData[idx+1:][4] = 1
            gini1 = gini()
