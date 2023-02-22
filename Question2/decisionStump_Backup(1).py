import csv
import numpy as np

testDataPath = "./data/testdata.csv"
trainDataPath = "./data/traindata.csv"

#Mapping from the value to its group
# trainData = {}
# testData = {}
trainData = []

#Parameters, need to rewrite these in accordance with what we see in documentation:
# stumpBaseGini, stumpBaseError = 0, 0

finalGini, finalError = [],[]

directionBaseGini, directionBaseError = [],[]

featureThreshold, featureError = [],[]

features = [-1, 1]


def gini(df):
    gini = 0
    giniOut = 0
    for idx, feature in enumerate(features):
        #creating boolean masks for where things are the same as the feature and the same as they were originally
        sameFeat = np.where(df[:, 4]==feature, True, False)
        consistentFeat = np.where(df[:, 3]==feature, True, False)
        
        #print("SameFeat \n", np.where(np.logical_and(sameFeat, consistentFeat)==True))
        consistentFeat = len(np.where(np.logical_and(sameFeat, consistentFeat)==True)[0])
        sameFeat = len(np.where(sameFeat==True)[0])
        
        if sameFeat == 0:
            continue
        
        #Added this in such that cases where stuff bottoms out at the start doesnt return a 0 Gini
        # if idx == 0:
        #     gini = 0
        #     giniOut = 0
            
        gini = 1- (consistentFeat/sameFeat)**2
        
        giniOut += gini*sameFeat   
    #Comes out as a percentage
    if (giniOut == 0):
        print("HERE")
    return giniOut/100

def calcError(df):
    return len(np.where(df[:, 3] != df[:, 4])[0])/len(df)

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
    for dimension in range(0, trainData.shape[1]-2):
        #updating the estimate per dimension
        finalGini.append(1)
        finalError.append(1)
        
        featureThreshold.append(1)
        featureError.append(1)
        
        directionBaseError.append(1)
        directionBaseGini.append(1)
        #need to sort traindata by dimension here
        trainData = trainData[trainData[:, dimension].argsort()]
        
        if dimension == 2:
            print(trainData)
        for featIdx, feature in enumerate(features):
            for idx, dataPoint in enumerate(trainData):
                #Identifying the feature that the datapoint belongs to
                #This section corresponds to sorting S using the jth coordinate (using dataPoint as center)
                if idx!=0: #passing the 'all left' case
                    trainData[:idx, 4] = features[1-featIdx] #we want this to be the other case
                trainData[idx:, 4] = feature
                #Added this because the bounds on the latter argument are inclusive and I want leq
                trainData[idx, 4] = feature
                
                gini1 = gini(trainData)
                error1 = calcError(trainData)
                
                #used for calculating the stump based on the GINI (which I think is close to what we want)
                if (finalGini[dimension] > gini1):
                    finalGini[dimension] = gini1
                    #stumpBaseGini = feature
                    featureThreshold[dimension] = idx
                    directionBaseGini[dimension] = 1

                #Used for the stump based on the error calculations
                if (finalError[dimension] > error1):
                    finalError[dimension] = error1
                    #stumpBaseError = feature
                    directionBaseError[dimension] = 1
                    featureError[dimension] = idx

                #Inverting the Direction:
                if idx!=0: #passing the 'all left' case
                    trainData[:idx, 4] = feature #we want this to be the other case
                trainData[idx:, 4] = features[1-featIdx]
                #Added this because the bounds on the latter argument are inclusive and I want leq
                trainData[idx, 4] = features[1-featIdx]

                gini1 = gini(trainData)
                error1 = calcError(trainData)
                
                if (finalGini[dimension] > gini1):
                    finalGini[dimension] = gini1
                    #stumpBaseGini = feature
                    featureThreshold[dimension] = idx
                    directionBaseGini[dimension] = -1

                #Used for the stump based on the error calculations
                if (finalError[dimension] > error1):
                    finalError[dimension] = error1
                    #stumpBaseError = feature
                    directionBaseError[dimension] = -1
                    featureError[dimension] = idx

print("Threshold: ", finalGini)
print("Polarization Parameters: ", directionBaseGini)
print("Feature Index: ", featureThreshold)