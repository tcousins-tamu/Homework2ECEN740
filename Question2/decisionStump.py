import csv
import numpy as np

testDataPath = "./data/testdata.csv"
trainDataPath = "./data/traindata.csv"
#trainDataPath = "./data/extradata.csv"

#Mapping from the value to its group
trainDataIdxMap = {}
testDataIdxMap = {}

trainData = []
testData = []

#Parameters, need to rewrite these in accordance with what we see in documentation:
# stumpBaseGini, stumpBaseError = 0, 0

finalError = [] #measure of the error scores by dimension
polarization = [] #Direction of the polarity
threshold = [] #Threshold of the values
trainingError = [] #training Error values

# featureIdxUnsorted = [] #Feature Index (unsorted) (will need to be adjusted)
# featureIdxSorted = [] #Feature Index (sorted)

features = [-1, 1]

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
    with open(trainDataPath, 'r') as trainFile:
        csvreader = csv.reader(trainFile)
        for idx, row in enumerate(csvreader):
            trainData.append(row + [-1])
            
    trainData = np.asarray(trainData, float)
    for dimension in range(0, trainData.shape[1]-2):
        #updating the estimate per dimension
        finalError.append(1)
        threshold.append("\0")
        polarization.append("\0")
        trainingError.append("\0")
        #need to sort traindata by dimension here
        trainData = trainData[trainData[:, dimension].argsort()]
        for featIdx, feature in enumerate(features):
            for idx, dataPoint in enumerate(trainData):
                #Identifying the feature that the datapoint belongs to
                #This section corresponds to sorting S using the jth coordinate (using dataPoint as center)
                if idx!=0: #passing the 'all left' case
                    trainData[:idx, 4] = features[1-featIdx] #we want this to be the other case
                    trainData[idx, 4] = features[1-featIdx]
                trainData[idx:, 4] = feature
                # print("This is the leq case", idx)
                # print(trainData)
                #Added this because the bounds on the latter argument are inclusive and I want leq
                
                error1 = calcError(trainData)

                #Used for the stump based on the error calculations
                if (finalError[dimension] > error1):
                    finalError[dimension] = error1
                    #stumpBaseError = feature
                    polarization[dimension] = -1
                    threshold[dimension] = dataPoint[dimension]
                    trainingError[dimension] = error1

                #Inverting the Direction:
                if idx!=0: #passing the 'all left' case
                    trainData[:idx, 4] = feature #we want this to be the other case
                trainData[idx:, 4] = features[1-featIdx]
                #Added this because the bounds on the latter argument are inclusive and I want geq
                #trainData[idx, 4] = feature[1-featIdx]
                
                error1 = calcError(trainData)
                
                # print("This is the geq case", idx)
                # print(trainData)
                #Used for the stump based on the error calculations
                if (finalError[dimension] > error1):
                    finalError[dimension] = error1
                    #stumpBaseError = feature
                    polarization[dimension] = 1
                    threshold[dimension] = dataPoint[dimension]
                    trainingError[dimension] = error1
    
    #Section for calculating the overall training error               
    trainingErrorOv = 0
    numcorrect = 0 #Total number of correct guesses
    for idx, dataPoint in enumerate(trainData):
        correctGuesses = np.full(trainData.shape[1]-2, False) #per iteration, which parameters are correct 
        for dimension in range(0, trainData.shape[1]-2):
            if polarization[dimension]>0: #if it is >=
                if (dataPoint[dimension]>=threshold[dimension]) & (dataPoint[-1] == features[-1]):
                    correctGuesses[dimension] = True
                elif (dataPoint[dimension]<threshold[dimension]) & (dataPoint[-1] == features[0]):
                    correctGuesses[dimension] = True
            else: #<=
                if (dataPoint[dimension]>threshold[dimension]) & (dataPoint[-1] == features[0]):
                    correctGuesses[dimension] = True
                elif (dataPoint[dimension]<=threshold[dimension]) & (dataPoint[-1] == features[-1]):
                    correctGuesses[dimension] = True
        neededVotes = np.ceil(float(correctGuesses.shape[0])/2)
        #print(neededVotes, correctGuesses, len(np.where(correctGuesses==True)[0]), numcorrect)
        if (len(np.where(correctGuesses==True)[0]) >= neededVotes):
            numcorrect= numcorrect+1
    trainingErrorOv = numcorrect/trainData.shape[0]
    
    print("TRAINING")
    print("Feature index: X1 \t X2 \t X3")
    print("Threshold: ", threshold)
    print("Polarization Parameters: ", polarization)
    print("Training Error: ", trainingError)
    print("Training Error Overall: ", 1-trainingErrorOv)
    # print("Feature Index: ", featureThreshold)

    #Testing Data Section###############################################################################
    ####################################################################################################
    #Reading in the training data and adding a column for estimated grouping
    with open(testDataPath, 'r') as testFile:
        csvreader = csv.reader(testFile)
        for idx, row in enumerate(csvreader):
            testData.append(row)
    
    #This section breaks down the test error by dimension
    testErrorByDim = []
    numcorrectByDim = []      
    testData = np.asarray(testData, float)
    for dimension in range(0, testData.shape[1]-1):
        #updating the estimate per dimension
        numcorrectByDim.append(0)
        for idx, dataPoint in enumerate(testData):
            #logging the total number of correct estimates
            if polarization[dimension]>0: #if it is >=
                if (dataPoint[dimension]>=threshold[dimension]) & (dataPoint[-1] == features[-1]):
                    numcorrectByDim[dimension] = numcorrectByDim[dimension]+1
                elif (dataPoint[dimension]<threshold[dimension]) & (dataPoint[-1] == features[0]):
                    numcorrectByDim[dimension] = numcorrectByDim[dimension]+1
            else: #<=
                if (dataPoint[dimension]>threshold[dimension]) & (dataPoint[-1] == features[0]):
                    numcorrectByDim[dimension] = numcorrectByDim[dimension]+1
                elif (dataPoint[dimension]<=threshold[dimension]) & (dataPoint[-1] == features[-1]):
                    numcorrectByDim[dimension] = numcorrectByDim[dimension]+1
    
    #This section assumes equivalent weights and shows net error
    numcorrect = 0 #Total number of correct guesses
    testError = 0
    for idx, dataPoint in enumerate(testData):
        correctGuesses = np.full(testData.shape[1]-1, False) #per iteration, which parameters are correct 
        for dimension in range(0, testData.shape[1]-1):
            if polarization[dimension]>0: #if it is >=
                if (dataPoint[dimension]>=threshold[dimension]) & (dataPoint[-1] == features[-1]):
                    correctGuesses[dimension] = True
                elif (dataPoint[dimension]<threshold[dimension]) & (dataPoint[-1] == features[0]):
                    correctGuesses[dimension] = True
            else: #<=
                if (dataPoint[dimension]>threshold[dimension]) & (dataPoint[-1] == features[0]):
                    correctGuesses[dimension] = True
                elif (dataPoint[dimension]<=threshold[dimension]) & (dataPoint[-1] == features[-1]):
                    correctGuesses[dimension] = True
        neededVotes = np.ceil(float(correctGuesses.shape[0])/2)
        #print(neededVotes, correctGuesses, len(np.where(correctGuesses==True)[0]), numcorrect)
        if (len(np.where(correctGuesses==True)[0]) >= neededVotes):
            numcorrect= numcorrect+1

    testErrorByDim = np.asarray(numcorrectByDim)/testData.shape[0]
    testError = numcorrect/testData.shape[0]
    
    print("\nTESTING")
    print("Test Error by Dimension: ", 1-testErrorByDim)
    print("Test Error Overall", 1-testError)