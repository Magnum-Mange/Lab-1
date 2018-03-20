# Lab 1 for course DD2437
import numpy as np
import matplotlib.pyplot as plt

##class ANN():

class singleLayerPerceptron():

    def __init__(self, inDim, outDim):
        self.inDim = inDim
        self.outDim = outDim
        self.W = np.random.multivariate_normal(np.zeros(inDim), 0.1 * np.identity(inDim), size = outDim)
##        print(np.shape(self.W))
##        print(self.W)
        self.eta = 0.001

    def deltaRule(self, X, T, epochs, isSeq):
        for epoch in range(epochs):
            oldW = np.copy(self.W)
            if isSeq:
                for patternNo in range(len(X)):
                    self.W += -self.eta * np.dot(np.dot(self.W, X[patternNo]) - T[patternNo], X[patternNo].T)
            else:
                self.W += -self.eta * np.dot((np.dot(self.W, X) - T), X.T)

            print(np.linalg.norm(self.W - oldW))
        

    def perceptronLearning(self, inData, isSeq):
        #todo
        return 0

    def predict(self, X, isSeq):
        if isSeq:
            return 0
        else:
            return np.dot(self.W, X)

class multiLayerPerceptron():

    def __init(self, L, eta, inDim, outDim):
        self.L = L
        self.eta = eta
        self.inDim = inDim
        self.outDim = outDim
        self.layers = np.random.multivariate_normal(np.zeros(inDim), 0.1 * np.identity(inDim), size = (L, outDim))

    def phi(self, x):
        return 2 / (1 + np.exp(-x)) - 1

    def backProp(self, X, T, epochs):

        # Forward pass

        return 0
        

def _31():
    # Generating linearly separable data
    ##mean1 = np.array([1, 1])
    ##mean2 = np.array([1, -1])
    ##cov = np.array([[0.1, 0], [0, 0.1]])

    # Generating non-linearly separable data

    mean1 = np.array([1, 1])
    mean2 = np.array([1, 0.5])
    cov = np.array([[0.1, 0], [0, 0.1]])

    trainingData = np.random.multivariate_normal(mean1, cov, size = 100)
    trainingData = np.concatenate([trainingData, np.ones([100, 1])], axis = 1)
    trainingData = np.concatenate([trainingData, np.concatenate([np.random.multivariate_normal(mean2, cov, size = 100), -1 * np.ones([100, 1])], axis = 1)])
    np.random.shuffle(trainingData)

    X = trainingData.T[0:2, :]
    X = np.concatenate([X, np.ones([1, 200])]) # Adding the bias term
    patterns = trainingData.T[2, :]

    plt.scatter(trainingData[:, 0], trainingData[:, 1])
    plt.show()

    testSize = 100
    testData = np.random.multivariate_normal(mean1, cov, size = testSize)
    testData = np.concatenate([testData, np.ones([testSize, 1])], axis = 1)
    testData = np.concatenate([testData, np.concatenate([np.random.multivariate_normal(mean2, cov, size = testSize), -1 * np.ones([testSize, 1])], axis = 1)])
    np.random.shuffle(testData)

    testX = testData.T[0:2, :]
    testX = np.concatenate([testX, np.ones([1, 2 * testSize])]) # Adding the bias term
    testPatterns = testData.T[2, :]
    ##print(testX)
    ##print(testPatterns)

    plt.scatter(testData[:, 0], testData[:, 1])
    plt.show()

    perceptron = singleLayerPerceptron(3, 1)
    perceptron.deltaRule(X, patterns, 20, False)
    correct = 0
    for dataPointNo in range(testSize):
    ##    print(testPatterns[dataPointNo])
    ##    print(perceptron.predict(testX[:, dataPointNo], False))
        if np.sign(testPatterns[dataPointNo]) == np.sign(perceptron.predict(testX[:, dataPointNo], False)):
            correct += 1

    print(correct / testSize)

def _32():
    return 0

_31()
    
