# Lab 1 for course DD2437
import numpy as np
import matplotlib.pyplot as plt

class singleLayerPerceptron():

    def __init__(self, inDim, outDim):
        self.inDim = inDim
        self.outDim = outDim
        self.W = np.random.multivariate_normal(np.zeros(inDim), 0.1 * np.identity(inDim), size = outDim)
##        print(np.shape(self.W))
##        print(self.W)
        self.eta = 0.001

    def deltaRule(self, X, T, epochs, isSeq, testX, testPatterns, testSize):
        accuracies = []
        for epoch in range(epochs):
            oldW = np.copy(self.W)
            if isSeq:
                for patternNo in range(len(X[0])):
##                    print(self.W)
                    self.W += -self.eta * np.dot(np.asscalar(np.dot(self.W, X[:, patternNo]) - T[patternNo]), X[:, patternNo].T)
            else:
                self.W += -self.eta * np.dot((np.dot(self.W, X) - T), X.T)

            correct = 0
            for dataPointNo in range(testSize):
                if np.sign(testPatterns[dataPointNo]) == np.sign(self.predict(testX[:, dataPointNo], False)):
                    correct += 1

            accuracies.append(correct / testSize)

##            print(np.linalg.norm(self.W - oldW))

        return accuracies
        

    def perceptronLearning(self, X, T, epochs, isSeq, testX, testPatterns, testSize):
        accuracies = []
        for epoch in range(epochs):
            oldW = np.copy(self.W)
            if isSeq:
                for patternNo in range(len(X[0])):
##                    print(self.W)
                    self.W += -self.eta * np.dot(np.asscalar(np.sign(np.dot(self.W, X[:, patternNo])) - T[patternNo]), X[:, patternNo].T)
            else:
                self.W += -self.eta * np.dot((np.sign(np.dot(self.W, X)) - T), X.T)

            correct = 0
            for dataPointNo in range(testSize):
                if np.sign(testPatterns[dataPointNo]) == np.sign(self.predict(testX[:, dataPointNo], False)):
                    correct += 1

            accuracies.append(correct / testSize)

##            print(np.linalg.norm(self.W - oldW))

        return accuracies

    def predict(self, X, isSeq):
        if isSeq:
            return 0
        else:
            return np.dot(self.W, X)

class multiLayerPerceptron():

    def __init__(self, topology, eta):
        self.topology = topology
        self.eta = eta
        self.layers = []
        for layerNo in range(len(topology) - 1):
            self.layers.append(np.random.multivariate_normal(np.zeros(self.topology[layerNo] + 1), 0.1 * np.identity(self.topology[layerNo] + 1), size = self.topology[layerNo + 1]))
##        print(self.layers)
        self.layers = np.array(self.layers)
##        print("WOLOWOLO ", np.shape(self.layers))

    def transferFunction(self, x):
        return 2 / (1 + np.exp(-x)) - 1

    def transferFunctionDerivative(self, x):
        return ((1 + self.transferFunction(x)) * (1 - self.transferFunction(x))) / 2

    def backPropagation(self, X, T, epochs):

        thetas = []
        for layer in self.layers:
            thetas.append(np.zeros(np.shape(layer)))
        
        for epoch in range(epochs):

            # Forward pass

##            print(np.shape(X))
##            print(np.shape(O))
##            print(np.shape(self.layers[0]))
            signals = []
            O = self.transferFunction(np.dot(self.layers[0], X))
##            print(np.shape(O))
            signals.append(O)
            for layer in self.layers[1:]:
                O = np.concatenate((O, np.ones([1, len(O[0])])))
##                print(np.shape(layer))
##                print(np.shape(O))
                O = self.transferFunction(np.dot(layer, O))
                signals.append(O)

            # Backward pass

            errors = []
            layerError = np.multiply(O - T, self.transferFunctionDerivative(signals[len(signals) - 1]))
##            print(np.shape(signals[0]))
##            print(np.shape(signals[1]))
##            print(signals)
##            print(np.shape(self.transferFunctionDerivative(signals[len(signals) - 1])))
##            print(layerError)
##            print("WALALALALALA ", np.shape(layerError))
            errors.append(layerError)
##            print(len(self.layers))
##            print(np.shape(self.layers[1]))
            for layerNo in range(len(self.layers) - 1, 0, -1):
##                print(np.shape(self.layers[layerNo].T))
                layerError = np.multiply(np.dot(self.layers[layerNo].T, layerError), self.transferFunctionDerivative(signals[layerNo]))
##                print(np.shape(layerError))
                layerError = layerError[:len(layerError) - 1]
##                print(np.shape(layerError))
                errors.append(layerError)

            errors.reverse()
##            print(np.shape(errors))
##            print(errors)

##            print(layerError)
##            print(np.shape(layerError))

            # Weight update

            alpha = 0.9
            O = X
            for layerNo in range(len(self.layers)):
##                print(np.shape(thetas[layerNo]))
##                print(np.shape(errors[layerNo]))
##                print(np.shape(O.T))
                thetas[layerNo] = alpha * thetas[layerNo] - (1 - alpha) * np.dot(errors[layerNo], O.T)
                self.layers[layerNo] += self.eta * thetas[layerNo]

##            print(self.layers)
        
        return 0

    def predict(self, X):
        layerInput = X
        for layer in self.layers:
            print(np.shape(layer))
            print(np.shape(layerInput))
            layerInput = np.append(self.transferFunction(np.dot(layer, layerInput)), 1)
            
        print(np.shape(layerInput))
        return layerInput[:1]
        

def _31():
    # Generating linearly separable data
##    mean1 = np.array([1, 1])
##    mean2 = np.array([1, -1])
##    cov = np.array([[0.1, 0], [0, 0.1]])

    # Generating non-linearly separable data

    mean1 = np.array([1, 1])
    mean2 = np.array([1, 0.5])
    cov = np.array([[0.1, 0], [0, 0.1]])

    trainingData = np.random.multivariate_normal(mean1, cov, size = 100)
    trainingData = np.concatenate([trainingData, np.ones([100, 1])], axis = 1)
    trainingData = np.concatenate([trainingData, np.concatenate([np.random.multivariate_normal(mean2, cov, size = 100), -1 * np.ones([100, 1])], axis = 1)])

    plt.scatter(trainingData[0:100, 0], trainingData[0:100, 1], color = 'b')
    plt.scatter(trainingData[100:200, 0], trainingData[100:200, 1], color = 'r')
    plt.title("Sample of non-linearly separable data")
    plt.show()
    
    np.random.shuffle(trainingData)

    X = trainingData.T[0:2, :]
    X = np.concatenate([X, np.ones([1, 200])]) # Adding the bias term
    patterns = trainingData.T[2, :]

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

##    print(patterns)
##    print(X[:, 0])

    epochs = 50
    trials = 100
    accuracies = np.zeros(epochs)
    for trial in range(trials):
        perceptron = singleLayerPerceptron(3, 1)
        accuracies += perceptron.perceptronLearning(X, patterns, epochs, True, testX, testPatterns, testSize)
        print(accuracies)

    plt.plot(np.linspace(1, epochs, epochs), accuracies / trials)
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Perceptron learning rule with sequential learning, non-linearly separable data")
    plt.show()
##    correct = 0
##    for dataPointNo in range(testSize):
    ##    print(testPatterns[dataPointNo])
    ##    print(perceptron.predict(testX[:, dataPointNo], False))
##        if np.sign(testPatterns[dataPointNo]) == np.sign(perceptron.predict(testX[:, dataPointNo], False)):
##            correct += 1

##    print(correct / testSize)

def _32():
    
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

    testSize = 100
    testData = np.random.multivariate_normal(mean1, cov, size = testSize)
    testData = np.concatenate([testData, np.ones([testSize, 1])], axis = 1)
    testData = np.concatenate([testData, np.concatenate([np.random.multivariate_normal(mean2, cov, size = testSize), -1 * np.ones([testSize, 1])], axis = 1)])
    np.random.shuffle(testData)

    testX = testData.T[0:2, :]
    testX = np.concatenate([testX, np.ones([1, 2 * testSize])]) # Adding the bias term
    testPatterns = testData.T[2, :]

    epochs = 20
    topology = [2, 2, 1]
    eta = 0.001
    perceptron = multiLayerPerceptron(topology, eta)
##    print(np.shape(X))
##    print(np.shape(patterns))
    perceptron.backPropagation(X, patterns, epochs)
    correct = 0
    for dataPointNo in range(testSize):
        if np.sign(testPatterns[dataPointNo]) == np.sign(perceptron.predict(testX[:, dataPointNo])):
            correct += 1

    print(correct / testSize)

    return 0

_32()
    
