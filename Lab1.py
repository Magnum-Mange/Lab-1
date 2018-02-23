# Lab 1 for course DD2437
import numpy as np
import matplotlib.pyplot as plt

##class ANN():

class Perceptron():

    def __init__(inDim, outDim):
        self.inDim = inDim
        self.outDim = outDim
        self.W = np.random.multivariate_normal(np.zeros(outDim), 0.1 * np.identity(outDim), size = inDim)

    def deltaRule():


# Generating linearly separable test data
mean1 = np.array([1, 1])
mean2 = np.array([1, -1])
cov = np.array([[0.1, 0], [0, 0.1]])
lin_sep_data = np.random.multivariate_normal(mean1, cov, size = 100)
lin_sep_data = np.concatenate([lin_sep_data, np.ones([100, 1])], axis = 1)
print(np.shape(lin_sep_data))
lin_sep_data = np.concatenate([lin_sep_data, np.concatenate([np.random.multivariate_normal(mean2, cov, size = 100), -1 * np.ones([100, 1])], axis = 1)])
np.random.shuffle(lin_sep_data)

X = lin_sep_data.T[0:2, :]
X = np.concatenate([X, np.ones([1, 200])]) # Adding the bias term
patterns = lin_sep_data.T[2, :]

plt.scatter(lin_sep_data[:, 0], lin_sep_data[:, 1])
plt.show()
