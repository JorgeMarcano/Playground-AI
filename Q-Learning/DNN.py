import numpy as np
import random
import pickle

class DNN():
    def __init__(self, layers):
        #for each layer, create a matrix of weights and a vector of biases
        self.weights = [2 * np.random.rand(layers[i+1], layers[i]) - 1 for i in range(len(layers) - 1)]
        self.biases = [2 * np.random.rand(layers[i+1], 1) - 1 for i in range(len(layers) - 1)]

        self.eta = 0.1

        self.leaky = 0

        #self.vecDactfct = np.vectorize(dactfct)

    def setValues(self, weights, biases):
        self.weights = [i.copy() for i in weights]
        self.biases = [i.copy() for i in biases]

    def ff(self, x):
        a = x
        As = [x]
        Zs = []
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, a) + b
            Zs.append(z)
            a = self.actfct(z)
            As.append(a)

        return As, Zs

    def actfct(x):
        return np.maximum(self.leaky*x, x)
    ##    return np.tanh(x)

    def dactfct(x):
        if x >= 0:
            return 1

        return self.leaky
    ##    return 1 - np.tanh(x)**2

    def costfct(z, y):
        return np.sum(((z - y)**2)) / 2

    def dcostfct(z, y):
        return (z-y)

    def backProp(self, x, y):
        As, Zs = self.ff(x)

        delta = np.multiply(dcostfct(As[-1], y), self.dactfct(Zs[-1]))

        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        nabla_b[-1] = delta
        nabla_w[-1] = delta.dot(As[-2].T)

        for L in range(2, len(SIZES) - 1):
            delta = np.multiply(self.weights[-L+1].T.dot(delta), self.dactfct(Zs[-L]))
            nabla_b[-L] = delta
            nabla_w[-L] = As[-L-1].T.dot(delta)

        return nabla_b, nabla_w

    def miniBatch(self, Xs, Ys):
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        for x, y in zip(Xs, Ys):
            delta_nabla_b, delta_nabla_w = self.backProp(x, y)
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]

        self.weights = [w-(self.eta/len(Xs))*nw for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b-(self.eta/len(Xs))*nb for b, nb in zip(self.biases, nabla_b)]
