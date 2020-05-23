import numpy as np
import pickle
import random
LEAKY = 0.1
INPUTS = 17
OUTPUTS = 2
SIZES = [INPUTS, 50, 50, 50, OUTPUTS]
LEARNING_RATE = 0.0005
eta = LEARNING_RATE
BATCH_SIZE = 100
TEST_SIZE = 10
TOTAL_SIZE = BATCH_SIZE + TEST_SIZE

def actfct(x):
    #return np.maximum(LEAKY*x, x)
    return np.tanh(x)

def dactfct(x):
##    if x >= 0:
##        return 1
##
##    return LEAKY
    return 1 - np.tanh(x)**2

vecDactfct = np.vectorize(dactfct)

def costfct(z, y):
    return np.sum(((z - y)**2)) / 2

def dcostfct(z, y):
    return (z-y)

def percentOff(z, y):
    return 100 * np.sum(np.divide(np.abs(z - y), np.maximum(y, 0.0005)))

class NN():
    def __init__(self):
        #for each layer, create a matrix of weights and a vector of biases
        self.weights = []
        self.biases = []
        for i in range(len(SIZES) - 1):
            self.weights.append(2 * np.random.rand(SIZES[i+1], SIZES[i]) - 1)
            self.biases.append(2 * np.random.rand(SIZES[i+1], 1) - 1)

        #get input data

    def ff(self, x):
        a = x[np.newaxis].T
        As = [x]
        Zs = []
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, a) + b
            Zs.append(z)
            a = actfct(z)
            As.append(a)

        return As, Zs

    def backProp(self, x, y):
        As, Zs = self.ff(x)
        
        delta = np.multiply(dcostfct(As[-1], y[np.newaxis].T), vecDactfct(Zs[-1]))

        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        nabla_b[-1] = delta
        nabla_w[-1] = delta.dot(As[-2].T)

        for L in range(2, len(SIZES) - 1):
            delta = np.multiply(self.weights[-L+1].T.dot(delta), vecDactfct(Zs[-L]))
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

        self.weights = [w-(eta/len(Xs))*nw for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b-(eta/len(Xs))*nb for b, nb in zip(self.biases, nabla_b)]

    def testBatch(self, Xs, Ys):
        averageOff = 0
        for x, y in zip(Xs, Ys):
            As, Zs = self.ff(x)
                
            averageOff += percentOff(As[-1], y[np.newaxis].T)

        averageOff /= len(Xs)

        return averageOff

files = ['0', '1', '2', '3']

print("Loading data")
examples = np.loadtxt('../data/data' + files[0] + '.txt', delimiter=", ")
answers = np.loadtxt('../data/answers' + files[0] + '.txt', delimiter=", ")
sizes = len(examples)
INPUTS = len(examples[0])
OUTPUTS = len(answers[0])
SIZES = [INPUTS, 50, 50, 50, OUTPUTS]
print("Done Loading")

print("Initializing NN")
net = NN()
print("Done Initializing")

currRound = -1
while (True):
    random.shuffle(files)
    for file in files:
        print("Loading data")
        examples = np.loadtxt('../data/data' + file + '.txt', delimiter=", ")
        answers = np.loadtxt('../data/answers' + file + '.txt', delimiter=", ")
        sizes = len(examples)
        print("Done Loading")
        
        print("Shuffling data")
        order = list(zip(examples, answers))
        random.shuffle(order)
        print("Done Shuffling")
        
        currRound += 1
        print("Starting Test Round:", currRound)
        for i in range(int(sizes / TOTAL_SIZE)):
            #run one batch
            #perform one test
            currBatch = order[TOTAL_SIZE * i : TOTAL_SIZE * i + BATCH_SIZE]
            net.miniBatch(*zip(*currBatch))
            
            currBatch = order[TOTAL_SIZE * i + BATCH_SIZE : TOTAL_SIZE * (i + 1)]
            
            print("Trial", i, ":", net.testBatch(*zip(*currBatch)), "% off")

    #after going thru the entire training set, save the current NN
    print("Saving backup")
    with open('backup.pkl', 'wb') as outfile:
        pickle.dump([net.weights, net.biases], outfile, pickle.HIGHEST_PROTOCOL)
    
    
