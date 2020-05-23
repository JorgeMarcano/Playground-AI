import numpy as np
import random

examples = np.loadtxt('../data.txt', delimiter=", ")
answers = np.loadtxt('../answers.txt', delimiter=", ")

order = list(zip(examples, answers))
random.shuffle(order)

#do a random matrix 2x18
randomMat = np.random.rand(2, 18)

for row in order:
    result = randomMat.dot(examples[row].T)
    print(result - answers[row])
