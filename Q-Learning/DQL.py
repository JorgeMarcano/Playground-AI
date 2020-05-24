import numpy as np
import random
from DNN import DNN

class DQLBot():
    def __init__(self, layers):
        self.actionSpace = layers[-1]
        self.stateSpace = layers[0]

        self.replayMemSize = 100000
        self.replayMem = []
        self.replayCount = 0
        self.replayBatchSize = 256

        self.epsilon = 1
        self.epsMax = 1
        self.epsMin = 0.01
        self.epsDecay = 0.001

        self.alpha = 0.001
        self.gamma = 0.999

        self.q = DNN(layers)
        self.targetNet = DNN(layers)
        self.targetNet.setValues(self.q.weights, self.q.biases)
        self.updateTargetRate = 10

        self.episodeNb = 1000
        self.maxStepsEpisodes = 10000000000

    def R(self, s, a):
        pass

    def bellman(self, sPrime, reward):
        return (reward + self.gamma * self.targetNet.ff(sPrime))

    def updateQ(self, s, a, sPrime, reward):
        optimalQ = bellman(sPrime, reward)

        self.q.backProp(s, optimalQ)

    #must return tuple of next state, reward, isEpisodeDone
    def performAction(self, s, a):
        pass

    def reset(self):
        pass

    def runEpisode(self):
        #reset the state
        self.reset()

        end = False
        currEpReward = 0

        #for every possible step
        for step in range(self.maxStepsEpisodes):

            #exploration or exploitation
            isExploration = random.uniform(0, 1)
            if isExploration > self.epsilon:
                #exploitation
                action = np.argmax(self.q.ff(self.currStateArray))
            else:
                #exploration
                action = random.randrange(self.actionSpace)

            nextStateArray, reward, end = self.performAction(self.currStateArray, action)

            #append memory if less than size
            if self.replayCount < self.replayMemSize:
                self.replayMem.append((self.currStateArray.copy(), action, nextStateArray.copy(), reward))
            else:
                self.replayMem[replayMemIndx] = (self.currStateArray.copy(), action, nextStateArray.copy(), reward)

            self.replayMemIndx += 1
            self.replayMemIndx %= self.replayMemSize
            self.replayCount += 1

            #sample random batch and train on that
            if len(self.replayMem) > self.replayBatchSize:
                for experience in random.sample(self.replayMem, self.replayBatchSize):
                    self.updateQ(*experience)

            self.currStateArray = nextStateArray

            currEpReward += reward

            if end:
                break

        return currEpReward

    def runAllEpisode(self):
        rewardsList = []

        self.replayMemIndx = 0

        for episode in range(self.episodeNb):
            reward = self.runEpisode()

            #update epsilon using exponential decay
            self.epsilon = self.epsMin + (self.epsMax - self.epsMin) * np.exp(- self.epsDecay * episode)

            #save the reward
            rewardsList.append(reward)

            if episode % self.updateTargetRate == 0:
                self.targetNet.setValues(self.q.weights, self.q.biases)

    def train(self):
        self.runAllEpisode()

    def test(self):
        #reset the state
        self.reset()

        end = False
        currEpReward = 0

        #for every possible step
        for step in range(self.maxStepsEpisodes):

            #always exploitation
            action = np.argmax(self.q[self.currState, : ])

            nextState, reward, end = self.performAction(self.currState, action)

            self.currState = nextState

            currEpReward += reward

            if end:
                break

        return currEpReward
