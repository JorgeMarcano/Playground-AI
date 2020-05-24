import numpy as np
import random

class QLBot():
    def __init__(self, Snb, Anb):
        self.q = np.zeros((Snb, Anb))
        self.stateSpace = Snb
        self.actionSpace = Anb

        self.epsilon = 1
        self.epsMax = 1
        self.epsMin = 0.01
        self.epsDecay = 0.001

        self.alpha = 0.05
        self.gamma = 0.99

        self.episodeNb = 30000
        self.maxStepsEpisodes = 100

        self.averageNb = 1000

    def R(self, s, a):
        pass

    def bellman(self, sPrime, reward):
        return (reward + self.gamma * np.max(self.q[sPrime, : ]))

    def updateQ(self, s, a, sPrime, reward):
        self.q[s, a] = (1 - self.alpha) * self.q[s, a] + self.alpha * self.bellman(sPrime, reward)

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
                action = np.argmax(self.q[self.currState, : ])
            else:
                #exploration
                action = random.randrange(self.actionSpace)

            nextState, reward, end = self.performAction(self.currState, action)

            self.updateQ(self.currState, action, nextState, reward)

            self.currState = nextState

            currEpReward += reward

            if end:
                break

        return currEpReward

    def runAllEpisode(self):
        rewardsList = []

        for episode in range(self.episodeNb):
            reward = self.runEpisode()

            #update epsilon using exponential decay
            self.epsilon = self.epsMin + (self.epsMax - self.epsMin) * np.exp(- self.epsDecay * episode)

            #save the reward
            rewardsList.append(reward)

            if ((episode + 1) % self.averageNb) == 0:
                print(episode, ":", str(sum(rewardsList)/self.averageNb))
                rewardsList = []

    def train(self):
        self.runAllEpisode()

    def trainContinous(self):
        rewardsList = []
        episode = 0

        while True:
            reward = self.runEpisode()

            #update epsilon using exponential decay
            self.epsilon = self.epsMin + (self.epsMax - self.epsMin) * np.exp(- self.epsDecay * episode)

            #save the reward
            rewardsList.append(reward)

            if ((episode + 1) % self.averageNb) == 0:
                print(episode, ":", str(sum(rewardsList)/self.averageNb))
                rewardsList = []

            episode += 1

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
