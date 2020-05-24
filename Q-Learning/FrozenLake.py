import gym
from QL import QLBot

class FrozenLakeQL(QLBot):
    def __init__(self):
        self.env = gym.make("FrozenLake-v0")
        super().__init__(self.env.observation_space.n, self.env.action_space.n)

    def performAction(self, s, a):
        return self.env.step(a)[0:3]

    def reset(self):
        self.currState = self.env.reset()

if  __name__ == "__main__":
    test = FrozenLakeQL()
    test.trainContinous()
