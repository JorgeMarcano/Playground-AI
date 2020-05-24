import gym
import numpy as np
from DQL import DQLBot

class CartPoleDQL(DQLBot):
    def __init__(self):
        self.env = gym.make("CartPole-v0").unwrapped
        self.env.reset()
        self.currentScreen = None
        self.imgW = self.render('rgb_array').shape[1]
        self.imgH = self.render('rgb_array').shape[0]
        self.layers = [self.imgW * self.imgH * 3, 24, 32, self.env.action_space.n]
        super().__init__(self.layers)

    def performAction(self, s, a):
        nextScreen, reward, isDone = self.env.step(a)[0:3]

        if self.currentScreen is None or isDone:
            self.currentScreen = self.getScreen()
            return np.zeros((self.imgW * self.imgH * 3, 1)), reward, isDone

        else:
            s1 = self.currentScreen
            s2 = self.getScreen()
            self.currentScreen = s2
            return (s2 - s1), reward, isDone

    def getScreen(self):
        screen = self.render('rgb_array')
        screen = screen.reshape(self.imgW * self.imgH * 3, 1) / 255

    def reset(self):
        self.env.reset()
        self.currentScreen = None
        self.currStateArray = np.zeros((self.imgW * self.imgH * 3, 1))

    def close(self):
        self.env.close()

    def render(self, mode='human'):
        return self.env.render(mode)

if  __name__ == "__main__":
    test = CartPoleDQL()
    test.train()
    test.close()
