import numpy as np

#from scipy.optimize import linprog
import matplotlib.pyplot as plt
import nashpy as nash
from game_solvers import *


RANDOM_SEED = 1200
np.random.seed(RANDOM_SEED)


class RandomGame(object):
    """
    A matrix game with unknown payoff, 
    reward for every action pair is masked with a noisy feedback
    """

    def __init__(self, n1, n2, T):
        self.n1 = n1
        self.n2 = n2
        self._true_game = np.zeros((self.n1, self.n2))
        self.sp_value = 0
        self.T = T
        self.t = 0
        self.k = 0
        self.reset()

    def reset(self):
        self._true_game = np.random.uniform(size=(self.n1, self.n2))
        mu, nu, val = Nash_solver(self._true_game)
        self.sp_value = val

    def play(self, it, jt):
        eta = np.random.normal(0, 0.25)
        reward = self._true_game[it, jt] + eta
        self.t += 1
        if self.t == self.T:
            self.t = 0
            self.k += 1
            self.reset()
        return reward

    @property
    def true_game(self):
        return self._true_game


class ContextualGame(RandomGame):
    """
    A contextual type of random game, each payoff entry is unknown, reward feedback is noisy, 
    but a set of expert predicted games are generated episode by episode to indicate the underlying true game
    """

    def __init__(self, n1, n2, theta, T):
        self.theta = theta
        self.n1 = n1
        self.n2 = n2
        self._true_game = np.zeros((self.n1, self.n2))
        self.sp_value = 0
        self._expert_games = np.zeros((len(self.theta), self.n1, self.n2))
        self.t = 0
        self.k = 0
        self.T = T
        self.reset()

    def reset(self):
        self._expert_games = np.random.uniform(
            size=(len(self.theta), self.n1, self.n2))
        self._true_game = np.sum([self.theta[i] * self._expert_games[i]
                                  for i in range(len(self.theta))], axis=0)
        mu, nu, val = Nash_solver(self._true_game)
        self.sp_value = val

    def play(self, it, jt):
        eta = np.random.normal(0, 0.5)
        reward = self.true_game[it, jt] + eta
        #self.update_t()
        return reward

    def update_t(self):
        self.t += 1
        if self.t == self.T:
            self.t = 0
            self.k += 1
            self.reset()

    @property
    def true_game(self):
        return self._true_game

    @property
    def expert_games(self):
        return self._expert_games
