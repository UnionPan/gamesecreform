import numpy as np
import pymc3 as pm

RANDOM_SEED = 1200
np.random.seed(RANDOM_SEED)

class StochasticGame(object):
    def __init__(defender, attacker, thetastar):
        self.game_matrix = np.zeros()
        self.theta = thetastar
        self.n1 = 

    def reset(self,):
        self.expert = expert
        self.game_matrix = np.sum(thetastar[i] * for i in range(len(thetastar))
        
    def play(self, it, jt):
        eta = pm.Uniform.sample()
        reward = self.game_matrix[it - 1, jt - 1] + eta
        

