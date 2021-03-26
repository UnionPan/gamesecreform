import numpy as np
from game_solvers import *


class GamePolicy(object):
    """
    A generic game policy, decribes how action is sampled based on history
    """

    def __str__(self):
        return 'A game policy'

    def choose(self, agent):
        return 0


class ExploringMinimax(GamePolicy):
    """
    A naive minimax game policy uses the game estimate to compute the minimax strategy, it uses a epsilon parameter to randomly draw some actions
    """

    def __init__(self, epsilon=0.1):
        self.epsilon = epsilon
        self._mu = None

    def __str__(self):
        return '\u03B5-greedy exploring minimax(\u03B5 = {})'.format(self.epsilon)

    def choose(self, player):
        if np.random.random() < self.epsilon:
            self._mu = np.ones(player.num_action) / \
                np.sum(np.ones(player.num_action))
            return np.random.choice(player.num_action)
        else:
            self._mu, nu, val = Nash_solver(player.game_estimates)
            cdf = np.cumsum(self._mu)
            s = np.random.random()
            return np.where(s < cdf)[0][0]

    @property
    def mu(self):
        return self._mu


class GreedyMinimax(ExploringMinimax):
    def __init__(self):
        super(GreedyMinimax, self).__init__(0)

    def __str__(self):
        return 'greedily doing minimax'


class RandomPolicy(ExploringMinimax):
    def __init__(self):
        super(RandomPolicy, self).__init__(1)

    def __str__(self):
        return 'random policy'


class UCBMinimax(GamePolicy):
    """
    An UCBminimax policy uses UCB estimates of each game entry to construct the matrix estimate,
    this estimate already encodes a exploration factor in it
    """

    def __init__(self, c=2):
        self.c = c
        self._mu = None

    def __str__(self):
        return 'ucb game policy {}'.format(self.c)

    def choose(self, player):
        # print(player.T)
        exploration = 2 * np.log(2 * player.T ** 2 * player.num_action *
                                 player.oppo_num_action) / player.action_attempts
        exploration[np.isnan(exploration)] = 0
        exploration = np.power(exploration, 1 / self.c)
        # print(exploration)
        ucbgame = player.game_estimates + exploration
        # print(player.game_estimates)
        # print(ucbgame)
        self._mu, nu, val = Nash_solver(ucbgame)
        cdf = np.cumsum(self._mu)
        s = np.random.random()
        return np.where(s < cdf)[0][0]

    @property
    def mu(self):
        return self._mu


class EXP3(GamePolicy):
    """
    Exponential weights exploration and exploitation for adversarial MAB
    """

    def __init__(self):
        self._mu = None

    def __str__(self):
        return 'EXP3 policy'

    def choose(self, advplayer):
        s = advplayer.cumulative_estimates
        gammat = min(np.sqrt(advplayer.num_action *
                             np.log(advplayer.num_action) / advplayer.t), 1)
        # print('gammat {}'.format(gammat))
        rhot = np.sqrt(2 * np.log(advplayer.num_action) /
                       max(advplayer.t, 1) * advplayer.num_action)
        # print('rhot {}'.format(rhot))
        wt = np.exp(rhot * advplayer.cumulative_estimates) / \
            np.sum(np.exp(rhot * advplayer.cumulative_estimates))
        wt[np.isnan(wt)] = 1
        self._mu = gammat / advplayer.num_action * \
            np.ones(advplayer.num_action) + (1 - gammat) * wt
        cdf = np.cumsum(self._mu)
        # print(self._mu)
        # print(advplayer.t)
        # print(advplayer.cumulative_estimates)
        no = np.random.random()
        return np.where(no < cdf)[0][0]

    @property
    def mu(self):
        return self._mu


class OFULinMat(GamePolicy):
    """
    Optimism in the face of Uncertainty in Linear Parameterized Matrix
    It assign every unknown game entry an exploration factor.
    Only compatible with contextual player
    """

    def __init__(self, B=3):
        self._mu = None
        self.B = B

    def __str__(self):
        return 'OFULinMat policy'

    def generatemu(self, contextualplayer):
        self.delta = 1 / (contextualplayer.T * contextualplayer.K)
        ucbgame = contextualplayer.game_estimates
        # print(contextualplayer.game_estimates)
        betak = np.sqrt(2 * np.log(np.sqrt(np.linalg.det(contextualplayer.V)) / (
            self.delta * contextualplayer.lamb ** 0.5))) + contextualplayer.lamb ** 0.5 * self.B

        for i in range(contextualplayer.num_action):
            for j in range(contextualplayer.oppo_num_action):
                zij = contextualplayer.expert_predictions[:, i, j]
                Vinv = np.linalg.inv(contextualplayer.V)
                exploration = betak * np.sqrt(np.dot(zij, Vinv @ zij))
                if np.isnan(exploration):
                    exploration = 0
                ucbgame[i, j] += exploration
        self._mu, nu, val = Nash_solver(ucbgame)

    def choose(self):
        cdf = np.cumsum(self._mu)
        s = np.random.random()
        return np.where(s < cdf)[0][0]

    @ property
    def mu(self):
        return self._mu


class LinMat(GamePolicy):
    """
    Optimism in the face of Uncertainty in Linear Parameterized Matrix
    It assign every unknown game entry an exploration factor.
    Only compatible with contextual player
    """

    def __init__(self, B=1):
        self._mu = None
        self.B = B

    def __str__(self):
        return 'OFULinMat policy'

    def generatemu(self, contextualplayer):
        ucbgame = contextualplayer.game_estimates
        self._mu, nu, val = Nash_solver(ucbgame)

    def choose(self):
        cdf = np.cumsum(self._mu)
        s = np.random.random()
        return np.where(s < cdf)[0][0]

    @ property
    def mu(self):
        return self._mu


class OmniPolicy(GamePolicy):
    def __init__(self):
        self._mu = None

    def choose(self, oplayer):
        self._mu = oplayer.mu
        cdf = np.cumsum(self._mu)
        s = np.random.random()
        return np.where(s < cdf)[0][0]

    @ property
    def mu(self):
        return self._mu
