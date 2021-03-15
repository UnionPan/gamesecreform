import numpy as np
import pymc3 as pm

class Player(object):
    """
    This is the an opponent aware player who estimates a game and execute strategy accordingly
    """
    def __init__(self, game, policy, prior=0, gamma=None, IsAttacker=False):
        self.policy = policy
        self.prior = prior
        self.gamma = gamma
        if IsAttacker is not True:
            self.num_action = game.n1
            self.oppo_num_action = game.n2
        else:
            self.num_action = game.n2
            self.oppo_num_action = game.n1
        self._game_estimates = prior*np.ones((self.num_action, self.oppo_num_action))
        self.action_attempts = np.ones((self.num_action, self.oppo_num_action))
        self.k = 0      # episode number
        self.t = 0  # time number
        self.T = game.T
        self.last_action = None
        self.last_op_action = None
    
    def __str__(self):
        return 'f/{}'.format(str(self.policy))

    def reset(self):
        """
        Resets the agent's memory to an initial state.
        """
        self._game_estimates[:] = self.prior
        self.action_attempts[:] = 1
        self.last_action = None
        self.last_op_action = None
        self.t = 0
    
    def choose(self):
        action = self.policy.choose(self)
        self.last_action = action
        return action

    def observe(self, reward, jt):
        self.last_op_action = jt
        self.action_attempts[self.last_action, self.last_op_action] += 1

        if self.gamma is None:
            g = 1 / self.action_attempts[self.last_action, self.last_op_action]
        else:
            g = self.gamma
        q = self._game_estimates[self.last_action, self.last_op_action]

        self._game_estimates[self.last_action, self.last_op_action] += g*(reward - q)
        self.t += 1
        if self.t == self.T: # an episode is over
            self.reset()
            self.k += 1
            
    @property
    def game_estimates(self):
        return self._game_estimates


class AdversarialPlayer(object):
    """
    This is the an opponent unaware player who uses cumulative value estimates to exploit and explore with a fading probability 
    """

    def __init__(self, game, policy, prior=0, gamma=None, IsAttacker = False):
        self.policy = policy
        if IsAttacker is not True:
            self.num_action = game.n1
            self.oppo_num_action = game.n2
        else:
            self.num_action = game.n2
            self.oppo_num_action = game.n1
        self.prior = prior
        self._cumulative_estimates = prior * np.ones(self.num_action)
        self.action_attempts = np.zeros(
            (self.num_action, self.oppo_num_action))
        self.k = 0      # episode number
        self.t = 0  # time number
        self.T = game.T
        self.last_action = None
        self.last_op_action = None

    def __str__(self):
        return 'f/{}'.format(str(self.policy))

    def reset(self):
        """
        Resets the agent's memory to an initial state.
        """
        self._cumulative_estimates[:] = self.prior
        self.action_attempts[:] = 1
        self.last_action = None
        self.last_op_action = None
        self.t = 0

    def choose(self):
        action = self.policy.choose(self)
        self.last_action = action
        return action

    def observe(self, reward, jt):
        self.last_op_action = jt
        self.action_attempts[self.last_action, self.last_op_action] += 1
        mu = self.policy.mu

        self._cumulative_estimates[self.last_action] += reward / mu[self.last_action]
        self.t += 1
        if self.t == self.T:  # an episode is over
            self.reset()
            self.k += 1

    @property
    def cumulative_estimates(self):
        return self._cumulative_estimates



class ContextualPlayer(object):
    """
    This is the an opponent aware player who estimates a game and execute strategy accordingly
    this player is able to estimate the true game based on expert predictions of the game
    """

    def __init__(self, cgame, policy, prior=0, gamma=None, lamb = 0.1):
        self.policy = policy
        self.num_action = cgame.n1
        self.oppo_num_action = cgame.n2
        self.prior = prior
        self.gamma = gamma
        self._game_estimates = prior * \
            np.ones((self.num_action, self.oppo_num_action))
        self.action_attempts = np.zeros(
            (self.num_action, self.oppo_num_action))
        self.k = 0      # episode number
        self.t = 0      # time number
        self.T = cgame.T
        self.last_action = None
        self.last_op_action = None
        self.expert_predictions = cgame.expert_games
        self.S = np.shape(self.expert_predictions)[0]
        self.V = lamb * np.eye(self.S)
        self.Y = np.zeros(self.S)
        self.thetahat = 1/ self.S * np.ones(self.S)

    def __str__(self):
        return 'f/{}'.format(str(self.policy))

    def reset(self):
        """
        Resets the agent's memory to an initial state.
        """
        self._game_estimates[:] = self.prior
        self.action_attempts[:] = 1
        self.last_action = None
        self.last_op_action = None
        self.t = 0

    def choose(self):
        action = self.policy.choose()
        self.last_action = action
        return action

    def observe(self, reward, jt):
        self.last_op_action = jt
        self.action_attempts[self.last_action, self.last_op_action] += 1

        self.t += 1
        zitjt = self.expert_predictions[:,
                                        self.last_action, self.last_op_action]
        self.ZTZ += np.array(np.ma.outerproduct(zitjt, zitjt))
        self.ZTX += reward * zitjt
        if self.t == self.T:  # an episode is over
            self.reset()
            self.k += 1
            self.estimate()

    def estimate(self):
        self.V += self.ZTZ
        self.Y += self.ZTX
        self.theta_hat = np.linalg.inv(self.V) @ self.Y
        self.expert_predictions = self.cgame.expert_games
        self._game_estimates = np.sum([self.theta_hat[i] * self.expert_predictions[i] for i in range(self.S)], axis=0)
        self.policy.generatemu(self)


    @property
    def game_estimates(self):
        return self._game_estimates
