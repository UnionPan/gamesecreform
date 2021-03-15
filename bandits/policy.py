import numpy as np

class Policy(object):
    """
    The policy describes how the action is sampled based on history.
    """
    def __str__(self):
        return 'generic policy'

    def choose(self, agent):
        return 0


class EpsilonGreedyPolicy(Policy):
    def __init__(self, epsilon):
        self.epsilon = epsilon
    
    def __str__(self):
        return '\u03B5-greedy (\u03B5 = {})'.format(self.epsilon)

    def choose(self, agent):
        if np.random.random() < self.epsilon:
            return np.random.choice(len(agent.value_estimates))
        else:
            action = np.argmax(agent.value_estimates)
            check = np.where(agent.value_estimates == agent.value_estimates[action])[0]
            if len(check) == 1:
                return action
            else:
                return np.random.choice(check)


class GreedyPolicy(EpsilonGreedyPolicy):
    def __init__(self):
        super(GreedyPolicy, self).__init__(0)

    def __str__(self):
        return 'greedy'



class RandomPolicy(EpsilonGreedyPolicy):
    """
    The Random Policy randomly selects from all available actions with no consideration of value estimates
    """
    def __init__(self):
        super(RandomPolicy, self).__init__(1)
        
    def __str__(self):
        return 'random'

class UCBPolicy(Policy):
    """
    Implementation of UCB policy
    """
    def __init__(self, c):
        self.c = c

    def __str__(self):
        return 'UCB (C = {})'.format(self.c)

    def choose(self, agent):
        exploration = np.log(agent.t + 1) / agent.action_attempts
        exploration[np.isnan(exploration)] = 0
        exploration = np.power(exploration, 1 / self.c)
        
        q = agent.value_estimates
        action = np.argmax(q)
        check = np.where(q == q[action])[0]
        if len(check) == 1:
            return action
        else:
            return np.random.choice(check)


class SoftmaxPolicy(Policy):
    """
    Implementation of softmax policy
    """
    def __str__(self):
        return 'Softmax'

    def choose(self, agent):
        q = agent.value_estimates
        pi = np.exp(q) / np.sum(np.exp(a))
        cdf = np.cumsum(pi)
        s = np.random.random()
        return np.where(s < cdf)[0][0]
