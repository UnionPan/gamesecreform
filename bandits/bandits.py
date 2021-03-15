import numpy as np
import pymc3 as pm

class MultiArmedBandit(object):
    """
    The All 0 Bandits
    """
    def __init__(self, k):
        self.k = k
        self.action_values = np.zeros(k)
        self.optimal = 0

    def reset(self):
        self.action_values = np.zeros(self.k)
        self.optimal = 0

    def pull(self, action):
        return 0, True

class GaussianBandit(MultiArmedBandit):
    """
    """
    def __init__(self, k, mu=0, sigma=1):
        super(GaussianBandit, self).__init__(k)
        self.mu = mu
        self.sigma = sigma
        self.reset()

    def reset(self):
        self.action_values = np.random.normal(self.mu, self.sigma, self.k)
        self.optimal = np.argmax(self.action_values)

    def pull(self, action):
        return (np.random.normal(self.action_values[action]), action == self.optimal)


class BinomialBandit(MultiArmedBandit):
    """
    A general instance of Bandits, every pull of the bandit will return total number of success out of N trials, each trial can represent a binary user rating {1, 0}
    and we have k such bandits, each represent a content. Therefore, Bernouli Bandits is just a special case of Binomial Bandits  
    """
    def __init__(self, k, n, p=None, t=None):
        super(BinomialBandit, self).__init__(k)
        self.n = n
        self.p = p
        self.t = t
        self.model = pm.Model()
        with self.model:
            self.bin = pm.Binomial('binomial', n=n * np.ones(k, dtype=np.int), p=np.ones(k) / n, shape=(1, k), transform=None)
            
        self._samples = None
        self._cursor = None 

        self.reset()

    def reset(self):
        if self.p is None:
            self.action_values = np.random.uniform(size=self.k)
        else:
            self.action_values = self.p
        
        self.bin.distribution.p = self.action_values
        if self.t is not None:
            self._samples = self.bin.random(size=self.t).squeeze()
            self._cursor = 0

        self.optimal = np.argmax(self.action_values)

    def pull(self, action):
        return self.sample[action], action == self.optimal

    @property
    def sample(self):
        if self._samples is None:
            return self.bin.random()
        else:
            val = self._samples[self._cursor]
            self._cursor += 1
            return val

class BernoulliBandit(BinomialBandit):
    def __init__(self, k, p=None, t=None):
        super(BernoulliBandit, self).__init__(k, 1, p=p, t=t)


if __name__ == "__main__":
    binbandit = BernoulliBandit(k=5, t=10)
    action = np.argmax(binbandit.action_values)
    print(action)
    print(binbandit.sample)
    