
from bandits.policy import EpsilonGreedyPolicy, GreedyPolicy, UCBPolicy
import matplotlib
matplotlib.use('qt4agg')
from bandits.agents import Agent, BetaAgent
from bandits.bandits import BernoulliBandit, BinomialBandit
from bandits.environment import Environment
import PyQt5

class BernoulliExample:
    label = 'Bayesian Bandits - Bernoulli'
    bandit = BernoulliBandit(k = 10, t=3 * 1000)
    agents = [
        Agent(bandit, EpsilonGreedyPolicy(0.1)),
        Agent(bandit, UCBPolicy(1)),
        BetaAgent(bandit, GreedyPolicy())
    ]

class BinomialExample:
    label = 'Bayesian Bandits - Binomial'
    bandit = BinomialBandit(k = 10, n = 5, t=3 * 1000)
    agents = [
        Agent(bandit, EpsilonGreedyPolicy(0.1)),
        Agent(bandit, UCBPolicy(1)),
        BetaAgent(bandit, GreedyPolicy())
    ]

if __name__ == "__main__":
    experiments = 500
    trials = 1000

    example = BernoulliExample()

    env = Environment(example.bandit, example.agents, example.label)
    scores, optimal = env.run(trials, experiments)
    env.plot_results(scores, optimal)
    env.plot_beliefs()
