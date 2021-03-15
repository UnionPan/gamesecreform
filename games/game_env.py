import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
from players import *
from games import *
from game_policies import *
from game_solvers import *


class RandomGameEnvironment(object):
    def __init__(self, game, duels, label='Stochastic Matrix Game (Without Expert Games)'):
        self.game = game    # All agents play the same game
        # a duel is a pair of agents whose performance against each other is measured
        self.duels = duels
        # duel[0]: defender
        # duel[1]: attacker
        self.label = label  # used for figure title

    def reset(self):
        self.game.reset()
        for duel in self.duels:
            duel[0].reset()
            duel[1].reset()

    def run(self, episodes=100, timesteps=1000):
        timesteps = self.defender.T
        rewards = np.zeros((len(self.duels), episodes * timesteps))
        values = np.zeros_like(rewards)
        expected_rewards = np.zeros_like(rewards)
        for k in range(episodes):
            ep_value = self.game.sp_value
            for t in range(timesteps):
                for idx, duel in enumerate(self.duels):
                    defe = duel[0].choose()
                    atta = duel[1].choose()
                    r = self.game.play(defe, atta)
                    duel[0].observe(r, atta)
                    duel[1].observe(-r, defe)
                    rewards[idx, k*timesteps + t] = r
                    values[idx, k*timesteps + t] = ep_value
                    expected_rewards[idx, k*timesteps +
                                     t] = np.dot(duel[0].policy.mu.T, self.game.true_game @ duel[1].policy.mu)

        return rewards, values, expected_rewards

    def plot_results(self, rewards, values):
        sns.set_style('white')
        sns.set_context('talk')
        plt.subplot(2, 1, 1)
        plt.title(self.label)
        plt.plot(rewards)
        plt.ylabel('Average Reward')
        #plt.legend(self.agents, loc=4)
        plt.subplot(2, 1, 2)
        plt.plot(optimal)
        plt.ylim(0, 100)
        plt.ylabel('% Optimal Action')
        plt.xlabel('Time Step')
        plt.legend(self.agents, loc=4)
        sns.despine()
        plt.show()

    def plot_regret(self, cumuregret):
        sns.set_style('white')
        sns.set_context('talk')


class ContextualGameEnvironment(object):
    def __init__(self, game, duels, label='Stochastic Matrix Game with Expert Games'):
        self.game = game
        self.duels = duels
        self.label = label

    def reset(self):
        self.game.reset()
        for duel in self.duels:
            duel[0].reset()
            duel[1].reset()

    def run(self, episodes=100, timesteps=1000):
        rewards = np.zeros(len(self.duels), episodes * timesteps)
        values = np.zeros_like(len(self.duels), episodes * timesteps)

        for k in range(episodes):
            for t in range(timesteps):
                for idx, duel in enumerate(self.duels):
                    defe = duel[0].choose()
                    atta = duel[1].choose()
                    r = self.game.play(defe, atta)
                    rewards[idx, k*timesteps + t] = r


if __name__ == "__main__":
    RG = RandomGame(10, 10, 1000)
    policy1 = UCBMinimax()
    policy2 = EXP3()
    defender = Player(RG, policy1)
    attacker = AdversarialPlayer(RG, policy2, IsAttacker=True)
    rewards = np.zeros(RG.T)
    values = np.zeros_like(rewards)
    expected_rewards = np.zeros_like(rewards)
    values = values + RG.sp_value
    regret = np.zeros_like(rewards)
    #print('the saddle point is {}'.format(RG.sp_value))
    #print('the true game is {}'.format(RG.true_game))
    for t in range(RG.T):
        # if t % 100 == 0:
        #    print('Timestep {} the defender game estimates {}'.format(t, defender.game_estimates))
        #    print('the true game is {}'.format(RG.true_game))
        defe = defender.choose()
        atta = attacker.choose()
        r = RG.play(defe, atta)
        defender.observe(r, atta)
        attacker.observe(-r, defe)
        rewards[t] += r
        expected_rewards[t] = np.dot(
            defender.policy.mu.T, RG.true_game @ attacker.policy.mu)

    cumuregret = np.cumsum(values - expected_rewards)
    print(cumuregret)
