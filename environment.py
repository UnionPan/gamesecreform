import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import scipy.stats as stats




class Environment(object):
    def __init__(self, game, defende, attacker, label='Stochastic Matrix Game'):
        self.game = game
        self.defender = defender
        self.attacker = attackers
        self.label = label

    def reset(self):
        self.game.reset()
        for agent in self.agents:
            agent.reset()

    def run(self, episodes=100, timesteps=20):
        rewards = np.zeros((episodes * timesteps)
         # zero sum rewards across the episodes
        values = np.zeros_like(rewards)
        for n in range(episodes):
            self.reset()
            for t in range(timesteps):
                it = defender.choose()
                jt = attacker.choose()
                reward = self.game.play(it, jt)
                defender.observe(reward, jt)

        return rewards, values, defender.theta

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

        
