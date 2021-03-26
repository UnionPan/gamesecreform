import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
from players import *
from games import *
from game_policies import *
from game_solvers import *


class RandomGameEnvironment(object):
    def __init__(self, game, defenders, attacler, label='Stochastic Matrix Game (Without Expert Games)'):
        self.game = game    # All agents play the same game
        # a duel is a pair of agents whose performance against each other is measured
        self.defenders = defenders
        # defender: defender
        # self.attacker: attacker
        self.attacker = attacker
        self.label = label  # used for figure title

    def reset(self):
        self.game.reset()
        for defender in self.defenders:
            defender.reset()
            self.attacker.reset()

    def run(self, episodes=100, timesteps=1000):
        timesteps = self.defenders[0].T
        rewards = np.zeros((len(self.duels), episodes * timesteps))
        values = np.zeros_like(rewards)
        expected_rewards = np.zeros_like(rewards)
        for k in range(episodes):
            ep_value = self.game.sp_value
            for t in range(timesteps):
                for idx, defender in enumerate(self.defenders):
                    defe = defender.choose()
                    atta = self.attacker.choose()
                    r = self.game.play(defe, atta)
                    defender.observe(r, atta)
                    self.attacker.observe(-r, defe)
                    rewards[idx, k*timesteps + t] = r
                    values[idx, k*timesteps + t] = ep_value
                    expected_rewards[idx, k*timesteps + t] = np.dot(
                        defender.policy.mu.T, self.game.true_game @ self.attacker.policy.mu)
                self.game.update_t()

        return rewards, values, expected_rewards

    def plot_results(self, rewards):
        sns.set_style('white')
        sns.set_context('talk')
        plt.subplot(2, 1, 1)
        plt.title(self.label)
        plt.plot(rewards)
        plt.ylabel('Average Reward')
        #plt.legend(self.agents, loc=4)
        plt.subplot(2, 1, 2)
        plt.plot(kldiv)
        plt.ylim(0, 100)
        plt.ylabel('% Optimal Action')
        plt.xlabel('Time Step')
        plt.legend(self.defenders, loc=4)
        sns.despine()
        plt.show()

    def plot_regret(self, cumuregret):
        sns.set_style('white')
        sns.set_context('talk')
        plt.title(self.label)
        plt.plot(cumuregret)
        plt.ylabel('Expected Reward')
        plt.xlabel('Time Step')
        plt.legend(self.defenders, loc=4)
        sns.despine()
        plt.show()


class ContextualGameEnvironment(object):
    def __init__(self, game, duels, label='Stochastic Matrix Game with Expert Games'):
        self.game = game
        self.duels = duels
        self.label = label

    def reset(self):
        self.game.reset()
        for duel in self.duels:
            defender.reset()
            self.attacker.reset()

    def run(self, episodes=100, timesteps=1000):
        timesteps = self.duels[0][0].T
        rewards = np.zeros((len(self.duels), episodes * timesteps))
        values = np.zeros_like(rewards)
        expected_rewards = np.zeros_like(rewards)
        for k in range(episodes):
            ep_value = self.game.sp_value
            for t in range(timesteps):
                for idx, duel in enumerate(self.duels):
                    defe = defender.choose()
                    atta = self.attacker.choose()
                    r = self.game.play(defe, atta)
                    defender.observe(r, atta)
                    self.attacker.observe(-r, defe)
                    rewards[idx, k*timesteps + t] = r
                    values[idx, k*timesteps + t] = ep_value[0]
                    expected_rewards[idx, k*timesteps + t] = np.dot(
                        defender.policy.mu.T, self.game.true_game @ self.attacker.policy.mu)
                self.game.update_t()

        return rewards, values, expected_rewards

    def plot_results(self, rewards, expected_rewards):
        sns.set_style('white')
        sns.set_context('talk')
        plt.subplot(2, 1, 1)
        plt.title(self.label)
        plt.plot(expected_rewards)
        plt.ylabel('Average Reward')
        #plt.legend(self.agents, loc=4)
        plt.subplot(2, 1, 2)
        plt.plot(rewards)
        plt.ylim(0, 100)
        plt.ylabel('% Actual Reward')
        plt.xlabel('Time Step')
        plt.legend(self.defenders, loc=4)
        sns.despine()
        plt.show()

    def plot_regret(self, values, rewards, expected_rewards):
        sns.set_style('white')
        sns.set_context('talk')
        plt.title(self.label)
        cumuregret = values - expected_rewards
        plt.plot(cumuregret)
        plt.ylabel('Absolute Regret')
        plt.xlabel('Time Step')
        plt.legend(self.defenders, loc=4)
        sns.despine()
        plt.show()


if __name__ == "__main__":
    """
    This experiment compares regret performance of different players in different game environment
    """
    EP, T, n1, n2, S = 15, 200, 10, 10, 10
    theta = np.random.normal(0.5, 1, size=S)
    
    # Define a random game and a contextual game
    RG, CG = RandomGame(n1, n2, T), ContextualGame(n1, n2, theta, T)

    # Define random game policies
    expminimax, minimax, ucbminimax, exp3, randplay = ExploringMinimax(), \
        GreedyMinimax(), UCBMinimax(), EXP3(), RandomPolicy()

    # Define contextual game policies
    ofulinmat, linmat, cexp3 = OFULinMat(), LinMat(), EXP3()

    # Define players for random game
    expmmplayer, mmplayer, ucbmmplayer, advplayer, randplayer = Player(RG, expminimax), Player(RG, minimax), \
        Player(RG, ucbminimax), Player(RG, exp3), Player(RG, randplay)

    # Define players for contextual game
    ofuplayer, linplayer, exp3player = ContextualPlayer(CG, ofulinmat, EP), ContextualPlayer(CG, linmat, EP),\
        AdversarialPlayer(CG, cexp3)

    # Define an omniscient attacker
    attacker = OmniAttacker(CG)

    rewards = np.zeros((2, EP*T))
    values, expected_rewards, regret = np.zeros_like(
        rewards), np.zeros_like(rewards), np.zeros_like(rewards)
    #print('the saddle point is {}'.format(RG.sp_value))
    #print('the true game is {}'.format(RG.true_game))
    dis = []
    for k in range(EP):
        dis.append(np.linalg.norm(theta - ofuplayer.theta_hat))
        print(np.linalg.norm(theta - ofuplayer.theta_hat))
        for t in range(T):
            attacker.observe()
            values[0, k * T + t] += CG.sp_value[0]
            values[1, k * T + t] += CG.sp_value[0]

            a0 = ofuplayer.choose()
            o0 = attacker.choose()
            r0 = CG.play(a0, o0)
            
            a1 = exp3player.choose()
            o1 = attacker.choose()
            r1 = CG.play(a1, o1)

            rewards[0, k * T + t] = r0
            rewards[1, k * T + t] = r1

            expected_rewards[0, k * T + t] = np.dot(
                ofuplayer.policy.mu.T, CG.true_game @ attacker.mu)
            expected_rewards[1, k * T + t] = np.dot(
                exp3player.policy.mu.T, CG.true_game @ attacker.mu)
            #print(exp3player.policy.mu)
            CG.update_t()

            #print('at episode {} attacker strategy {}'.format(k, attacker.mu))

            ofuplayer.observe(r0, o0)
            exp3player.observe(r1, o1)
            #attacker.observe()
            
            #print('the K time for exp3player, ofuplayer and game are {} {} {}'.format(
            #    exp3player.k, ofuplayer.k, CG.k))
            #print('the T time for exp3player, ofuplayer and game are {} {} {}'.format(exp3player.t, ofuplayer.t, CG.t))
            
    
    regret = values[0] - expected_rewards[0]
    regret1 = values[1] - expected_rewards[1]
    cumuregret = np.cumsum(regret)
    cumuregret1 = np.cumsum(regret1)

    figureregret = plt.figure(figsize=(10, 3))
    ax = figureregret.add_subplot(121)
    ax2 = figureregret.add_subplot(122)
    ax.set_title('Algortihm comparison')
    ax.plot(cumuregret, label=r'OFULinMat', color = 'red')
    ax.plot(cumuregret1, label=r'EXP3', color = 'green')
    ax.set_ylabel('cumulative regret')
    ax.set_xlabel('Time Step')
    ax.set_facecolor(color='#d6d6c2')
    ax.grid(True)
    ax.legend()
    plt.title(r'Distance between $\hat{\theta}$ and $\theta^*$')
    ax2.plot(dis, label = r'$\Vert \hat{\theta} - \theta^*\Vert_2$', color = 'blue', marker = 'v', markerfacecolor = 'red')
    ax2.set_ylabel(r'$l_2$ norm')
    ax2.set_xlabel('episodes')
    ax2.set_facecolor(color='#d6d6c2')
    ax2.legend()
    ax2.grid(True)
    plt.show()


