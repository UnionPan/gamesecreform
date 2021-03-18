from game_env import *
from game_policies import *
from game_solvers import *
from games import *
from players import *

RANDOM_SEED = 1204
np.random.RandomState(seed=RANDOM_SEED)


if __name__ == "__main__":
    K = 1000
    T = 1000
    n1, n2, S = 10, 10, 10
    theta = np.random.normal(0.5, 1, size=S)
    Cgame = ContextualGame(n1, n2, theta, T)

    duels = []

    c_strategy = OFULinMat()
    c_player = ContextualPlayer(Cgame, c_strategy, K)

    a_strategy = EXP3()
    a_player = AdversarialPlayer(Cgame, a_strategy)

    o_attacker = OmniAttacker(Cgame)

    duels.append([c_player, o_attacker])
    duels.append([a_player, o_attacker])

    Exp = ContextualGameEnvironment(Cgame, duels)
    rewards, values, expected_rewards = Exp.run(episodes=K)
    Exp.plot_regret(values, regret, expected_rewards)
