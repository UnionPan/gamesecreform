import numpy as np
#from scipy.optimize import linprog
from cvxopt import matrix, solvers
import matplotlib.pyplot as plt
import nashpy as nash

# Nash Solver using cvxopt


def Nash_solver(M):
    '''
    Using cvxopt linear programming to solve zero sum matrix game
    take in the game matrix
    row player maximizes the payoff, column player minimizes it.
    return f,g and game value v
    '''
    n, m = np.shape(M)[0], np.shape(M)[1]

    # minimize matrix c
    c = [-1] + [0 for i in range(n)]
    c = np.array(c, dtype="float")
    c = matrix(c)
    # constraints G*x <= h
    G = np.matrix(M, dtype="float").T  # reformat each variable is in a row
    G *= -1  # minimization constraint
    G = np.vstack([G, np.eye(n) * -1])  # > 0 constraint for all vars
    new_col = [1 for i in range(m)] + [0 for i in range(n)]
    G = np.insert(G, 0, new_col, axis=1)  # insert utility column
    G = matrix(G)

    h = ([0 for i in range(m)] + [0 for i in range(n)])
    h = np.array(h, dtype="float")
    h = matrix(h)
    # contraints Ax = b
    A = [0] + [1 for i in range(n)]
    A = np.matrix(A, dtype="float")
    A = matrix(A)
    b = np.matrix(1, dtype="float")
    b = matrix(b)
    sol = solvers.lp(c=c, G=G, h=h, A=A, b=b, solver="glpk",
                     options={'glpk': {'msg_lev': 'GLP_MSG_OFF'}})

    c1 = [-1] + [0 for i in range(m)]
    c1 = np.array(c1, dtype="float")
    c1 = matrix(c1)
    # constraints G*x <= h
    G1 = np.matrix(M, dtype="float")  # reformat each variable is in a row
    # minimization constraint
    G1 = np.vstack([G1, np.eye(m) * -1])  # > 0 constraint for all vars
    new_col1 = [1 for i in range(n)] + [0 for i in range(m)]
    G1 = np.insert(G1, 0, new_col1, axis=1)  # insert utility column
    G1 = matrix(G1)

    h1 = ([0 for i in range(n)] + [0 for i in range(m)])
    h1 = np.array(h1, dtype="float")
    h1 = matrix(h1)
    # contraints Ax = b
    A1 = [0] + [1 for i in range(m)]
    A1 = np.matrix(A1, dtype="float")
    A1 = matrix(A1)

    sol1 = solvers.lp(c=c1, G=G1, h=h1, A=A1, b=b, solver="glpk", options={
                      'glpk': {'msg_lev': 'GLP_MSG_OFF'}})

    if sol["x"] == None:
        print(M)
    f, g, v = sol["x"][1:], sol1["x"][1:], sol["x"][:1]
    #f, g, v = sol["x"], sol["z"], sol["primal objective"]

    return f, g, v

# value solver using cvxopt


def value_of_matrix(M):
    '''
    Using Nashpy solver support enumerating algorithm
    return f,g and game value v
    '''
    n, m = np.shape(M)[0], np.shape(M)[1]
    # minimize matrix c
    c = [-1] + [0 for i in range(n)]
    c = np.array(c, dtype="float")
    c = matrix(c)

    # constraints G*x <= h
    G = np.matrix(M, dtype="float").T  # reformat each variable is in a row
    # print(G)
    G *= -1  # minimization constraint
    G = np.vstack([G, np.eye(n) * -1])  # > 0 constraint for all vars
    new_col = [1 for i in range(m)] + [0 for i in range(n)]
    G = np.insert(G, 0, new_col, axis=1)  # insert utility column

    G = matrix(G)

    h = ([0 for i in range(m)] + [0 for i in range(n)])
    h = np.array(h, dtype="float")
    h = matrix(h)

    # contraints Ax = b
    A = [0] + [1 for i in range(n)]
    A = np.matrix(A, dtype="float")

    A = matrix(A)

    b = np.matrix(1, dtype="float")
    b = matrix(b)

    sol = solvers.lp(c=c, G=G, h=h, A=A, b=b, solver="glpk",
                     options={'glpk': {'msg_lev': 'GLP_MSG_OFF'}})

    if sol["x"] == None:
        print(M)
    return sol["x"][0]


# Nash solver using nashpy
def Nash_solver1(M):
    'using Nashpy to solve a matrix game'
    zsgame = nash.Game(M)
    #print('the game matrix is {}'.format(M))
    n1, n2 = np.shape(M)[0], np.shape(M)[1]
    label = 0
    eq = zsgame.lemke_howson(initial_dropped_label=label)
    #f = np.ones(np.shape(M)[0])/ np.shape(M)[0]
    #g = np.empty(np.shape(M)[1])/ np.shape(M)[1]

    f, g = eq[0], eq[1]
    #print('the policy is {}'.format(f))
    if np.isnan(f).any() or np.isnan(g).any():
        f = np.random.random(n1)
        f /= np.sum(f)
        g = np.random.random(n2)
        g /= np.sum(g)
        return f, g, 0
    else:
        v = zsgame[f, g][0]
    return f, g, v


# value solver using nashpy
def value_of_matrix1(M):
    'using Nashpy'
    zsgame = nash.Game(M)
    label = 0
    eq = zsgame.lemke_howson(initial_dropped_label=label)
    #f = np.ones(np.shape(M)[0])/ np.shape(M)[0]
    #g = np.empty(np.shape(M)[1])/ np.shape(M)[1]
    #v = 0
    f, g = eq[0], eq[1]
    v = zsgame[f, g][0]
    return v


if __name__ == "__main__":
    for t in range(100):
        print(t)
        M = np.random.uniform(size=(10, 10))
        print(M)
        f, g, v = Nash_solver(M)
        print(f)
