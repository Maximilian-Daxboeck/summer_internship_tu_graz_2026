from categorial_sampling import *
import numpy as np

def index_to_vec(index):
    vec = np.zeros(3)
    vec[index] = 1.
    return vec

def vec_to_index(vec):
    return np.argmax(vec)

def markov_step(x, P):
    p = np.matmul(P, x)
    s = sample(p,1)
    return index_to_vec(s)

def simulate_markov_chain(initial, steps, P):
    states = initial
    for _ in range(steps):
        for k in range(len(states)):
            states[k] = markov_step(states[k], P)

    return states


if __name__=='__main__':
    initial_distribution = np.array([1/3, 1/3, 1/3])

    P = np.array([[0.8, 0.4, 0.4],
                  [0.1, 0.4, 0.3],
                  [0.1, 0.2, 0.3]])

    initial_samples = [index_to_vec(i) for i in sample(initial_distribution, 100)]

    s = simulate_markov_chain(initial_samples, 100, P)
    s = [vec_to_index(sample) for sample in s]
    make_histogram(s)


