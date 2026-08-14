import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from typing import Tuple

def sample_categorical(
        mu: np.ndarray,
        num_samples: int
) -> np.ndarray:

    pass

def markov_chain() -> Tuple[np.ndarray, ...]:
    """
    Function, returning the initial distribution mu and
    transition matrix P of a (mu, P) markov chain.
    """

    pass

def simulate_markov_chain(
        mu: np.ndarray,
        p: np.ndarray,
        path_len: int,
        num_chains: int=1
) -> np.ndarray:
    pass

def make_histogram(
        data: np.ndarray,
        num_classes: int
) -> plt.Figure:
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    pass

def visualise_paths(chains: np.ndarray) -> plt.Figure:

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)

    ax.set_xlabel('time')
    ax.set_ylabel('state')

    pass

def main():
    # --- categorical sampling
    mu = np.array(['TODO: Fill me'])
    samples = sample_categorical(mu, 10000)
    fig_1 = make_histogram(samples, num_classes=len(mu))

    # --- simulate Markov chain
    mu, p = markov_chain()
    chains = simulate_markov_chain(mu, p, path_len=10, num_chains=100)
    fig_2 = visualise_paths(chains)

    plt.show()


if __name__ == '__main__':
    sns.set(style='darkgrid')
    main()