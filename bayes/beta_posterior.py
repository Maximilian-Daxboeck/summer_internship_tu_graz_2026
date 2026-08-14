import numpy as np
from matplotlib import pyplot as plt

def compute_posterior_log_pdf(
        theta: np.ndarray,
        num_successes: float,
        num_trials: int,
        alpha: float,
        beta: float
) -> np.ndarray:

    return (num_successes + alpha - 1) * np.log(theta) + (num_trials - num_successes + beta - 1) * np.log(1 - theta)

def main():
    fig = plt.figure()
    ax = fig.add_subplot()

    alpha = 2
    beta = 2
    theta = np.linspace(1e-6, 1-1e-6, 567)
    succ_rate = 0.75


    sample_size_list = [100, 1000, 10000]
    for n in sample_size_list:
        num_successes = np.floor(succ_rate * n)
        y = compute_posterior_log_pdf(theta, num_successes, n, alpha, beta)
        y = y - np.max(y)
        ax.plot(theta, np.exp(y), label='n = {:d}'.format(n))

        mle_estimate = num_successes / n
        map_estimate = (alpha + num_successes - 1) / (alpha + beta + n - 2)
        print('MLE: {:.5f}, MAP: {:.5f}'.format(mle_estimate, map_estimate))

    ax.legend()
    plt.show()


if __name__ == '__main__':
    main()


