from matplotlib import pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from typing import Tuple, List, Dict, Any

BETA_0_LOW = -2.5
BETA_0_HIGH = 2.5
BETA_1_LOW = -2.5
BETA_1_HIGH = 2.5

DATA_X_LOW = -1.5
DATA_X_HIGH = 1.5
DATA_Y_LOW = -1.5
DATA_Y_HIGH = 1.5

# --- dataset generation

def generate_dataset(
        beta_0_true: float,
        beta_1_true: float,
        sig_sq_true: float,
        size: int
) -> Tuple[np.ndarray, ...]:
    x = np.linspace(-1, 1, size)
    y = beta_0_true + beta_1_true * x + np.sqrt(sig_sq_true) * np.random.randn(size)
    return x, y

# --- plotting and simulation

def plot_bayesian_update(
    ax_0: plt.Axes,
    ax_1: plt.Axes,
    xx: np.ndarray,
    y: np.ndarray,
    beta_samples: np.ndarray,
    sig_sq_samples: np.ndarray,
    n_obs: int
) -> None:
    u = np.linspace(BETA_0_LOW, BETA_0_HIGH, 123)
    v = np.linspace(BETA_1_LOW, BETA_1_HIGH, 123)
    uu, vv = np.meshgrid(u, v)

    sig_sq = np.mean(sig_sq_samples)
    log_post = np.array([
        log_beta_posterior_density(
            z,
            xx[:n_obs],
            y[:n_obs],
            sig_sq)
        for z in np.stack([uu.ravel(), vv.ravel()], axis=1)
    ])

    # --- posterior density
    ax_0.contourf(u, v, np.exp(log_post - np.max(log_post)).reshape(uu.shape), levels=20, cmap='viridis')
    ax_0.set_title('posterior density')
    ax_0.set_xlabel(r'$\beta_0$')
    ax_0.set_ylabel(r'$\beta_1$')
    ax_0.set_xlim(BETA_0_LOW, BETA_0_HIGH)
    ax_0.set_ylim(BETA_1_LOW, BETA_1_HIGH)

    # --- plot data and regression lines
    ax_1.scatter(xx[:n_obs, 1], y[:n_obs], c='black', label='data', s=0.7)

    t = np.linspace(-1, 1, 10)
    for i in np.random.choice(len(beta_samples), 10):
        b0, b1 = beta_samples[i]
        ax_1.plot(t, b0 + b1 * t, color='red', alpha=0.3)

    ax_1.set_title('data and regression lines')
    ax_1.set_xlim(DATA_X_LOW, DATA_X_HIGH)
    ax_1.set_ylim(DATA_Y_LOW, DATA_Y_HIGH)

def simulate_sequential_bayesian_update(
        x: np.ndarray,
        y: np.ndarray,
        prior_params: Dict[str, Any],
        stages: List[int]
) -> None:
    fig = plt.figure(figsize=(16, 8))
    spec = gridspec.GridSpec(nrows=2, ncols=len(stages), figure=fig)

    for idx, n in enumerate(stages):
        if n == 0:
            ax_0 = fig.add_subplot(spec[0, idx])
            u = np.linspace(BETA_0_LOW, BETA_0_HIGH, 123)
            v = np.linspace(BETA_1_LOW, BETA_1_HIGH, 123)
            uu, vv = np.meshgrid(u, v)

            log_prior = np.array([
                log_beta_prior_density(
                    z,
                    sig_sq=prior_params['normal']['sig_sq'])
                for z in np.stack([uu.ravel(), vv.ravel()], axis=1)])

            ax_0.contourf(u, v, np.exp(log_prior - np.max(log_prior)).reshape(uu.shape), levels=20, cmap='viridis')
            ax_0.set_title('prior density')
            ax_0.set_xlabel(r'$\beta_0$')
            ax_0.set_ylabel(r'$\beta_1$')
            ax_0.set_xlim(BETA_0_LOW, BETA_0_HIGH)
            ax_0.set_ylim(BETA_1_LOW, BETA_1_HIGH)
            ax_0.set_aspect('equal')
        else:
            ax_0 = fig.add_subplot(spec[0, idx])
            ax_1 = fig.add_subplot(spec[1, idx])

            x_subset = x[:n]
            y_subset = y[:n]
            beta_arr, sig_sq_arr = gibbs_sampler(
                x_subset,
                y_subset,
                mu_0=prior_params['normal']['mu_0'],
                sig_sq=prior_params['normal']['sig_sq'],
                cov_0=prior_params['normal']['cov_0'],
                a0=prior_params['gamma']['a0'],
                b0=prior_params['gamma']['b0'],
                num_samples=1000,
                burn_in=700)
            plot_bayesian_update(ax_0, ax_1, x, y, beta_arr, sig_sq_arr, n)

            ax_0.set_aspect('equal')
            ax_1.set_aspect('equal')

            print('posterior mean (n = {:d})'.format(n))
            print(' > beta_0: {:.5f}'.format(np.mean(beta_arr[:, 0])))
            print(' > beta_1: {:.5f}'.format(np.mean(beta_arr[:, 1])))
            print(' > sig_sq: {:.5f}'.format(np.mean(sig_sq_arr)))

    plt.tight_layout()
    plt.show()

def log_beta_prior_density( # µ_0=[0,...,0] und cov_0=id
    u: np.ndarray,
    sig_sq: float
) -> np.ndarray:

    density = 1/sig_sq**(u.size/2) * np.exp(-0.5/sig_sq * np.dot(u,u)) # proportional zur Dichte

    return np.log(density)

def log_beta_posterior_density(# µ_0=[0,0] und cov_0=[[1,0],[0,1]]                   #! beta mit gegebenem sig_sq
    u: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    sig_sq: float
) -> np.ndarray:

    other_matrix = np.linalg.inv(np.matmul(x.transpose(1, 0), x) + np.eye(x.shape[1]))

    mu_n = np.matmul(other_matrix, np.matmul(x.transpose(1, 0), y))
    cov_n = sig_sq * other_matrix

    density = 1/sig_sq**(u.size/2) * np.exp(-0.5/sig_sq * np.dot(np.dot(u-mu_n, np.linalg.inv(cov_n)), u-mu_n))

    return np.log(density)

def sample_box_muller(                                                          # done
        mu: np.ndarray,
        cov: np.ndarray
) -> np.ndarray:
    return np.random.multivariate_normal(mu, cov)

def sample_beta(                                  # fertig
    x: np.ndarray,
    y: np.ndarray,
    sig_sq: float,
    mu_0: np.ndarray,
    cov_0: np.ndarray
) -> np.ndarray:

    inv_cov_0 = np.linalg.inv(cov_0)
    other_matrix = np.linalg.inv(np.matmul(x.transpose(1,0), x) + inv_cov_0)

    mu = np.matmul(other_matrix, np.matmul(x.transpose(1,0), y) + np.matmul(inv_cov_0, mu_0))
    cov = sig_sq * other_matrix

    return sample_box_muller(mu, cov)

def sample_sig_sq(                                                              # done
    a: float,
    b: float
) -> np.ndarray:
    return 1.0 / np.random.gamma(a, 1.0 / b)

def gibbs_sampler(                                                              # done
    x: np.ndarray,
    y: np.ndarray,
    mu_0: np.ndarray,
    cov_0: np.ndarray,
    sig_sq: float,
    a0: float,
    b0: float,
    num_samples: int,
    burn_in: int
) -> Tuple[np.ndarray, ...]:


    beta_samples = []
    sig_sq_samples = []


    for k in range(0, num_samples + burn_in):
        beta_samples += [sample_beta(x,y,sig_sq,mu_0,cov_0)]

        a_n = a0 + (cov_0.shape[0] + y.size)/2

        diff_1 = y - np.matmul(x,beta_samples[-1])
        diff_2 = beta_samples[-1] - mu_0
        b_n = b0 + 0.5 * (np.dot(diff_1, diff_1) + np.dot(np.dot(diff_2, np.linalg.inv(cov_0)), diff_2))
        sig_sq = sample_sig_sq(a_n, b_n)
        sig_sq_samples += [sig_sq]

    return np.asarray(beta_samples[burn_in:]), np.asarray(sig_sq_samples[burn_in:])

def main():
    dataset_size = 100
    x, y = generate_dataset(
        beta_0_true=-0.7,
        beta_1_true=0.5,
        sig_sq_true=0.01,
        size=dataset_size)
    xx = np.column_stack([np.ones(dataset_size), x])

    prior_params = {
        'gamma': {'a0': 1.0, 'b0': 1.0},
        'normal': {'mu_0': np.zeros(2), 'sig_sq': 1.0, 'cov_0': np.eye(xx.shape[1])}
    }
    stages = [0, 2, 5, 20, dataset_size]
    simulate_sequential_bayesian_update(
        xx,
        y,
        prior_params,
        stages)

if __name__ == '__main__':
    np.random.seed(123)
    main()

