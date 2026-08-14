from typing import Any, List, Tuple
import math
import torch
from torch.utils.data import TensorDataset, DataLoader
import random
from matplotlib import pyplot as plt

from data_utils import load_data, create_training_dataset, collate_function
from visualisation import visualise_training_dataset, visualise_energy_landscape, \
    visualise_samples, visualise_optimisation_stats, visualise_gradient_field
from energy import EnergyModel, EnergyGradientModel

def sample(
        **kwargs: Any
) -> List[torch.Tensor]:
    """
    Function which samples from the distribution induced by the current EBM.

    NOTE:
    ----
        > Passing arguments in this case, works with keywords, i.e.
            sample(gradient_func=..., gamma=...)
        > Accessing the parameters inside the functions works as follows
            gradient_func = kwargs['gradient_func'],
            gamma = kwargs['gamma']

    """
    sample_list = []


    return sample_list

# ### #########################################################################
# ### CONTRASTIVE DIVERGENCE
# ######################################################################### ###

def max_likelihood_loss(
        model: EnergyModel,
        x: torch.Tensor,
        x_0: torch.Tensor,
        **sampling_kwargs: Any
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Implementation of the loss function

    :param model: PyTorch module representing the energy
    :param x: Current training batch
    :param x_0: Current initial guess for sampling
    :param sampling_kwargs: keyword arguments for sampling method
    :return: Tuple of loss and (final) sample
    """



    return loss, x_hat

def contrastive_divergence(
        model: EnergyModel,
        dataset: TensorDataset,
        max_num_iterations: int,
        batch_size: int,
        **sampling_kwargs: Any
) -> List[torch.Tensor]:
    """
    Main training routine for the training of an EBM using the maximum likelihood approach.

    NOTES
    -----
        > The initial guess for the sample generating function does not need to be generated each time
            by means of torch.rand(), or torch.randn()
        > It is convenient to use previous iterates as initial guesses.

    :param model: PyTorch module representing the energy function of the EBM
    :param dataset: Training dataset
    :param max_num_iterations: Maximal number of iterations
    :param batch_size: Batch size
    :param sampling_kwargs: Keyword arguments for sampling scheme
    :return: List of training losses
    """
    x_0 = torch.randn(batch_size, 2)
    x_0 = x_0.to(device=next(model.parameters()).device, dtype=next(model.parameters()).dtype)

    optimiser = 'TODO: Fill me'
    loss_list = []

    data_loader = DataLoader(dataset, batch_size=batch_size, collate_fn=collate_function)

    k = 0
    stop = False
    while not stop:
        for batch in data_loader:

            # Training loop

            if (k + 1) == max_num_iterations:
                print('reached maximal number of iterations')
                stop = True
                break
            else:
                k += 1

    return loss_list

def learn_ebm_contrastive_divergence(
        dataset,
        dtype: torch.dtype,
        device: torch.device
) -> Tuple[EnergyModel, List[torch.Tensor], List[torch.Tensor]]:
    energy_model = EnergyModel(
        num_hidden_units='TODO: Fill me',
        num_hidden_neurons='TODO: Fill me',
        activation_func='TODO: Fill me')
    energy_model.to(dtype=dtype, device=device)

    sampling_kwargs = {'TODO': 'Fill me'}

    max_num_iterations = 'TODO: Fill me'
    batch_size = 'TODO: Fill me'
    loss_list = contrastive_divergence(
        energy_model,
        dataset,
        max_num_iterations,
        batch_size=batch_size,
        **sampling_kwargs)

    sample_list = sample(**sampling_kwargs)

    return energy_model, loss_list, sample_list

# ### #########################################################################
# ### DENOISING SCORE MATCHING
# ######################################################################### ###

def denoising_score_matching_loss(
        model: EnergyGradientModel,
        x: torch.Tensor,
        sigma: float=0.05
) -> torch.Tensor:
    """
    Function implementing the denoising score matching loss

    :param model: PyTorch module representing the gradient of the energy function
    :param x: PyTorch tensor representing the current training batch
    :param sigma: Noise level
    :return: Denoising score matching loss
    """
    pass

def denoising_score_matching(
        model: EnergyGradientModel,
        dataset: TensorDataset,
        max_num_iterations: int,
        batch_size: int
) -> List[torch.Tensor]:
    """
    Function implementing the training loop of denoising score matching.

    :param model: PyTorch module representing the gradient of the energy to be trained
    :param dataset: Training dataset
    :param max_num_iterations: Maximal number of iterations to be performed
    :param batch_size: Size of training batches
    :return: List of training losses
    """
    optimiser = 'TODO: Fill me'
    loss_list = []

    data_loader = DataLoader(dataset, batch_size=batch_size, collate_fn=collate_function)
    k = 0
    stop = False
    while not stop:
        for batch in data_loader:

            # Training loop


            if (k + 1) == max_num_iterations:
                print('reached maximal number of iterations')
                stop = True
                break
            else:
                k += 1

    return loss_list

def learn_ebm_denoising_score_matching(
        dataset,
        dtype: torch.dtype,
        device: torch.device

) -> Tuple[EnergyGradientModel, List[torch.Tensor], List[torch.Tensor]]:
    model = EnergyGradientModel(
        activation_func='TODO: Fill me',
        num_hidden_neurons='TODO: Fill me',
        num_hidden_units='TODO: Fill me')
    model.to(dtype=dtype, device=device)

    max_num_iterations = 'TODO: Fill me'
    batch_size = 'TODO: Fill me'
    loss_list = denoising_score_matching(
        model,
        dataset,
        max_num_iterations,
        batch_size=batch_size)

    sampling_kwargs = {'TODO': 'Fill me'}
    sample_list = sample(**sampling_kwargs)

    return model, loss_list, sample_list

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dtype = torch.float32

    method = 'contrastive_divergence' # 'denoising_score_matching'

    # --- define dataset
    num_data_samples = 2 ** 14
    data = load_data(
        num_data_samples,
        'swiss_roll' if method == 'contrastive_divergence' else 'rings')
    dataset = create_training_dataset(data, device, dtype)

    fig_dataset = visualise_training_dataset(dataset)

    sample_list = []
    loss_list = []
    if method == 'contrastive_divergence':
        energy_model, loss_list, sample_list = learn_ebm_contrastive_divergence(
            dataset,
            dtype,
            device)

        a = 3.0
        fig_energy = visualise_energy_landscape(
            energy_model,
            x_box_low=-a,
            x_box_high=a,
            y_box_low=-a,
            y_box_high=a,
            num_samples=200,
            dtype=dtype)
    elif method == 'denoising_score_matching':
        energy_gradient_model, loss_list, sample_list = learn_ebm_denoising_score_matching(
            dataset,
            dtype,
            device)

        a = 1.5
        fig_gradient = visualise_gradient_field(
            energy_gradient_model,
            x_box_low=-a,
            x_box_high=a,
            y_box_low=-a,
            y_box_high=a,
            num_samples=100,
            dtype=dtype)
    else:
        raise ValueError('Unknown learning method.')

    fig_samples = visualise_samples(torch.cat([sample_list[-1]]))
    fig_optim_stats = visualise_optimisation_stats([l.cpu().item() for l in loss_list])

    plt.show()

if __name__ == '__main__':
    torch.manual_seed(123)
    random.seed(123)
    main()
