import matplotlib.pyplot as plt
import seaborn as sns
from mindscope_utilities.visual_behavior_ophys import calculate_response_matrix, calculate_dprime_matrix

def plot_response_matrix(stimuli, ax=None, vmin=0, vmax=1, cmap='viridis', cbar_ax=None):
    '''
    makes a plot of the response matrix given a table of stimuli
    '''
    
    return_fig_ax = False
    if ax is None:
        return_fig_ax = True
        fig, ax = plt.subplots()

    response_matrix = calculate_response_matrix(stimuli)

    im = ax.imshow(
        response_matrix,
        cmap=cmap,
        aspect='equal',
        vmin=vmin,
        vmax=vmax,
    )
    ax.set_xticks(range(len(response_matrix)))
    ax.set_xticklabels(response_matrix.columns)
    ax.set_yticks(range(len(response_matrix)))
    ax.set_yticklabels(response_matrix.index)
    ax.set_xlabel('image name')
    ax.set_ylabel('previous image name')

    if cbar_ax is not None:
        plt.colorbar(im, cax=cbar_ax, label='response probability')

    if return_fig_ax:
        return fig, ax


def plot_dprime_matrix(stimuli, ax=None, vmin=0, vmax=1.5, cmap='magma', cbar_ax=None):
    '''
    makes a plot of the response matrix given a table of stimuli
    '''

    return_fig_ax = False
    if ax is None:
        return_fig_ax = True
        fig, ax = plt.subplots()

    dprime_matrix = calculate_dprime_matrix(stimuli)

    im = ax.imshow(
        dprime_matrix,
        cmap=cmap,
        aspect='equal',
        vmin=vmin,
        vmax=vmax,
    )
    ax.set_xticks(range(len(dprime_matrix)))
    ax.set_xticklabels(dprime_matrix.columns)
    ax.set_yticks(range(len(dprime_matrix)))
    ax.set_yticklabels(dprime_matrix.index)
    ax.set_xlabel('image name')
    ax.set_ylabel('previous image name')

    if cbar_ax is not None:
        plt.colorbar(im, cax=cbar_ax, label="d'")

    if return_fig_ax:
        return fig, ax