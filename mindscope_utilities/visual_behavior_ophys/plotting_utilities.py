import matplotlib.pyplot as plt
import seaborn as sns
from mindscope_utilities.visual_behavior.ophys import calculate_response_matrix, calculate_d_prime_matrix

def plot_response_matrix(stimuli, ax=None, cmap='viridis'):
    
    return_fig_ax = False
    if ax is None:
        return return_fig_ax = True
        fig, ax = plt.subplots()

    response_matrix = make_response_matrix(stimuli)

    sns.heatmap(
        response_matrix,
        ax=ax[0, ii],
        cmap='viridis',
        vmin=0,
        vmax=1,
#         cbar=cbar,
#         annot=annotate, 
#         annot_kws=akws,
#         fmt=annot_format,
        square=True
    )
