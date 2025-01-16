import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from simpledbf import Dbf5
import os
import optuna
from mpl_toolkits.mplot3d import Axes3D
import plotly.express as px
import seaborn as sns
import plotly.io as pio

import warnings

# Suppress the specific warning
warnings.filterwarnings(
    "ignore",
    message="Choices for a categorical distribution should be a tuple of None, bool, int, float and str for persistent storage but contains",
    category=UserWarning,
    module="optuna.distributions",
)

pd.set_option('display.max_rows', 100)      # Number of rows to display
pd.set_option('display.max_columns', None) # Ensure all columns are displayed
pd.set_option('display.width', 1000)       # Adjust the total width for the display
pd.set_option('display.colheader_justify', 'center')


# Make sure to provide only the study name, not a path
study = optuna.load_study(
    study_name='new_NN-study3',  # The study name, not the path
    storage='sqlite:///Studies/new_NN-study3.db'  # Path to the database
)

# Define the explicit mapping from layer sizes to integers

param_names = [name for name in study.trials[0].params.keys() if name != 'layer_sizes']

# Print the parameter names
print("Other parameter names:", param_names)
layer_sizes_mapping = {
    tuple([0]): 1,  # linear
    tuple([100]): 2,  # tiny
    tuple([100, 50]): 3,  # small
    tuple([200, 100]): 4,  # medium
    tuple([300, 200, 100]): 5,  # large
    tuple([512, 256, 64]): 6,  # verylarge
    tuple([1000, 800, 512]): 7  # huge
}

data = []
for trial in study.trials:
    if trial.state == optuna.trial.TrialState.COMPLETE:
        # Map the layer_sizes list to the corresponding integer
        layer_sizes_key = tuple(trial.params['layer_sizes'])
        layer_sizes_int = layer_sizes_mapping.get(layer_sizes_key, 0)

        # Extract other parameters
        trial_params = trial.params.copy()
        trial_params.pop('layer_sizes', None)  # Remove 'layer_sizes' since we're using the mapped integer

        # Create a data row with trial number, all parameters, and the objective value
        data_row = {'trial_number': trial.number, 'layer_sizes_int': layer_sizes_int, 'value': trial.value}
        data_row.update(trial_params)

        # Append the row to the data list
        data.append(data_row)

# Convert the list of data to a DataFrame
df = pd.DataFrame(data)
print(df)









opt_hist = optuna.visualization.plot_optimization_history(study)
opt_importance = optuna.visualization.plot_param_importances(study)
opt_slice = optuna.visualization.plot_slice(study)
opt_contour = optuna.visualization.plot_contour(study)
opt_intermediate = optuna.visualization.plot_intermediate_values(study)
opt_edf = optuna.visualization.plot_edf([study])
# opt_hypervol = optuna.visualization.plot_hypervolume_history(study)
#opt_pa_co = optuna.visualization.plot_parallel_coordinate(study)
#opt_pareto = optuna.visualization.plot_pareto_front(study)
opt_rank = optuna.visualization.plot_rank(study)
opt_time = optuna.visualization.plot_timeline(study)
#opt_term = optuna.visualization.plot_terminator_improvement(study)

# Display plots
pio.show(opt_hist)
pio.show(opt_importance)
pio.show(opt_slice)
pio.show(opt_contour)
pio.show(opt_intermediate)
pio.show(opt_edf)
# pio.show(opt_hypervol)
#pio.show(opt_pa_co)
#pio.show(opt_pareto)
pio.show(opt_rank)
pio.show(opt_time)
#pio.show(opt_term)

