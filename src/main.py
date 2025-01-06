from flask import Flask, request, jsonify, render_template
from modules.environment import WheelEnv
from modules.agent import DQNAgent
from models.models import NNModel
import numpy as np
import optuna
import tensorflow 
from keras.models import load_model
import sys
import logging
import plotly.io as pio
import os



linear = [0]
tiny = [100]
small = [100, 50]
medium = [200, 100]
large = [300, 200, 100]
verylarge = [512, 256, 64]

# Define the logging directory
PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))  # Path to the project folder
LOGGING_DIR = os.path.join(PROJECT_ROOT, "SessionLogs")
os.makedirs(LOGGING_DIR, exist_ok=True)

env = WheelEnv()
agent = DQNAgent(env,logging_dir=LOGGING_DIR, batch_size=32, learning_rate=0.00025, gamma=0, layer_sizes=[512, 256, 64],
                 Model_type="NN", use_per=True, memory_size=1000, max_episode_len=100,
                epsilon=0.9, epsilon_min=0.01,
                 epsilon_decay=0.0005, logging=True)

agent.run(30)

"""
def call_func_with_default_if_none(func, environment, params):
    params = {k: v for k, v in params.items() if v is not None}
    return func(environment, **params)

logging = True
counter = 0
filename = f"test_tree_{counter}.txt"

param_env = {
        'logging': logging,
        'render': True,
        'filename': filename
    }

param_agent = {
        'batch_size': 32,
        'learning_rate': 0.00025,
        'gamma': gamma,
        'layer_sizes': [512, 256, 100],
        'shift': False,
        'Model_type': "NN",
        'use_per': True,
        'memory_size': mem_size,
        'max_episode_len': 150,
        'epsilon': 1,
        'epsilon_min': 0.1,
        'epsilon_decay': 0.005,
        'logging': True,
        'filename': filename
    }

env = WheelEnv(**param_env)


agent.run(11)
#acc, eff = agent.evaluate(states, tensions, rewards)
#print(acc)
#print(eff)




sampler = optuna.samplers.TPESampler()
optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler(sys.stdout))
study_name = "Tree-study"  # Unique identifier of the study.
storage_name = "sqlite:///{}.db".format(study_name)
study = optuna.create_study(
    sampler=sampler,
    direction='maximize', study_name=study_name, storage=storage_name)
study.optimize(func=objective, n_trials=10)

opt_hist = optuna.visualization.plot_optimization_history(study)
opt_importance = optuna.visualization.plot_param_importances(study)
opt_slice = optuna.visualization.plot_slice(study)
opt_contour = optuna.visualization.plot_contour(study)

# Display plots
pio.show(opt_hist)
pio.show(opt_importance)
pio.show(opt_slice)
pio.show(opt_contour)
"""