
import pandas as pd

import ast

import numpy as np

import re

import matplotlib.pyplot as plt







### make plots nicer, adjust heatmap to show up/down drive/nondrive for each spoke







# Load the CSV file

file_path = r"SessionLogs/big_model/session1.csv"  # Replace with the path to your CSV file

data = pd.read_csv(file_path)




# Convert `initialstate`, `predictions`, `actions`, and `rewards` columns from strings to lists

data['initialstate'] = data['initialstate'].apply(ast.literal_eval)

def preprocess_predictions(pred):

    # Handle None or empty entries

    if pred is None or pred == "None" or pred.strip() == "":

        return None

    if isinstance(pred, str):

        # Replace "array(...)" with "..." and ensure it is parsable

        pred = re.sub(r'array\((.+?)\)', r'[\1]', pred)  # Convert array(...) to [...]

        pred = re.sub(r'dtype=float32', '', pred)  # Remove dtype information

        pred = re.sub(r'\s+', ' ', pred)  # Normalize whitespace

        try:

            # Use ast.literal_eval to safely parse the string into a Python object

            return ast.literal_eval(pred.strip())

        except (ValueError, SyntaxError):

            return None  # Return None for unparsable data

    return pred 



data['predictions'] = data['predictions'].apply(preprocess_predictions)

data['actions'] = data['actions'].apply(ast.literal_eval)

data['rewards'] = data['rewards'].apply(ast.literal_eval)



# Example: Convert datetime to a proper datetime object

data['datetime'] = pd.to_datetime(data['datetime'])





# Example: Extract specific data for further analysis

initial_states = np.array(data['initialstate'].tolist())

actions = np.array(data['actions'].tolist())

rewards = np.array(data['rewards'].tolist())

predictions = data['predictions'].tolist()  # Leave as list since it contains mixed data types





from scipy.ndimage import uniform_filter1d  # For moving averages



# Example moving average function

def moving_average(data, window_size):

    return uniform_filter1d(data, size=window_size, mode='nearest')



# Assuming `rewards`, `predictions`, and `actions` are defined elsewhere

last_rewards = []

min_rewards = []

max_rewards = []

first_rewards = []



# Extract rewards data

for i in rewards:

    first_rewards.append(i[0])

    last_rewards.append(i[-1])

    min_rewards.append(np.min(i))

    max_rewards.append(np.max(i))



# Episode numbers

episode_numbers = np.arange(len(rewards))



# Compute moving averages for scatter data

window_size = 5  # You can adjust the window size

last_rewards_smooth = moving_average(last_rewards, window_size)

first_rewards_smooth = moving_average(first_rewards, window_size)

min_rewards_smooth = moving_average(min_rewards, window_size)

max_rewards_smooth = moving_average(max_rewards, window_size)



# Figure 1: Scatter plots with moving averages

plt.figure(figsize=(10, 6))

plt.scatter(episode_numbers, last_rewards, label='Last Rewards', alpha=0.6)

plt.plot(episode_numbers, last_rewards_smooth, label='Last Rewards (Moving Avg)', linestyle='--', color='blue')

plt.scatter(episode_numbers, min_rewards, label='Min Rewards', alpha=0.6)

plt.plot(episode_numbers, min_rewards_smooth, label='Min Rewards (Moving Avg)', linestyle='--', color='orange')

plt.scatter(episode_numbers, max_rewards, label='Max Rewards', alpha=0.6)

plt.plot(episode_numbers, max_rewards_smooth, label='Max Rewards (Moving Avg)', linestyle='--', color='green')

plt.scatter(episode_numbers, first_rewards, label='First Rewards', alpha=0.6)

plt.plot(episode_numbers, first_rewards_smooth, label='Max Rewards (Moving Avg)', linestyle='--', color='grey')

plt.xlabel('Episode')

plt.ylabel('Rewards')

plt.title('Rewards Scatter Plot with Moving Averages')

plt.legend()

plt.grid()

#plt.show()




num_actions = 72



# Initialize a matrix to store the frequency of each action (rows = episodes, columns = actions)

action_frequencies = np.zeros((len(actions), num_actions))



# Fill the frequency matrix by counting the occurrences of each action for each episode

for i, action_list in enumerate(actions):

    for action in action_list:

        action_frequencies[i, action] += 1



# Separate the even and odd actions

even_action_frequencies = action_frequencies[:, ::2]  # Even actions (columns 0, 2, 4, ..., 70)

odd_action_frequencies = action_frequencies[:, 1::2]   # Odd actions (columns 1, 3, 5, ..., 71)



# Create the subplots

fig, axs = plt.subplots(1, 2, figsize=(15, 6))



# Plot for even actions

cax1 = axs[0].imshow(even_action_frequencies, aspect='auto', cmap='Blues', interpolation='nearest')

fig.colorbar(cax1, ax=axs[0], label='Frequency')

axs[0].set_xlabel('Even Action Index')

axs[0].set_ylabel('Episode')

axs[0].set_title('Heatmap of Even Action Frequencies')



# Plot for odd actions

cax2 = axs[1].imshow(odd_action_frequencies, aspect='auto', cmap='Blues', interpolation='nearest')

fig.colorbar(cax2, ax=axs[1], label='Frequency')

axs[1].set_xlabel('Odd Action Index')

axs[1].set_ylabel('Episode')

axs[1].set_title('Heatmap of Odd Action Frequencies')



# Adjust layout

plt.tight_layout()

#plt.show()



# Reward differences calculation

reward_differences = []

for j in range(len(predictions)):

    print(j)

    for i, pred in enumerate(predictions[j]):

        if pred is not None:

            actions_ = actions[i]

            rewards_ = rewards[i]

            pred = pred[0]

            pred = pred[0]

            for l, act in enumerate(actions_):

                reward_differences.append(pred[act] - rewards_[l])



# Compute moving average for reward differences

reward_differences_smooth = moving_average(reward_differences, 3000)



# Figure 2: Reward differences plot with moving averages

plt.figure(figsize=(10, 6))

plt.plot(reward_differences, label='Reward Differences', alpha=0.6)

plt.plot(reward_differences_smooth, label='Reward Differences (Moving Avg)', linestyle='--', color='red')

plt.xlabel('Steps')

plt.ylabel('Reward Differences')

plt.title('Reward Differences with Moving Average')

plt.legend()

plt.grid()

plt.show()
