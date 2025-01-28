import pandas as pd
import ast
import numpy as np
import re
import matplotlib.pyplot as plt

# Load the CSV file
file_path = 'SessionLogs\session10.csv'  # Replace with the path to your CSV file
data = pd.read_csv(file_path)
print(data['predictions'].head())


# Convert `initialstate`, `predictions`, `actions`, and `rewards` columns from strings to lists
data['initialstate'] = data['initialstate'].apply(ast.literal_eval)
def preprocess_predictions(pred):
    if isinstance(pred, str):
        pred = re.sub(r'array\((.+?)\)', r'[\1]', pred)  # Replace array(...) with [...], valid for ast.literal_eval
        try:
            return ast.literal_eval(pred)  # Attempt parsing
        except (ValueError, SyntaxError):
            return None
    return pred

data['predictions'] = data['predictions'].apply(preprocess_predictions)
data['actions'] = data['actions'].apply(ast.literal_eval)
data['rewards'] = data['rewards'].apply(ast.literal_eval)

# Example: Convert datetime to a proper datetime object
data['datetime'] = pd.to_datetime(data['datetime'])

# Now you can access the data as needed
print(data.head())

# Example: Extract specific data for further analysis
initial_states = np.array(data['initialstate'].tolist())
actions = np.array(data['actions'].tolist())
rewards = np.array(data['rewards'].tolist())
predictions = data['predictions'].tolist()  # Leave as list since it contains mixed data types

# Example: Access the first row's data
print("First initial state:", initial_states[0])
print("First action sequence:", actions[0])
print("First rewards sequence:", rewards[0])
print("First prediction:", predictions[0])

last_rewards = []
for i in rewards:
    last_rewards.append(i[-1])
plt.plot(last_rewards)
plt.show()