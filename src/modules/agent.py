import sklearn
import numpy as np
import random
import tensorflow
from keras.models import Model, load_model
from keras.layers import Input, Dense, Lambda, Add
from keras.optimizers import Adam, RMSprop
from keras import backend as K
from collections import deque
import os
import json
from .memory import Memory
from models.models import NN_Model
from datetime import datetime
import csv




class DQNAgent:
    def __init__(self, env, logging_dir, batch_size=32, learning_rate=0.00025, gamma=0, layer_sizes=[512, 256, 64],
                 Model_type="NN", use_per=True, memory_size=1000, max_episode_len=100,
                epsilon=0.1, epsilon_min=0.01,
                 epsilon_decay=0.005, logging=True):
        
        # Environment/logging Variables
        self.env = env
        self.state_size = self.env.observation_space.shape[0]
        self.action_size = self.env.action_space.n
        self.logging = logging
        self.logging_dir = logging_dir

        # Agent/Session variables

        self.EPISODES = 1
        self.max_episode_len = max_episode_len
        self.memory_size = memory_size
        self.MEMORY = Memory(self.memory_size)
        self.memory = deque(maxlen=2000)
        self.gamma = gamma  # discount rate
        self.epsilon = epsilon  # exploration probability at start
        self.epsilon_min = epsilon_min  # minimum exploration probability
        self.epsilon_decay = epsilon_decay  # exponential decay rate for exploration prob
        self.USE_PER = use_per
        self.batch_size = batch_size
        self.ddqn = False
        self.Predictions = []
        self.Actions = []
        self.Rewards = []
        self.State_0 = []
        self.Type = None

        # Model variables
        self.layer_sizes = layer_sizes
        self.learning_rate = learning_rate
        self.Model_type = Model_type

        self.Save_Path = 'Models'
        if not os.path.exists(self.Save_Path): os.makedirs(self.Save_Path)
        self.scores, self.episodes, self.average = [], [], []

        # self.Model_name = os.path.join(self.Save_Path, "_e_greedy.h5")

        if self.Model_type == "NN":
            self.model = NN_Model(input_shape=(self.state_size,), action_space=72, learning_rate=learning_rate,
                                 layer_sizes=layer_sizes)


        if self.logging:
            self.log_parameters(gamma=self.gamma, epsilon=self.epsilon, layers=self.layer_sizes,
                                epsilon_min=self.epsilon_min, epsilon_decay=self.epsilon_decay,
                                memory_size=self.memory_size, batch_size=self.batch_size, model_type=self.Model_type,
                                use_per=self.USE_PER, learning_rate=self.learning_rate)
            self.create_session_csv()

    def log_parameters(self, learning_rate, gamma, epsilon, layers, epsilon_min, epsilon_decay, memory_size, batch_size, model_type, use_per):
        """
        Log parameters to a new session parameters file.

        Args:
            learning_rate: Learning rate parameter.
            gamma: Discount factor.
            epsilon: Exploration rate.
            **kwargs: Additional optional parameters to log.
        """
        # Find the next highest session number for .txt files
        existing_files = [f for f in os.listdir(self.logging_dir) if f.endswith("parameters.txt")]
        session_numbers = [int(f.split("session")[1].split("parameters")[0]) for f in existing_files]
        next_session_number = max(session_numbers, default=0) + 1
        param_file = os.path.join(self.logging_dir, f"session{next_session_number}parameters.txt")

        # Write parameters to the file
        with open(param_file, "w") as f:
            f.write(f"Learning Rate: {learning_rate}\n")
            f.write(f"Gamma: {gamma}\n")
            f.write(f"Epsilon: {epsilon}\n")
            f.write(f"lay: {layers[0]:d},{layers[1]:d},{layers[2]:d}\n")
            f.write(f"Epsilon_min: {epsilon_min}\n")
            f.write(f"Epsilon_decay: {epsilon_decay}\n")
            f.write(f"Memory Size: {memory_size}\n")
            f.write(f"Batch Size: {batch_size}\n")
            f.write(f"Model Type: {model_type}\n")
            f.write(f"Use PER: {use_per}\n")


        print(f"Parameters logged to {param_file}")

    def create_session_csv(self):
        # Find the next highest session number for .csv files
        existing_files = [f for f in os.listdir(self.logging_dir) if f.endswith(".csv")]
        session_numbers = [int(f.split("session")[1].split(".csv")[0]) for f in existing_files]
        next_session_number = max(session_numbers, default=0) + 1
        self.csv_file = os.path.join(self.logging_dir, f"session{next_session_number}.csv")

        # Create the CSV file with the header
        header = ["datetime", "episodetype", "episodenumber", "initialstate", "predictions", "actions", "rewards"]
        with open(self.csv_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(header)

        print(f"Session CSV created at {self.csv_file}")

    def log_to_csv(self, episodetype, episodenumber, initialstate, predictions, actions, rewards):
        """
        Log data to the CSV file.

        Args:
            episodetype: Type of episode (e.g., 'train', 'test').
            episodenumber: Current episode number.
            initialstate: Initial state as an array or list.
            predictions: Predictions made by the model, or None if random actions are taken.
            actions: Actions taken.
            rewards: Rewards received.
        """
        if not hasattr(self, 'csv_file'):
            raise ValueError("CSV file not created. Call `create_session_csv()` first.")

        # Helper function to safely format data
        def safe_array2string(data):
            if data is None:
                return "None"
            try:
                return np.array2string(np.array(data), precision=2, separator=',').replace('\n', '')
            except Exception as e:
                return str(data)

        # Format each field
        initialstate_str = safe_array2string(initialstate)
        predictions_str = safe_array2string(predictions)
        actions_str = safe_array2string(actions)

        row = [
            datetime.now().isoformat(),
            episodetype,
            episodenumber,
            initialstate_str,
            predictions_str,
            actions_str,
            rewards
        ]

        # Append to the CSV file
        with open(self.csv_file, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(row)

        print(f"Data logged to {self.csv_file}")


    def remember(self, state, action, reward, next_state, done):
        experience = state, action, reward, next_state, done
        if self.USE_PER:
            self.MEMORY.store(experience)
        else:
            self.memory.append(experience)


    

    def evalact(self, state):

        try:
            q_values = self.model.predict(state.reshape(1, -1))
            return np.argmax(q_values)
        except sklearn.exceptions.NotFittedError:
            random_action = random.randrange(self.action_size)
            return random_action

    def act(self, state, decay_step):

        explore_probability = self.epsilon_min + (self.epsilon - self.epsilon_min) * np.exp(
            -self.epsilon_decay * decay_step)
        #remove this from here and just change the display text variable
        self.env.update_text( self.epsilon, self.gamma, self.batch_size, explore_probability, self.epsilon_min, self.layer_sizes, self.Model_type, self.learning_rate)

        if explore_probability > np.random.rand():
            self.Predictions.append(None)
            random_action = random.randrange(self.action_size)
            self.Actions.append(random_action)
            return random_action, explore_probability

        else:
            try:
                q_values = self.model.predict(state.reshape(1, -1))
                self.Predictions.append(q_values)
                return np.argmax(q_values), explore_probability
            except sklearn.exceptions.NotFittedError:
                self.Predictions.append(None)
                random_action = random.randrange(self.action_size)
                self.Actions.append(random_action)
                return random_action, explore_probability


    def replay(self):

        if self.USE_PER:
            # Sample minibatch from the PER memory
            tree_idx, minibatch = self.MEMORY.sample(self.batch_size)
        else:
            # Randomly sample minibatch from the deque memory
            minibatch = random.sample(self.memory, min(len(self.memory),
                                                       self.batch_size))  # Initialize the arrays for storing the samples

        state = np.zeros((self.batch_size, self.state_size))
        action, reward, done = [], [], []
        next_state = np.zeros((self.batch_size, self.state_size))

        # Store the samples in the arrays

        for i in range(self.batch_size):# used to be min(self.batch_size,len(self.memory)) but it was causing an error with shorter episodes
            state[i] = minibatch[i][0]
            action.append(minibatch[i][1])
            reward.append(minibatch[i][2])
            next_state[i] = minibatch[i][3]
            done.append(minibatch[i][4])

        try:  # implement ddq and advantage
            target = self.model.predict(state)
            target_old = np.array(target)
            # predict best action in ending state using the main network
            target_next = self.model.predict(next_state)
            # predict Q-values for ending state using the target network
            # target_val = self.target_model.predict(next_state)
        except sklearn.exceptions.NotFittedError:
            target = reward
            target_next = 0

        for i in range(len(minibatch)):

            # correction on the Q value for the action used
            if done[i]:
                target[i][action[i]] = reward[i]
            else:
                if self.ddqn:  # Double - DQN
                    # current Q Network selects the action
                    # a'_max = argmax_a' Q(s', a')
                    a = np.argmax(target_next[i])
                    # target Q Network evaluates the action
                    # Q_max = Q_target(s', a'_max)
                    #target[i][action[i]] = reward[i] + self.gamma * (target_val[i][a])
                else:  # Standard - DQN
                    # DQN chooses the max Q value among next actions
                    # selection and evaluation of action is on the target Q Network
                    # Q_max = max_a' Q_target(s', a')
                    target[i][action[i]] = reward[i] + self.gamma * (np.amax(target_next[i]))

        if self.USE_PER:
            indices = np.arange(self.batch_size, dtype=np.int32)
            absolute_errors = np.abs(target_old[indices, np.array(action)] - target[indices, np.array(action)])
            # Update priority
            self.MEMORY.batch_update(tree_idx, absolute_errors)

        if self.Model_type == "NN":
            self.model.fit(state, target, batch_size=self.batch_size, verbose=0)





    def evaluate(self, states, tensions, rewards):  # todo: check if this eval function works

        agentrewards = []
        agentsteps = []
        accuracy = []
        efficiency = []

        for i in range(30):

            done = False
            print('eval ', i)
            state = np.array(states[i][0]).flatten()
            # print(state)
            tension = tensions[i][0]
            runtime = 0

            while not done:
                action = self.evalact(state)
                spoke_index = action // 2
                adjustment = -0.5 if action % 2 == 0 else 0.5
                tension[spoke_index] += adjustment

                next_state, reward, done_sim = self.env.wheel_clac(tension)
                runtime = runtime + 1
                done = False
                if reward > rewards[i][-2]:
                    done = True
                if runtime > 70:
                    done = True
                agentrewards.append(reward)
                agentsteps.append(runtime)
                state = next_state

            best_reward_index = np.argmax(agentrewards)
            accuracy.append(rewards[i][-2] / agentrewards[best_reward_index])
            efficiency.append((len(rewards[i]) - 1) / agentsteps[best_reward_index])

        return accuracy, efficiency

    def ievaluate(self, states, tensions, rewards):  # todo: check if this eval function works

        agentrewards = []
        agentsteps = []
        accuracy = []
        efficiency = []

        for i in range(10):

            done = False
            state = np.array(states[i][0]).flatten()
            # print(state)
            tension = tensions[i][0]
            runtime = 0

            while not done:

                action = self.evalact(state)
                spoke_index = action // 2
                adjustment = -0.5 if action % 2 == 0 else 0.5
                tension[spoke_index] += adjustment

                next_state, reward, done_sim = self.env.wheel_clac(tension)
                runtime = runtime + 1
                done = False
                if reward > rewards[i][-2]:
                    done = True
                if runtime > 70:
                    done = True
                agentrewards.append(reward)
                agentsteps.append(runtime)
                state = next_state

            best_reward_index = np.argmax(agentrewards)
            accuracy.append(rewards[i][-2] / agentrewards[best_reward_index])
            efficiency.append((len(rewards[i]) - 1) / agentsteps[best_reward_index])

        return accuracy, efficiency

    def run(self, EPISODES):

        decay_step = 0
        for e in range(EPISODES):
            state, first = self.env.reset()
            state = np.reshape(state, [1, self.state_size])
            self.State_0 = self.env.tensionchanges
            self.Type = "train"
            done = False
            i = 0
            tensorflow.keras.backend.clear_session()
            while not done:
                decay_step += 1
                action, explore_probability = self.act(state, decay_step)
                next_state, reward, done, _ = self.env.step(action)
                self.Rewards.append(reward)
                next_state = np.reshape(next_state, [1, self.state_size])

                self.remember(state, action, reward, next_state, done)
                state = next_state
                i += 1
                if i >= self.max_episode_len:
                    if self.logging:
                        self.log_to_csv(self.Type, e, self.State_0, self.Predictions, self.Actions, self.Rewards)
                        self.Type = None
                        self.Predictions = []
                        self.Actions = []
                        self.Rewards = []
                    break
                if i % 10 == 0:
                    self.replay()
                if done:
                    if self.logging:
                        self.log_to_csv(self.Type, e, self.State_0, self.Predictions, self.Actions, self.Rewards)
                        self.Type = None
                        self.Predictions = []
                        self.Actions = []
                        self.Rewards = []
    def save(self, name):
        self.model.save(name)

    def load(self, name):
        self.model = load_model(name)