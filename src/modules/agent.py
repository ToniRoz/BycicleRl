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
from sklearn.tree import DecisionTreeRegressor
from sklearn.exceptions import NotFittedError
from .memory import Memory
from models.model import NNModel



class DQNAgent:
    def __init__(self, env, batch_size=32, learning_rate=0.00025, gamma=0, layer_sizes=[512, 256, 64], shift=False,
                 Model_type="NN", use_per=True, memory_size=1000, max_episode_len=100, tree_max_depth=2,
                 tree_max_leafs=100, epsilon=0.1, epsilon_min=0.01,
                 epsilon_decay=0.005, logging=True, filename="dump.txt"):
        self.SHIFT = False
        self.env = env
        self.state_size = self.env.observation_space.shape[0]
        self.action_size = self.env.action_space.n
        self.logging = logging

        if shift:
            self.SHIFT = True
            self.n_action_types = 4
            self.shift_values = [i * -60 for i in  # 60 since the 3d state will be flattend "-" because shift to left
                                 range(18)]  # Update the shift_values based on the action to state relation

        self.EPISODES = 1
        self.max_episode_len = max_episode_len
        self.memory_size = memory_size
        self.MEMORY = Memory(self.memory_size)
        self.memory = deque(maxlen=2000)
        self.gamma = gamma  # discount rate

        self.USE_PER = use_per

        self.ddqn = False

        self.Model_type = Model_type

        # EXPLORATION HYPERPARAMETERS for epsilon and epsilon greedy strategy
        self.epsilon = epsilon  # exploration probability at start
        self.epsilon_min = epsilon_min  # minimum exploration probability
        self.epsilon_decay = epsilon_decay  # exponential decay rate for exploration prob

        self.batch_size = batch_size

        self.Save_Path = 'Models'
        if not os.path.exists(self.Save_Path): os.makedirs(self.Save_Path)
        self.scores, self.episodes, self.average = [], [], []

        # self.Model_name = os.path.join(self.Save_Path, "_e_greedy.h5")

        if self.Model_type == "NN" and self.SHIFT:
            self.model = NNModel(input_shape=(self.state_size,), action_space=4, learning_rate=learning_rate,
                                 layer_sizes=layer_sizes)
        if self.Model_type == "NN" and not self.SHIFT:
            self.model = NNModel(input_shape=(self.state_size,), action_space=72, learning_rate=learning_rate,
                                 layer_sizes=layer_sizes)
        if self.Model_type == "Tree":
            self.tree_max_depth = tree_max_depth
            self.tree_max_leafs = tree_max_leafs
            self.model = DecisionTreeRegressor(max_depth=self.tree_max_depth, max_leaf_nodes=self.tree_max_leafs)
            print(f"Model = Regression Tree with param: depth = {self.tree_max_depth}, max_leaf_node = {self.tree_max_leafs}")  # need to tune hyperparam here

        params = locals()
        del params['self']  # Remove 'self' entry from the dictionary

        # Convert non-serializable types to strings
        for key, value in params.items():
            try:
                json.dumps(value)
            except TypeError:
                params[key] = str(value)

        # If logging is enabled, write the parameters to a file
        if logging:
            # Generate the filename with the current date
            self.filename = filename

            # Write the parameters to the file
            with open(self.filename, 'w') as f:
                f.write(json.dumps(params, indent=4))

    def shift_state(self, state, shift_value):

        return np.roll(state, shift_value)  # Shift the state to the left by shift_value

    def remember(self, state, action, reward, next_state, done):
        experience = state, action, reward, next_state, done
        if self.USE_PER:
            self.MEMORY.store(experience)
        else:
            self.memory.append(experience)


    def shift_act(self, state, decay_step):
        # EPSILON GREEDY STRATEGY
        explore_probability = self.epsilon_min + (self.epsilon - self.epsilon_min) * np.exp(
            -self.epsilon_decay * decay_step)

        if explore_probability > random.random():
            random_action = random.randrange(self.action_size)
            if self.logging:
                with open(self.filename, 'a') as f:
                    f.write(f" random action: {random_action}\n")
            return random_action, explore_probability
        else:
            try:
                # Prepare the batch of shifted states
                shifted_states = [self.shift_state(state, shift) for shift in self.shift_values]
                q_values = self.model.predict(np.array(shifted_states).reshape(len(shifted_states), -1), verbose=0)

                # Flatten q_values
                q_values = q_values.flatten()
                if self.logging:
                    with open(self.filename, 'a') as f:
                        f.write(f" prediction: {np.max(q_values)}\n for action {np.argmax(q_values)}\n")

                return np.argmax(q_values), explore_probability
            except NotFittedError:
                random_action = random.randrange(self.action_size)
                if self.logging:
                    with open(self.filename, 'a') as f:
                        f.write(f" random action: {random_action}\n")
                return random_action

    def shift_evalact(self, state):

        q_values = []
        for shift in self.shift_values:
            shifted_state = self.shift_state(state, shift)
            q_values_action = self.model.predict(shifted_state.reshape(1, -1), verbose=0)[0]
            for q_value in q_values_action:
                q_values.append(q_value)

        if self.logging:
            with open(self.filename, 'a') as f:
                f.write(f" eval_prediction: {np.max(q_values)}\n for action {np.argmax(q_values)}\n")

        return np.argmax(q_values)

    def evalact(self, state):

        try:
            q_values = self.model.predict(state.reshape(1, -1))
            if self.logging:
                with open(self.filename, 'a') as f:
                    f.write(f"eval prediction: {np.max(q_values)}\n for action {np.argmax(q_values)}\n")
            return np.argmax(q_values)
        except NotFittedError:
            random_action = random.randrange(self.action_size)
            if self.logging:
                with open(self.filename, 'a') as f:
                    f.write(f" eval error model not fitted random action: {random_action}\n")
            return random_action

    def act(self, state, decay_step):

        explore_probability = self.epsilon_min + (self.epsilon - self.epsilon_min) * np.exp(
            -self.epsilon_decay * decay_step)

        if explore_probability > np.random.rand():
            random_action = random.randrange(self.action_size)
            if self.logging:
                with open(self.filename, 'a') as f:
                    f.write(f" random action: {random_action}\n")
            return random_action, explore_probability

        else:
            try:
                q_values = self.model.predict(state.reshape(1, -1))
                if self.logging:
                    with open(self.filename, 'a') as f:
                        f.write(f" prediction: {np.max(q_values)}\n for action {np.argmax(q_values)}\n")
                return np.argmax(q_values), explore_probability
            except NotFittedError:
                random_action = random.randrange(self.action_size)
                if self.logging:
                    with open(self.filename, 'a') as f:
                        f.write(f" random action: {random_action}\n")
                return random_action, explore_probability

    def shift_replay(self):
        # Sample a mini-batch from memory
        minibatch, tree_idx = (self.MEMORY.sample(self.batch_size) if self.USE_PER
                               else (random.sample(self.memory, min(len(self.memory), self.batch_size)), None))

        # Initialize the arrays for storing the samples
        state, next_state = np.zeros((self.batch_size, self.state_size)), np.zeros((self.batch_size, self.state_size))
        action, reward, done = np.zeros(self.batch_size), np.zeros(self.batch_size), np.zeros(self.batch_size,
                                                                                              dtype=bool)

        # Store the samples in the arrays
        for i in range(self.batch_size):
            state[i], action[i], reward[i], next_state[i], done[i] = minibatch[i]

        # Compute the action types and shift values, and adjust the states
        action_type = action % self.n_action_types
        for i in range(len(minibatch)):
            shift_value = self.shift_values[int(action[i] // 4)]
            state[i] = np.roll(state[i], shift_value)

        try:  # check logic here if value estimation is right
            q_values = self.model.predict(state)
        except NotFittedError:
            q_values = reward

        # Copy the predicted Q-values as the target Q-values
        target_q_values = np.array(q_values)

        # Adjust the target Q-values
        try:
            for i in range(len(minibatch)):
                if done[i]:
                    target_q_values[i][int(action_type[i])] = reward[i]
                else:
                    shifted_next_states = [np.roll(next_state[i], shift_value).reshape(1, -1) for shift_value in
                                           self.shift_values]
                    q_values_next_shifted = self.model.predict(np.array(shifted_next_states))
                    q_values_next_shifted = q_values_next_shifted.flatten()
                    max_q_next = max(q_values_next_shifted)
                    target_q_values[i][int(action_type[i])] = reward[i] + self.gamma * max_q_next
        except NotFittedError:
            pass

        # Compute the absolute errors between the old and new Q-values
        if self.USE_PER:
            absolute_errors = np.abs(q_values - target_q_values).max(axis=1)
            self.MEMORY.batch_update(tree_idx, absolute_errors)

        batch_size = self.batch_size if self.Model_type == "NN" else None
        self.model.fit(state, target_q_values, batch_size=batch_size, verbose=0)

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

        for i in range(min(self.batch_size,len(self.memory))):
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
        except NotFittedError:
            target = reward
            target_next = 0

        for i in range(len(minibatch)):
            if i == 0:
                print(len(minibatch))
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
                    target[i][action[i]] = reward[i] + self.gamma * (target_val[i][a])
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

    def tree_replay(self):

        minibatch = random.sample(self.memory, len(self.memory))  # Initialize the arrays for storing the samples

        state = np.zeros((len(minibatch), self.state_size))
        action, reward, done = [], [], []
        next_state = np.zeros((len(minibatch), self.state_size))

        # Store the samples in the arrays

        for i in range(len(minibatch)):
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
        except NotFittedError:
            target = np.zeros((len(minibatch),self.action_size))
            target_next = np.zeros((len(minibatch),self.action_size))

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
                    target[i][action[i]] = reward[i] + self.gamma * (target_val[i][a])
                else:  # Standard - DQN
                    # DQN chooses the max Q value among next actions
                    # selection and evaluation of action is on the target Q Network
                    # Q_max = max_a' Q_target(s', a')
                    target[i][action[i]] = reward[i] + self.gamma * (np.amax(target_next[i]))

        self.model.fit(state, target)





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
                if self.SHIFT:
                    action = self.shift_evalact(state)
                else:
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
        if self.SHIFT:
            if self.Model_type == "Tree":
                for e in range(EPISODES):
                    state, reward = self.env.reset()
                    state = np.reshape(state, [1, 1080])
                    for time in range(self.max_episode_len):
                        decay_step += 1
                        action, explore_probability = self.shift_act(state, decay_step)
                        next_state, reward, done, _ = self.env.step(action)
                        if self.logging:
                            with open(self.filename, 'a') as f:
                                f.write(f" reward: {reward}\n")
                        next_state = np.reshape(next_state, [1, 1080])
                        self.remember(state, action, reward, next_state, done)
                        state = next_state
                        if done:
                            break
                    self.shift_replay(self.memory_size)
            else:
                for e in range(EPISODES):
                    state, first = self.env.reset()
                    state = np.reshape(state, [1, self.state_size])
                    done = False
                    i = 0
                    while not done:
                        decay_step += 1
                        action, explore_probability = self.shift_act(state, decay_step)
                        next_state, reward, done, _ = self.env.step(action)
                        if self.logging:
                            with open(self.filename, 'a') as f:
                                f.write(f" reward: {reward}\n")
                        next_state = np.reshape(next_state, [1, self.state_size])

                        self.remember(state, action, reward, next_state, done)
                        state = next_state
                        i += 1
                        if i >= self.max_episode_len:
                            break
                        if i % 10 == 0:
                            self.shift_replay()
        else:
            if self.Model_type == "Tree":
                for e in range(EPISODES):
                    state, reward = self.env.reset()
                    state = np.reshape(state, [1, 1080])
                    for time in range(self.max_episode_len):
                        decay_step += 1
                        action, explore_probability = self.act(state, decay_step)
                        next_state, reward, done, _ = self.env.step(action)
                        if self.logging:
                            with open(self.filename, 'a') as f:
                                f.write(f" reward: {reward}\n")
                        next_state = np.reshape(next_state, [1, 1080])
                        self.remember(state, action, reward, next_state, done)
                        state = next_state
                        if done:
                            break

                    self.tree_replay()
            else:
                decay_step = 0
                for e in range(EPISODES):
                    state, first = self.env.reset()
                    state = np.reshape(state, [1, self.state_size])
                    done = False
                    i = 0
                    tensorflow.keras.backend.clear_session()
                    while not done:
                        decay_step += 1
                        action, explore_probability = self.act(state, decay_step)
                        next_state, reward, done, _ = self.env.step(action)
                        if self.logging:
                            with open(self.filename, 'a') as f:
                                f.write(f" reward: {reward}\n")
                        next_state = np.reshape(next_state, [1, self.state_size])

                        self.remember(state, action, reward, next_state, done)
                        state = next_state
                        i += 1
                        if i >= self.max_episode_len:
                            break
                        if i % 10 == 0:
                            self.replay()
    def save(self, name):
        self.model.save(name)

    def load(self, name):
        self.model = load_model(name)