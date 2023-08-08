from sklearn.exceptions import NotFittedError
from sklearn.tree import DecisionTreeRegressor
import random
import pylab
import numpy as np
from keras.models import Model, load_model
from keras.layers import Input, Dense, Lambda, Add
from keras.optimizers import Adam, RMSprop
from keras import backend as K
from gym.spaces import Discrete, Box
import sys
import logging
import tensorflow
import os
import psutil
from pympler import asizeof
from collections import deque
from gym import Env
from gym.spaces import Discrete, Box
from bikewheelcalc import BicycleWheel, Rim, Hub, ModeMatrix
from matplotlib.colors import TwoSlopeNorm
import matplotlib.pyplot as plt
import time
import datetime
import json
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib import patches as mpatches
import random
import optuna
import plotly
import plotly.io as pio

'''
this model implements the symmetry and variable network size to test the best acting and training configuration
the done flag is now raised with a predetermined (maybe still suboptimal) spoketension vector
todo:
    implement advantage and ddqn
    implement method to set hyperparam in agent init
    minmax the taining performance and test set eval


'''


class SumTree(object):
    data_pointer = 0

    # all nodes all values = 0
    def __init__(self, capacity):
        self.capacity = capacity

        # Generate the tree with all nodes values = 0
        # To understand this calculation (2 * capacity - 1) look at the schema below
        # Remember we are in a binary node (each node has max 2 children) so 2x size of leaf (capacity) - 1 (root node)
        # Parent nodes = capacity - 1
        # Leaf nodes = capacity
        self.tree = np.zeros(2 * capacity - 1)

        # Contains the experiences (so the size of data is capacity)
        self.data = np.zeros(capacity, dtype=object)

    def add(self, priority, data):
        # Look at what index we want to put the experience
        tree_index = self.data_pointer + self.capacity - 1

        """ tree:
                    0
                   / \
                  0   0
                 / \ / \
        tree_index  0 0  0  We fill the leaves from left to right
        """

        # Update data frame
        self.data[self.data_pointer] = data

        # Update the leaf
        self.update(tree_index, priority)

        # Add 1 to data_pointer
        self.data_pointer += 1

        if self.data_pointer >= self.capacity:  # If we're above the capacity, we go back to first index (we overwrite)
            self.data_pointer = 0

    def update(self, tree_index, priority):
        # Change = new priority score - former priority score
        change = priority - self.tree[tree_index]
        self.tree[tree_index] = priority

        # then propagate the change through tree
        # this method is faster than the recursive loop
        while tree_index != 0:
            tree_index = (tree_index - 1) // 2
            self.tree[tree_index] += change

    def get_leaf(self, v):
        parent_index = 0

        while True:
            left_child_index = 2 * parent_index + 1
            right_child_index = left_child_index + 1

            # If we reach bottom, end the search
            if left_child_index >= len(self.tree):
                leaf_index = parent_index
                break
            else:  # downward search, always search for a higher priority node
                if v <= self.tree[left_child_index]:
                    parent_index = left_child_index
                else:
                    v -= self.tree[left_child_index]
                    parent_index = right_child_index

        data_index = leaf_index - self.capacity + 1

        return leaf_index, self.tree[leaf_index], self.data[data_index]

    @property
    def total_priority(self):
        return self.tree[0]  # Returns the root node


class WheelEnv(Env):

    def __init__(self, len_theta=360, n_spokes=36,
                 render=True, logging=True,
                 filename="dump.txt"):  # theta as 360 so that state lines up with spokes pos.

        self.len_theta = len_theta
        self.n_spokes = n_spokes

        self.best_reward = 0

        self.render = render

        self.logging = logging

        self.filename = filename

        self.max_tension = 2

        self.theta = np.linspace(-np.pi, np.pi, 360)

        self.observation_space = Box(low=-50, high=50, shape=([1080]))

        self.action_space = Discrete(n_spokes * 2)

        # self.episode_length = self.episode_max

        self.plot_scaling = 5

        ### Define Parameters ###

        hub_width = 0.05
        hub_diameter = 0.05

        self.rim_radius = 0.3
        rim_area = 100e-6
        rim_I_lat = 200. / 69e9
        rim_I_rad = 100. / 69e9
        rim_J_tor = 25. / 26e9
        rim_young_mod = 69e9
        rim_shear_mod = 26e9
        rim_I_warp = 0.0

        spokes_crossings = 3
        spokes_diameter = 2.0e-3
        spokes_young_mod = 210e9

        number_modes = 40

        init_tension = 800.

        # Create wheel and rim

        self.wheel = BicycleWheel()
        self.wheel.hub = Hub(width=hub_width, diameter=hub_diameter)
        self.wheel.rim = Rim(radius=self.rim_radius, area=rim_area,
                             I_lat=rim_I_lat, I_rad=rim_I_rad, J_tor=rim_J_tor, I_warp=rim_I_warp,
                             young_mod=rim_young_mod, shear_mod=rim_shear_mod)
        self.wheel.lace_cross(n_spokes=n_spokes, n_cross=spokes_crossings, diameter=spokes_diameter,
                              young_mod=spokes_young_mod)

        # Create a ModeMatrix
        self.mm = ModeMatrix(self.wheel, N=number_modes)

        # apply spokes tension
        self.wheel.apply_tension(init_tension)

        self.tensionchanges = np.random.rand(self.n_spokes) * self.max_tension - (self.max_tension / 2)
        plt.style.use('dark_background')
        self.norm = TwoSlopeNorm(vmin=-self.max_tension, vcenter=0, vmax=self.max_tension)
        self.rewards = []  # Initialize empty list for rewards
        self.fig = plt.figure(figsize=(15, 5))  # change the size to accommodate the new subplot
        self.ax1 = self.fig.add_subplot(131)  # first subplot for bar plot
        self.ax2 = self.fig.add_subplot(132, projection='3d')  # second subplot for 3D plot
        self.ax3 = self.fig.add_subplot(133)  # third subplot for reward plot
        self.ax3.set_title("Reward per Episode")
        self.ax3.set_xlabel("Episode")
        self.ax3.set_ylabel("Reward")
        self.spokes_lines = []
        if self.render:
            self.init_plot()

    def init_plot(self):
        self.bars = self.ax1.bar(range(self.n_spokes), self.tensionchanges)
        self.ax1.set_ylim([-self.max_tension, self.max_tension])
        self.ax2.set_xlabel('X')
        self.ax2.set_ylabel('Y')
        self.ax2.set_zlabel('Z')
        self.ax2.quiver([0, 0, 0], [0, 0, 0], [0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], color='r',
                        arrow_length_ratio=0.1, length=0.1)

        self.x = self.rim_radius * np.cos(self.theta)
        self.y = np.zeros_like(self.theta)
        self.z = self.rim_radius * np.sin(self.theta)
        self.ax2.plot(self.x, self.y, self.z)

        self.spokes_lines = self.draw_spokes()

        m = cm.ScalarMappable(cmap=plt.cm.seismic, norm=self.norm)
        m.set_array([])  # just to avoid warning message
        colorbar = plt.colorbar(m, ax=self.ax2, shrink=0.5)
        self.ax1.set_title("Spoke Tensions")
        self.ax2.set_title("Wheel 3D Visualization")

        ticks = np.linspace(-self.max_tension, self.max_tension, 5)  # 5 ticks evenly spaced
        tick_labels = [str(tick) for tick in ticks]

        colorbar.set_ticks(ticks)
        colorbar.set_ticklabels(tick_labels)

        self.ax1.spines['top'].set_visible(False)
        self.ax1.spines['right'].set_visible(False)

        plt.show(block=False)

    def update_reward_plot(self):
        self.ax3.clear()  # clear previous plot
        self.ax3.plot(self.rewards, label="Reward")  # plot the new reward data
        self.ax3.legend()  # show the legend
        self.fig.canvas.draw()  # update the figure
        self.fig.canvas.flush_events()
        plt.pause(0.001)

    def draw_spokes(self):
        spokes_lines = []
        for i in range(self.n_spokes):
            spoke_end_x = self.x[i * (len(self.x) // self.n_spokes)]
            spoke_end_y = self.y[i * (len(self.y) // self.n_spokes)]
            spoke_end_z = self.z[i * (len(self.z) // self.n_spokes)]

            spoke_tension = self.tensionchanges[i]

            line, = self.ax2.plot([0, spoke_end_x], [0, spoke_end_y], [0, spoke_end_z],
                                  color=plt.cm.seismic(self.norm(spoke_tension)))  # Apply colormap
            spokes_lines.append(line)
        return spokes_lines

    def update_3Dplot(self, rad_def, lat_def, tan_def):
        self.ax2.clear()  # clear the plot
        # self.ax2.quiver([0, 0, 0], [0, 0, 0], [0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], color='r',
        #                arrow_length_ratio=0.1, length=0.1)
        self.ax2.set_title("Wheel 3D Visualization")
        self.ax2.set_ylim([-0.2, 0.2])
        self.ax2.set_xlim([-(self.rim_radius * 1.1), self.rim_radius * 1.1])
        self.ax2.set_zlim([-(self.rim_radius * 1.1), self.rim_radius * 1.1])
        self.ax2.set_xlabel('X')
        self.ax2.set_ylabel('Y')
        self.ax2.set_zlabel('Z')
        self.ax2.plot(self.x, self.y, self.z, color="green", linestyle="dotted")
        # self.ax2.set(facecolor="dark green")
        self.ax2.set_axis_off()

        x_disp = np.zeros(len(self.x))
        y_disp = np.zeros(len(self.x))
        z_disp = np.zeros(len(self.x))

        for i in range(len(self.theta)):
            normal = [-self.x[i], -self.y[i], -self.z[i]] / np.linalg.norm([self.x[i], self.y[i], self.z[i]])
            normal_tan = [self.z[i], self.y[i], -self.x[i]] / np.linalg.norm([self.x[i], self.y[i], self.z[i]])
            x_disp[i] = self.x[i] + normal[0] * lat_def[i] / 1000 * self.plot_scaling + normal_tan[0] * tan_def[
                i] / 1000 * self.plot_scaling
            y_disp[i] = self.y[i] + normal[1] * lat_def[i] / 1000 * self.plot_scaling + rad_def[
                i] / 1000 * self.plot_scaling
            z_disp[i] = self.z[i] + normal[2] * lat_def[i] / 1000 * self.plot_scaling + normal_tan[2] * tan_def[
                i] / 1000 * self.plot_scaling

        self.ax2.plot(x_disp, y_disp, z_disp, color="grey", linestyle="solid")

        # Remove the previous spokes lines from the plot
        for line in self.spokes_lines:
            line.remove()

        # Re-initialize the list
        self.spokes_lines = []

        # Update the spokes
        for i in range(self.n_spokes):
            spoke_end_x = x_disp[i * (len(x_disp) // self.n_spokes)]
            spoke_end_y = y_disp[i * (len(y_disp) // self.n_spokes)]
            spoke_end_z = z_disp[i * (len(z_disp) // self.n_spokes)]

            spoke_tension = self.tensionchanges[i]

            line, = self.ax2.plot([0, spoke_end_x], [0, spoke_end_y], [0, spoke_end_z],
                                  color=plt.cm.seismic(self.norm(spoke_tension)))
            self.spokes_lines.append(line)

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        plt.pause(0.001)

    def update_plot(self):
        for bar, h in zip(self.bars, self.tensionchanges):
            bar.set_height(h)
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        plt.pause(0.001)  # slight pause to allow for update

    def step(self, action):
        spoke_index = action // 2
        adjustment = -0.5 if action % 2 == 0 else 0.5
        self.previous_tensionchanges = self.tensionchanges
        self.tensionchanges[spoke_index] += adjustment
        if self.logging:
            with open(self.filename, 'a') as f:
                f.write(f" tensionchanges: {self.tensionchanges}\n")
        next_state, reward, done = self.wheel_clac(self.tensionchanges)

        if self.render:
            self.update_plot()  # update the plot after adjusting the spoke tension
            self.update_3Dplot(next_state[0::3], next_state[1::3],
                               next_state[2::3])  # update 3D plot after spoke tension change
            self.rewards.append(reward)  # store the reward in the list
            self.update_reward_plot()
        return next_state, reward, done, {}

    def reset(self):
        self.tensionchanges = np.random.rand(self.n_spokes) * 2 - 1
        self.previous_tensionchanges = self.tensionchanges
        self.rewards = []
        if self.logging:
            with open(self.filename, 'a') as f:
                f.write(f" reset with tensionchanges: {self.tensionchanges}\n")
        self.best_tension = self.tensionchanges % 0.5
        state, reward, done = self.wheel_clac(self.tensionchanges)
        discard, self.best_reward, done = self.wheel_clac(self.best_tension)

        return state, reward

    def wheel_clac(self, spoketension):

        a = spoketension

        # Calculate stiffness matrix
        K = self.mm.K_rim(tension=True) + self.mm.K_spk(smeared_spokes=False, tension=True)

        # use adjustment vector and matrix to change spoke tension
        F = self.mm.A_adj().dot(a)

        # Solve for the mode coefficients
        dm = np.linalg.solve(K, F)

        # Get radial deflection

        rad_def = self.mm.rim_def_rad(self.theta, dm)
        lat_def = self.mm.rim_def_lat(self.theta, dm)
        # rot_def = mm.rim_def_rot(theta, dm)
        tan_def = self.mm.rim_def_tan(self.theta, dm)
        tot_def = np.column_stack((rad_def, lat_def, tan_def))

        reward = - np.sum(np.linalg.norm(tot_def, axis=1))

        if reward >= self.best_reward:  # done if model finds naive best guess
            done = True
        if self.max_tension <= max(abs(self.previous_tensionchanges)) < max(
                abs(self.tensionchanges)):  # enforce max tension through reward
            reward = reward - 1000

        next_state = tot_def.flatten()

        done = False  # decide when an episode is done

        return next_state, reward, done


class Memory(object):  # stored as ( state, action, reward, next_state ) in SumTree
    PER_e = 0.01  # Hyperparameter that we use to avoid some experiences to have 0 probability of being taken
    PER_a = 0.6  # Hyperparameter that we use to make a tradeoff between taking only exp with high priority and sampling randomly
    PER_b = 0.4  # importance-sampling, from initial value increasing to 1

    PER_b_increment_per_sampling = 0.001

    absolute_error_upper = 1.  # clipped abs error

    def __init__(self, capacity):
        # Making the tree
        self.tree = SumTree(capacity)

    def store(self, experience):
        # Find the max priority
        max_priority = np.max(self.tree.tree[-self.tree.capacity:])

        # minimum priority
        if max_priority == 0:
            max_priority = self.absolute_error_upper

        self.tree.add(max_priority, experience)

    def sample(self, n):
        # Create a minibatch array that will contains the minibatch
        minibatch = []

        b_idx = np.empty((n,), dtype=np.int32)

        # Calculate the priority segment
        # divide the Range[0, ptotal] into n ranges
        priority_segment = self.tree.total_priority / n  # priority segment

        for i in range(n):
            # A value is uniformly sample from each range
            a, b = priority_segment * i, priority_segment * (i + 1)
            value = np.random.uniform(a, b)

            # Experience that correspond to each value is retrieved
            index, priority, data = self.tree.get_leaf(value)

            b_idx[i] = index

            minibatch.append([data[0], data[1], data[2], data[3], data[4]])

        return b_idx, minibatch

    def batch_update(self, tree_idx, abs_errors):
        abs_errors += self.PER_e  # convert to abs and avoid 0
        clipped_errors = np.minimum(abs_errors, self.absolute_error_upper)
        ps = np.power(clipped_errors, self.PER_a)

        for ti, p in zip(tree_idx, ps):
            self.tree.update(ti, p)


def NNModel(input_shape, action_space, learning_rate, layer_sizes=[512, 256, 64]):  # lr=0.00025
    X_input = Input(input_shape)
    X = X_input

    for size in layer_sizes:
        if size > 0:  # Create a layer only if the size is more than zero
            X = Dense(size, activation="relu", kernel_initializer='random_normal', bias_initializer='zeros')(X)

    X = Dense(action_space, activation="linear", kernel_initializer='random_normal', bias_initializer='zeros')(X)

    model = Model(inputs=X_input, outputs=X, name='model')
    model.compile(loss="mean_squared_error",
                  optimizer=RMSprop(learning_rate=learning_rate, rho=0.95, epsilon=0.01),
                  metrics=["accuracy"])

    model.summary()
    return model


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

    import random

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
            q_values = self.model.predict(states)
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
                        print('test')
                        state = next_state
                        i += 1
                        if i >= self.max_episode_len:
                            break
                        if i % 10 == 0:
                            self.replay()

    def load(self, name):
        self.model = load_model(name)
        # w = self.model.get_weights()
        # print(w)

    def save(self, name):
        self.model.save(name)


def load_data(file_name):
    data = np.load(file_name, allow_pickle=True)
    return data


# Load the data
data = load_data('results_all.npz')

# Accessing the stored arrays
states = data['states']
rewards = data['rewards']
tensions = data['tensions']
actions = data['actions']

linear = [0]
tiny = [100]
small = [100, 50]
medium = [200, 100]
large = [300, 200, 100]
verylarge = [512, 256, 64]
''' is this the working model?
env = WheelEnv()
agent = DQNAgent(env, 12, 0.00025, 0.1, [512, 256, 64])
agent.run(1)
acc, eff = agent.evaluate(states, tensions, rewards)
print(acc)
print(eff)
'''


def call_func_with_default_if_none(func, environment, params):
    params = {k: v for k, v in params.items() if v is not None}
    return func(environment, **params)


counter = 1


def objective(trial):
    tensorflow.keras.backend.clear_session()
    gamma = trial.suggest_uniform('gamma', 0.0, 0.9)
    max_depth = trial.suggest_int('max_depth', 1, 50)
    max_leafs = trial.suggest_int('max_leafs', 100, 2000)
    mem_size = trial.suggest_int('mem_size', 1000, 10000)
    print(max_leafs)

    global counter
    counter = counter + 1

    logging = True
    filename = f"test_tree_{counter}.txt"

    param_env = {
        'logging': logging,
        'render': False,
        'filename': filename
    }

    param_agent = {
        'batch_size': 32,
        'learning_rate': 0.00025,
        'gamma': gamma,
        'layer_sizes': [512, 256, 100],
        'shift': False,
        'Model_type': "Tree",
        'use_per': False,
        'memory_size': mem_size,
        'max_episode_len': 150,
        'tree_max_depth': max_depth,
        'tree_max_leafs': max_leafs,
        'epsilon': 1,
        'epsilon_min': 0.1,
        'epsilon_decay': 0.005,
        'logging': True,
        'filename': filename
    }

    env = WheelEnv(**param_env)
    agent = call_func_with_default_if_none(DQNAgent, env, param_agent)
    agent.run(150)
    #agent.save(f"tree_model_{counter}")

    acc, eff = agent.evaluate(states, tensions, rewards)  # compute intermediate score
    print('eval done, acc: ', acc, "ef: ", eff)
    score = np.sum(acc)  # + np.sum(eff)
    return score


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

'''

logging = True
filename = "test2.txt"

param_env = {
    'logging': logging,
    'render': False,
    'filename': filename
}

param_agent = {
    'batch_size': 32,
    'learning_rate': 0.00025,
    'gamma': 0.2,
    'layer_sizes': [512, 256, 100],
    'shift': False,
    'Model_type': "NN",
    'use_per': True,
    'memory_size': 5000,
    'max_episode_len': 100,
    'tree_max_depth': 2,
    'tree_max_leafs': 200,
    'epsilon': 0.5,
    'epsilon_min': 0.1,
    'epsilon_decay': 0.005,
    'logging': True,
    'filename': filename
}


env = WheelEnv(**param_env)
agent = call_func_with_default_if_none(DQNAgent, env, param_agent)
agent.load('03.08.23')
agent.run(300)
agent.save('04.08.23')

'''
