from gym import Env
from gym.spaces import Discrete, Box
from bikewheelcalc import BicycleWheel, Rim, Hub, ModeMatrix
from matplotlib.colors import TwoSlopeNorm
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm




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

        self.max_tension = 3

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
        self.reward_sums = []  # Initialize reward sum
        self.fig = plt.figure(figsize=(15, 10))  # Adjust height to fit two rows of subplots


        # Add text elements at the top of the figure
        # Agent parameters
        self.batch_size = self.fig.text(0.1, 0.95, "None", fontsize=12)
        self.gamma = self.fig.text(0.1, 0.92, "None", fontsize=12)
        self.epsilon = self.fig.text(0.25, 0.95, "None", fontsize=12)
        self.epsilon_min = self.fig.text(0.25, 0.92, "None", fontsize=12)
        self.explore_probability = self.fig.text(0.4, 0.92, "None", fontsize=12)

        # model parameters
        self.model_type = self.fig.text(0.65, 0.95, "None", fontsize=12)
        self.layer_sizes = self.fig.text(0.65, 0.92, "None", fontsize=12)
        self.learning_rate = self.fig.text(0.8, 0.92, "None", fontsize=12)

        # First row: Original subplots
        self.ax1 = self.fig.add_subplot(231)  # First subplot in a 2x3 grid
        self.ax2 = self.fig.add_subplot(232, projection='3d')  # Second subplot in the first row
        self.ax3 = self.fig.add_subplot(233)  # Third subplot in the first row
        self.ax4 = self.fig.add_subplot(234)  # First subplot in the second row
        #self.ax5 = self.fig.add_subplot(235)  # Second subplot in the second row
        self.ax6 = self.fig.add_subplot(236)  # Third subplot in the second row

        # Example: Set titles for each subplot (optional)
        self.ax6.set_title('Session')
        self.ax6.set_xlabel("Episode")
        self.ax6.set_ylabel("cum. Reward")
        #self.ax5.set_title('New Plot 2')

        self.ax3.set_title("Reward per step")
        self.ax3.set_xlabel("step")
        self.ax3.set_ylabel("Reward")
        self.spokes_lines = []
        if self.render:
            self.init_plot()

    def update_text(self, epsilon, gamma, batch_size, explore_probability, epsilon_min,layers,model_type,learning_rate):

        # agent param
        self.explore_probability.set_text(f"expl-prob.: {explore_probability:.2f}")
        self.gamma.set_text(f"gamma: {gamma:.2f}")
        self.epsilon.set_text(f"epsilon: {epsilon:.2f}")
        self.epsilon_min.set_text(f"e_min: {epsilon_min:.2f}")
        self.batch_size.set_text(f"batch.s: {batch_size:.2f}")
        # model param
        self.layer_sizes.set_text(f"lay: {layers[0]:d},{layers[1]:d},{layers[2]:d}")
        self.model_type.set_text("Model type: "+model_type)
        self.learning_rate.set_text(f"learning.r: {learning_rate:.2e}")

    def init_plot(self):
        self.bars = self.ax1.bar(range(self.n_spokes), self.tensionchanges)
        self.ax1.set_ylim([-self.max_tension - 2, self.max_tension + 2])
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

    def update_episode_plot(self):
        self.ax6.clear()
        self.ax6.plot(self.reward_sums, label="Session")
        self.ax6.set_xlabel("Episode")
        self.ax6.set_ylabel("cum. Reward")
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
        self.previous_tensionchanges = np.copy(self.tensionchanges)
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
        self.reward_sums.append(np.sum(self.rewards))
        self.update_episode_plot()
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
        
        #this code does not work since tensionchanges = previous tension
        # above problem is now fixed
        # what is better? to repeadetly punish as long as the spoketension is to high or to only punish once?
        
        #if (self.max_tension <= max(abs(spoketension))) and (max(abs(self.previous_tensionchanges)) < max(
        #        abs(spoketension))):  
        #    reward = reward - 5000
        for spoke in spoketension:

            if (self.max_tension <= abs(spoke)):
                reward = reward - 5000
            

        next_state = tot_def.flatten()

        done = False  # decide when an episode is done

        return next_state, reward, done