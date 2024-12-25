import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
import copy
import pickle
from maps.SumoEnv import SumoEnv

class DqnAgent:
    def __init__(self, observation_space_n):
        """
        Initialize the DQN Agent.
        
        Parameters:
        observation_space_n (int): The size of the observation space, representing the input to the neural network.
        """
        self.observation_space_n = observation_space_n

        # Set the computing device (MPS for Mac GPUs or CPU as fallback)
        device = "mps" if torch.backends.mps.is_available() else "cpu"
        print(f"Using device: {device}")

        # Set random seed for reproducibility
        random_seed = 33
        torch.manual_seed(random_seed)
        np.random.seed(random_seed)
        random.seed(random_seed)

        # Define and initialize the neural networks
        self.policy_network = self._initialize_network()
        self.target_network = copy.deepcopy(self.policy_network)  # Target network for stability

        # Environment setup with traffic flow parameters
        self.highway_flow = 5000
        self.ramp_flow = 2000
        self.environment = SumoEnv(gui=False, flow_on_HW=self.highway_flow, flow_on_Ramp=self.ramp_flow)

        # Initialize state buffer to store the most recent states
        self.state_buffer = deque(maxlen=3)
        for _ in range(3):  # Pre-fill the buffer with empty states
            state_matrix = [[0 for _ in range(251)] for _ in range(4)]
            self.state_buffer.appendleft(state_matrix)

        # Traffic flow data for interpolation during simulation
        self.traffic_flow_data = [(t * 60, hw, ramp) for t, hw, ramp in [
            (0, 1000, 500), (10, 2000, 1300), (20, 3200, 1800),
            (30, 2500, 1500), (40, 1500, 1000), (50, 1000, 700), (60, 800, 500)
        ]]

        # Simulation and training hyperparameters
        self.simulation_step_length = 60  # Simulation step length in seconds
        self.mu, self.omega, self.tau = 0.1, -0.4, 0.05  # Reward coefficients for metrics
        self.epochs, self.batch_size = 40, 32  # Training parameters
        self.max_steps = 3600 / self.simulation_step_length  # Maximum simulation steps
        self.learning_rate, self.gamma = 5e-5, 0.99  # Learning rate and discount factor
        self.eps_start, self.eps_min = 0.8, 0.05  # Epsilon-greedy parameters
        self.eps_decay_factor, self.sync_frequency = 0.05, 5  # Decay factor and sync frequency
        self.eps_decay_exponential = True  # Use exponential decay for epsilon

        # Optimizer and loss function for training
        self.optimizer = optim.Adam(self.policy_network.parameters(), lr=self.learning_rate)
        self.loss_function = nn.MSELoss()

        # Replay buffer for storing experiences
        self.replay_buffer_size = 50000
        self.replay_buffer = deque(maxlen=self.replay_buffer_size)

    def _initialize_network(self):
        """
        Define the neural network architecture.

        Returns:
        nn.Sequential: The constructed neural network model.
        """
        input_size, layer1, layer2, layer3, layer4 = self.observation_space_n, 128, 64, 32, 8
        model = nn.Sequential(
            nn.Linear(input_size, layer1), nn.ReLU(),
            nn.Linear(layer1, layer2), nn.ReLU(),
            nn.Linear(layer2, layer3), nn.ReLU(),
            nn.Linear(layer3, layer4), nn.ReLU(),
            nn.Linear(layer4, 1), nn.Sigmoid()  # Output a single continuous action value in [0, 1]
        )
        return model

    def observe_state(self):
        """
        Retrieve and preprocess the current state from the environment.

        Returns:
        torch.Tensor: Flattened and processed state tensor.
        """
        state_matrix = self.environment.getStateMatrixV2()  # Retrieve state matrix from the environment
        self.state_buffer.appendleft(state_matrix)  # Add the new state to the buffer
        flat_state_array = np.concatenate(self.state_buffer).flatten()  # Flatten the buffer into a 1D array
        return torch.from_numpy(flat_state_array).float()

    def calculate_reward(self):
        """
        Compute the reward using environment metrics and predefined coefficients.

        Returns:
        float: Calculated reward value.
        """
        return (self.mu * self.environment.getSpeedHW() +
                self.omega * self.environment.getNumberVehicleWaitingTL() +
                self.tau * self.environment.getSpeedRamp())

    def perform_step(self, action):
        """
        Execute a simulation step with the given action.

        Parameters:
        action (float): The action value for controlling the traffic light proportions.
        """
        for _ in range(self.simulation_step_length):
            hw_flow, ramp_flow = self._interpolate_traffic_flow(self.environment.getCurrentStep(), self.traffic_flow_data)
            self.environment.setFlowOnHW(hw_flow)
            self.environment.setFlowOnRamp(ramp_flow)
            print(f"Light proportions: {action}")
            self.environment.doSimulationStep(action)
            print(f"Traffic light status: {self.environment.getTrafficLightState()}")

    def reset_environment(self):
        """
        Reset the environment and state buffer to initial conditions.
        """
        for _ in range(3):
            state_matrix = [[0 for _ in range(251)] for _ in range(4)]
            self.state_buffer.appendleft(state_matrix)
        self.environment.reset()

    def train_agent(self):
        """
        Train the DQN agent using the defined environment and hyperparameters.

        Returns:
        tuple: Trained policy network, array of loss values, and array of rewards.
        """
        total_losses, total_rewards, total_steps = [], [], 0
        for epoch in range(self.epochs):
            print("Epoch:", epoch)
            epsilon = self._update_epsilon(epoch)  # Update exploration parameter
            self.reset_environment()
            state = self.observe_state()
            is_done = False
            while not is_done:
                total_steps += 1

                # Select action using epsilon-greedy strategy
                q_value = self.policy_network(state)
                action = q_value.item() if random.random() >= epsilon else random.uniform(0, 1)
                self.perform_step(action)

                next_state = self.observe_state()
                reward = self.calculate_reward()
                total_rewards.append(reward)

                # Store experience in the replay buffer
                experience = (state, action, reward, next_state, False)
                self.replay_buffer.append(experience)
                state = next_state

                # Train the policy network if enough experiences are available
                if len(self.replay_buffer) > self.batch_size:
                    minibatch = random.sample(self.replay_buffer, self.batch_size)
                    self._train_step(minibatch)

                # Sync the target network periodically
                if total_steps % self.sync_frequency == 0:
                    self.target_network.load_state_dict(self.policy_network.state_dict())

                if total_steps >= self.max_steps:
                    is_done = True

        return self.policy_network, np.array(total_losses), np.array(total_rewards)

    def _train_step(self, minibatch):
        """
        Perform a single training step on a mini-batch of experiences.

        Parameters:
        minibatch (list): A list of sampled experiences from the replay buffer.
        """
        state_batch = torch.cat([s1.unsqueeze(0) for (s1, a, r, s2, d) in minibatch])
        action_batch = torch.Tensor([a for (s1, a, r, s2, d) in minibatch])
        reward_batch = torch.Tensor([r for (s1, a, r, s2, d) in minibatch])
        next_state_batch = torch.cat([s2.unsqueeze(0) for (s1, a, r, s2, d) in minibatch])
        done_batch = torch.Tensor([d for (s1, a, r, s2, d) in minibatch])

        # Compute current Q-values and target Q-values
        Q1 = self.policy_network(state_batch).squeeze()
        with torch.no_grad():
            Q2 = self.target_network(next_state_batch).squeeze()

        target = reward_batch + self.gamma * ((1 - done_batch) * Q2)
        loss = self.loss_function(Q1, target.detach())

        # Backpropagate the loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def test_agent(self, model, gui=True):
        """
        Test the trained model in the environment.

        Parameters:
        model (nn.Module): The trained policy network to be tested.
        gui (bool): Whether to display the GUI during testing.

        Returns:
        dict: Statistics from the environment after testing.
        """
        if gui:
            self.environment.close()
            self.environment = SumoEnv(gui=True, flow_on_HW=self.highway_flow, flow_on_Ramp=self.ramp_flow)
        self.reset_environment()
        state = self.observe_state()
        is_done = False
        while not is_done:
            q_value = model(state)
            action = np.argmax(q_value.data.numpy())
            self.perform_step(action)
            state = self.observe_state()

        self.environment.close()
        return self.environment.getStatistics()

    def _update_epsilon(self, current_epoch):
        """
        Update the epsilon value for the epsilon-greedy policy.

        Parameters:
        current_epoch (int): The current epoch in the training process.

        Returns:
        float: The updated epsilon value.
        """
        if self.eps_decay_exponential:
            return self.eps_min + (self.eps_start - self.eps_min) * np.exp(-self.eps_decay_factor * current_epoch)
        else:
            decay_rate = (self.eps_start - self.eps_min) / self.epochs
            return max(self.eps_min, self.eps_start - decay_rate * current_epoch)

    def _interpolate_traffic_flow(self, step, data_points):
        """
        Interpolate traffic flow values based on the current simulation step.

        Parameters:
        step (int): The current simulation step.
        data_points (list): List of tuples representing traffic flow data points.

        Returns:
        tuple: Interpolated highway and ramp traffic flows.
        """
        times, hw_flows, ramp_flows = zip(*data_points)
        hw_flow = np.interp(step, times, hw_flows)
        ramp_flow = np.interp(step, times, ramp_flows)
        return int(hw_flow), int(ramp_flow)

# Main script
if __name__ == "__main__":
    agent = DqnAgent(observation_space_n=3012)  # Initialize the DQN agent
    trained_model, losses, rewards = agent.train_agent()  # Train the agent

    # Save training results and model
    results = {
        "model": trained_model,
        "losses": losses,
        "rewards": rewards
    }
    with open('training_results.pkl', 'wb') as file:
        pickle.dump(results, file)
        print("Training results saved successfully.")
    torch.save(trained_model, 'Models/TrainedModel.pth')  # Save the trained model