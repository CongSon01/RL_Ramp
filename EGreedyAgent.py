import torch
import numpy as np
import random
from collections import deque
import copy
import pickle
from maps.SumoEnv import SumoEnv

class EGreedyAgent:
    def __init__(self, observation_space_n, action_space_n):
        """
        Initialize the E-Greedy Agent.

        Parameters:
        observation_space_n (int): The size of the observation space, representing the input to the Q-table.
        action_space_n (int): The number of possible actions in the environment.
        """
        self.observation_space_n = observation_space_n
        self.action_space_n = action_space_n

        # Set random seed
        random_seed = 33
        np.random.seed(random_seed)
        random.seed(random_seed)

        # Environment setup
        self.highway_flow = 5000
        self.ramp_flow = 2000
        self.environment = SumoEnv(gui=False, flow_on_HW=self.highway_flow, flow_on_Ramp=self.ramp_flow)

        # State (1 matrix)
        self.state = np.zeros((4, 251))  # Adjust based on your state dimensions (4 x 251 as example)

        # Traffic flow data for simulation
        self.traffic_flow_data = [(t * 60, hw, ramp) for t, hw, ramp in [
            (0, 1000, 500), (10, 2000, 1300), (20, 3200, 1800),
            (30, 2500, 1500), (40, 1500, 1000), (50, 1000, 700), (60, 800, 500)
        ]]  

        # Simulation and training parameters
        self.simulation_step_length = 60
        self.mu, self.omega, self.tau = 0.1, -0.4, 0.05  # mu: speed on HW, omega: waiting vehicles at TL, tau: speed on ramp
        self.epochs, self.max_steps, self.learning_rate = 40, 3600 // self.simulation_step_length, 0.1
        self.eps_start, self.eps_min = 0.8, 0.05
        self.eps_decay_factor = 0.05
        self.batch_size = 32

        # Q-table initialization
        self.q_table = np.zeros((self.observation_space_n, self.action_space_n))

    def observe_state(self):
        """Retrieve the current state from the environment."""
        state_matrix = self.environment.getStateMatrixV2()
        flat_state_array = np.concatenate(state_matrix).flatten()
        return int(np.sum(flat_state_array))

    def calculate_reward(self):
        """Calculate reward based on environment metrics."""
        return (self.mu * self.environment.getSpeedHW() +
                self.omega * self.environment.getNumberVehicleWaitingTL() +
                self.tau * self.environment.getSpeedRamp())

    def perform_step(self, action):
        """Execute a simulation step with the given action."""
        for _ in range(self.simulation_step_length):
            hw_flow, ramp_flow = self._interpolate_traffic_flow(self.environment.getCurrentStep(), self.traffic_flow_data)
            self.environment.setFlowOnHW(hw_flow)
            self.environment.setFlowOnRamp(ramp_flow)
            self.environment.doSimulationStep(action)

    def reset_environment(self):
        """Reset the environment and state."""
        self.state = np.zeros((4, 251))  # Reset state to default values
        self.environment.reset()

    def train_agent(self):
        """Train the E-Greedy agent using the defined environment and parameters."""
        total_rewards = []
        for epoch in range(self.epochs):
            print(f"Epoch: {epoch}")
            epsilon = self._update_epsilon(epoch)
            self.reset_environment()
            state = self.observe_state()

            is_done = False
            while not is_done:
                # Select action using epsilon-greedy policy
                if random.random() > epsilon:
                    action = np.argmax(self.q_table[state])  # Exploitation: choose the best action
                else:
                    action = random.randint(0, self.action_space_n - 1)  # Exploration: choose random action

                # Perform action in the environment
                self.perform_step(action)

                # Observe next state and calculate reward
                next_state = self.observe_state()
                reward = self.calculate_reward()
                total_rewards.append(reward)

                # Cập nhật bảng Q
                self.q_table[state, action] = self.q_table[state, action] + self.learning_rate * (reward + np.max(self.q_table[next_state]) - self.q_table[state, action])

                state = next_state
                if self.environment.step >= self.max_steps:
                    is_done = True

        return self.q_table, np.array(total_rewards)

    def _update_epsilon(self, current_epoch):
        """Update epsilon value for exploration-exploitation balance."""
        return self.eps_min + (self.eps_start - self.eps_min) * np.exp(-self.eps_decay_factor * current_epoch)

    def _interpolate_traffic_flow(self, step, data_points):
        """Interpolate traffic flow values based on the current step."""
        times, hw_flows, ramp_flows = zip(*data_points)
        hw_flow = np.interp(step, times, hw_flows)
        ramp_flow = np.interp(step, times, ramp_flows)
        return int(hw_flow), int(ramp_flow)

    def save_model(self, file_path):
        """Save the Q-table model to a file."""
        torch.save(self.q_table, file_path)
        print(f"Model saved to {file_path}")

    def load_model(self, file_path):
        """Load the Q-table model from a file."""
        self.q_table = torch.load(file_path)
        print(f"Model loaded from {file_path}")

# Main script
if __name__ == "__main__":
    agent = EGreedyAgent(observation_space_n=1004, action_space_n=5)  # Adjust observation and action space
    q_table, rewards = agent.train_agent()

    # Save training results and Q-table
    results = {
        "q_table": q_table,
        "rewards": rewards
    }

    # Save Q-table to .pth file
    model_file_path = 'egreedy.pth'
    agent.save_model(model_file_path)

    with open('training_results_egreedy.pkl', 'wb') as file:
        pickle.dump(results, file)
        print("Training results saved successfully.")
