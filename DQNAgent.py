import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
import copy
import pickle
from maps.SumoEnv import SumoEnv 
import time
class DqnAgent:
    def __init__(self, observation_space_n):
        self.observation_space_n = observation_space_n

        # Set the device      
        device = "mps" if torch.backends.mps.is_available() else "cpu"
        print(f"Using device: {device}")

        # Set random seed
        random_seed = 33
        torch.manual_seed(random_seed)
        np.random.seed(random_seed)
        random.seed(random_seed)

        # Define the neural network
        self.policy_network = self._initialize_network()
        self.target_network = copy.deepcopy(self.policy_network)

        # Environment and replay buffer setup
        self.highway_flow = 5000
        self.ramp_flow = 2000
        self.environment = SumoEnv(gui=False, flow_on_HW=self.highway_flow, flow_on_Ramp=self.ramp_flow) 
        self.state_buffer = deque(maxlen=3)
        for _ in range(3):
            state_matrix = [[0 for _ in range(251)] for _ in range(4)]
            self.state_buffer.appendleft(state_matrix)

        # Traffic flow data for simulation
        self.traffic_flow_data = [(t * 60, hw, ramp) for t, hw, ramp in [
            (0, 1000, 500), (10, 2000, 1300), (20, 3200, 1800),
            (30, 2500, 1500), (40, 1500, 1000), (50, 1000, 700), (60, 800, 500)
        ]]

        # Simulation and training parameters
        self.simulation_step_length = 60
        self.mu, self.omega, self.tau = 0.1, -0.4, 0.05  # mu: speed on HW, omega: waiting vehicles at TL, tau: speed on ramp

        self.epochs, self.batch_size = 40, 32
        self.max_steps = 3600 / self.simulation_step_length
        self.learning_rate, self.gamma = 5e-5, 0.99
        self.eps_start, self.eps_min = 0.8, 0.05
        self.eps_decay_factor, self.sync_frequency = 0.05, 5
        self.eps_decay_exponential = True

        # Optimizer, loss function, and experience replay
        self.optimizer = optim.Adam(self.policy_network.parameters(), lr=self.learning_rate)
        self.loss_function = nn.MSELoss()
        self.replay_buffer_size = 50000
        self.replay_buffer = deque(maxlen=self.replay_buffer_size)

    def _initialize_network(self):
        """Create the neural network model."""
        input_size, layer1, layer2, layer3, layer4 = self.observation_space_n, 128, 64, 32, 8
        model = nn.Sequential(
            nn.Linear(input_size, layer1), nn.ReLU(),
            nn.Linear(layer1, layer2), nn.ReLU(),
            nn.Linear(layer2, layer3), nn.ReLU(),
            nn.Linear(layer3, layer4), nn.ReLU(),
            nn.Linear(layer4, 1), nn.Sigmoid()  # Output single continuous action value in [0, 1]
        )
        return model

    def observe_state(self):
        """Retrieve the current state from the environment."""
        state_matrix = self.environment.getStateMatrixV2()
        self.state_buffer.appendleft(state_matrix)
        flat_state_array = np.concatenate(self.state_buffer).flatten()
        return torch.from_numpy(flat_state_array).float()

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
            # print(f"Light proportions: {action}")
            self.environment.doSimulationStep(action)
            # print(f"Traffic light status: {self.environment.getTrafficLightState()}")

    def reset_environment(self):
        """Reset the environment and state buffer."""
        for _ in range(3):
            state_matrix = [[0 for _ in range(251)] for _ in range(4)]
            self.state_buffer.appendleft(state_matrix)
        self.environment.reset()

    def train_agent(self):
        """Train the DQN agent using the defined environment and parameters."""
        total_losses, total_rewards, total_steps = [], [], 0
        for epoch in range(self.epochs):
            print("Epoch:", epoch)
            epsilon = self._update_epsilon(epoch)
            self.reset_environment()
            state = self.observe_state()
            is_done = False
            while not is_done:
                total_steps += 1

                # Select action
                q_value = self.policy_network(state)
                action = q_value.item() if random.random() >= epsilon else random.uniform(0, 1)
                self.perform_step(action)

                next_state = self.observe_state()
                reward = self.calculate_reward()
                total_rewards.append(reward)

                # Store experience
                experience = (state, action, reward, next_state, False)
                self.replay_buffer.append(experience)
                state = next_state

                # Train if buffer has enough samples
                if len(self.replay_buffer) > self.batch_size:
                    minibatch = random.sample(self.replay_buffer, self.batch_size)
                    self._train_step(minibatch)

                if total_steps % self.sync_frequency == 0:
                    self.target_network.load_state_dict(self.policy_network.state_dict())

                if total_steps >= self.max_steps:
                    is_done = True

        return self.policy_network, np.array(total_losses), np.array(total_rewards)

    def _train_step(self, minibatch):
        """Train the model on a mini-batch of experiences."""
        state_batch = torch.cat([s1.unsqueeze(0) for (s1, a, r, s2, d) in minibatch])
        action_batch = torch.Tensor([a for (s1, a, r, s2, d) in minibatch])
        reward_batch = torch.Tensor([r for (s1, a, r, s2, d) in minibatch])
        next_state_batch = torch.cat([s2.unsqueeze(0) for (s1, a, r, s2, d) in minibatch])
        done_batch = torch.Tensor([d for (s1, a, r, s2, d) in minibatch])

        Q1 = self.policy_network(state_batch).squeeze()
        with torch.no_grad():
            Q2 = self.target_network(next_state_batch).squeeze()

        target = reward_batch + self.gamma * ((1 - done_batch) * Q2)
        loss = self.loss_function(Q1, target.detach())

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def test_agent(self, model, gui=True):
        """Test the trained model in the environment."""
        if gui:
            self.environment.close()
            self.environment = SumoEnv(gui=True, flow_on_HW=self.highway_flow, flow_on_Ramp=self.ramp_flow)
        self.reset_environment()
        state = self.observe_state()
        is_done = False
        start_time = time.time()
        time_limit = 30 * 60  # 30 phút (1800 giây)
        while not is_done:
            q_value = model(state)
            action = np.argmax(q_value.data.numpy())
            self.perform_step(action)
            state = self.observe_state()
            if time.time() - start_time > time_limit:
                is_done = True
        self.environment.close()
        return self.environment.getStatistics()

    def _update_epsilon(self, current_epoch):
        """Update epsilon value for exploration-exploitation balance."""
        if self.eps_decay_exponential:
            return self.eps_min + (self.eps_start - self.eps_min) * np.exp(-self.eps_decay_factor * current_epoch)
        else:
            decay_rate = (self.eps_start - self.eps_min) / self.epochs
            return max(self.eps_min, self.eps_start - decay_rate * current_epoch)

    def _interpolate_traffic_flow(self, step, data_points):
        """Interpolate traffic flow values based on the current step."""
        times, hw_flows, ramp_flows = zip(*data_points)
        hw_flow = np.interp(step, times, hw_flows)
        ramp_flow = np.interp(step, times, ramp_flows)
        return int(hw_flow), int(ramp_flow)

# Main script
if __name__ == "__main__":
    agent = DqnAgent(observation_space_n=3012)
    trained_model, losses, rewards = agent.train_agent()

    # Save training results and model
    results = {
        "model": trained_model,
        "losses": losses,
        "rewards": rewards
    }
    with open('training_results.pkl', 'wb') as file:
        pickle.dump(results, file)
        print("Training results saved successfully.")
    torch.save(trained_model, 'Models/DQNModel.pth')
