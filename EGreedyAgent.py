import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
from maps.SumoEnv import SumoEnv 

class EGreedyAgent:
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
        self.model = self.create_model()

        # Environment and replay buffer setup
        self.flow_on_HW = 5000
        self.flow_on_Ramp = 2000
        self.env = SumoEnv(gui=False, flow_on_HW=self.flow_on_HW, flow_on_Ramp=self.flow_on_Ramp) 
        self.state_matrices = deque(maxlen=3)
        for _ in range(3):
            state_matrix = [[0 for _ in range(251)] for _ in range(4)]
            self.state_matrices.appendleft(state_matrix)

        # Traffic flow data for simulation
        self.data_points = [(t * 60, hw, ramp) for t, hw, ramp in [
            (0, 1000, 500), (10, 2000, 1300), (20, 3200, 1800),
            (30, 2500, 1500), (40, 1500, 1000), (50, 1000, 700), (60, 800, 500)
        ]]

        # Simulation and training parameters
        self.simulationStepLength = 60
        self.epsilon = 0.8
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.01
        self.gamma = 0.95
        self.learning_rate = 0.005
        self.epochs, self.batch_size = 50, 32
        self.max_steps = 3600 / self.simulationStepLength

        # Optimizer, loss function, and experience replay
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.loss_fn = nn.MSELoss()
        self.mem_size = 50000
        self.replay = deque(maxlen=self.mem_size)

    def create_model(self):
        l1, l2, l3 = self.observation_space_n, 64, 1
        model = nn.Sequential(
            nn.Linear(l1, l2), nn.ReLU(),
            nn.Linear(l2, l3), nn.Sigmoid()
        )
        return model

    def obs(self):
        state_matrix = self.env.getStateMatrixV2()
        self.state_matrices.appendleft(state_matrix)
        flat_state_array = np.concatenate(self.state_matrices).flatten()
        return torch.from_numpy(flat_state_array).float()

    def step(self, action):
        for _ in range(self.simulationStepLength):
            hw_flow, ramp_flow = self.interpolate_flow(self.env.getCurrentStep(), self.data_points)
            self.env.setFlowOnHW(hw_flow)
            self.env.setFlowOnRamp(ramp_flow)
            self.env.doSimulationStep(action)

    def train(self):
        for epoch in range(self.epochs):
            print(f"Epoch: {epoch}")
            self.reset()
            state1 = self.obs()
            done = False

            while not done:
                qval = self.model(state1)
                action_ = qval.item() if random.random() > self.epsilon else random.uniform(0, 1)

                self.step(action_)
                state2 = self.obs()
                reward = self.env.getReward()
                done = False

                exp = (state1, action_, reward, state2, done)
                self.replay.append(exp)
                state1 = state2

                if len(self.replay) > self.batch_size:
                    self.replay_train()

                self.epsilon = max(self.epsilon_min, self.epsilon * (1 - self.epsilon_decay))

    def replay_train(self):
        minibatch = random.sample(self.replay, self.batch_size)
        for state1, action_, reward, state2, done in minibatch:
            target = reward
            if not done:
                target += self.gamma * self.model(state2).item()
            output = self.model(state1)
            loss = self.loss_fn(output, torch.tensor([target]))

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    def reset(self):
        for _ in range(3):
            state_matrix = [[0 for _ in range(251)] for _ in range(4)]
            self.state_matrices.appendleft(state_matrix)
        self.env.reset()

    def interpolate_flow(self, step, data_points):
        times, hw_flows, ramp_flows = zip(*data_points)
        hw_flow = np.interp(step, times, hw_flows)
        ramp_flow = np.interp(step, times, ramp_flows)
        return int(hw_flow), int(ramp_flow)

if __name__ == "__main__":
    agent = EGreedyAgent(observation_space_n=3012)
    agent.train()
