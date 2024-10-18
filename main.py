import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
import copy
import os
import sys

from maps.SumoEnv import SumoEnv 

class SumoDQNAgent:
    def __init__(self, action_space_n, observation_space_n):
        self.action_space_n = action_space_n
        self.observation_space_n = observation_space_n

        # Set the device      
        device = "mps" if torch.backends.mps.is_available() else "cpu"
        print(f"Using device: {device}")

        # Set random seed
        randomSeed = 33
        torch.manual_seed(randomSeed)
        np.random.seed(randomSeed)
        random.seed(randomSeed)

        # Define the neural network
        self.model = self.create_model()
        self.target_model = copy.deepcopy(self.model)

        # Creating an instance of the environment
        self.flow_on_HW = 5000
        self.flow_on_Ramp = 2000
        self.env = SumoEnv(gui=False, flow_on_HW = self.flow_on_HW, flow_on_Ramp = self.flow_on_Ramp) 
        self.state_matrices = deque(maxlen=3) # Create queue for the obs matrices (3 DQN steps included)

        ### Optional to collect the reward over several steps ###
        # self.rewardArray = deque(maxlen=3) # Create queue for the rewards (3 DQN steps included)
        # self.rewardArray.extend([0, 0, 0])
        # self.simStepReward = 0 # Reward für einen Simulationsschritt

        # Initialize the deque with state matrices (all zeros)
        for _ in range(3):
            state_matrix = [[0 for _ in range(251)] for _ in range(4)]
            self.state_matrices.appendleft(state_matrix)

        # Traffic flow data for the simulation
        self.data_points = [
            (0, 1000, 500),    # Data point at 7:50 a.m.
            (10, 2000, 1300),  # Data point at 8:00 a.m.
            (20, 3200, 1800),  # Data point at 8:10 a.m.
            (30, 2500, 1500),  # Data point at 8:20 a.m.
            (40, 1500, 1000),  # Data point at 8:30 a.m.
            (50, 1000, 700),   # Data point at 8:40 a.m.
            (60, 800, 500),    # Data point at 8:50 a.m.
        ]
        # Conversion of time from minutes to steps (assumes 1 step = 1 second)
        self.data_points = [(t * 60, hw, ramp) for t, hw, ramp in self.data_points]

        ### Set the simulation parameters ###
        self.simulationStepLength = 2 # 2 seconds per simulation step (time that elapses in the simulation per DQN step)
        self.mu = 0.05 # Reward weighting for the speed on the HW
        self.omega = -0.5 # Reward weighting for the traffic light queue length
        self.tau = 0.2 # Reward weighting for the speed on the ramp

        ### Set hyperparameters ###
        self.epochs = 70
        self.batch_size = 32
        self.max_steps = 3600 / self.simulationStepLength # 1 hour traffic simulation

        self.learning_rate = 5e-5
        self.gamma = 0.99

        self.eps_start = 0.8                   # Epsilon start
        self.eps_min = 0.05                    # Epsilon min
        self.eps_dec_exp = True                # Exponential
        self.eps_decay_factor = 0.05           # Epsilon decay factor

        self.sync_freq = 5
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.loss_fn = nn.MSELoss()

        # Experience repetition
        self.mem_size = 50000
        self.replay = deque(maxlen=self.mem_size)

    def create_model(self):
        # Define the network architecture
        l1 = self.observation_space_n
        l2, l3, l4, l5 = 128, 64, 32, 8, 
        l6 = self.action_space_n
        model = nn.Sequential(
            nn.Linear(l1, l2),
            nn.ReLU(),
            nn.Linear(l2, l3),
            nn.ReLU(),
            nn.Linear(l3, l4),
            nn.ReLU(),
            nn.Linear(l4, l5),
            nn.ReLU(),
            nn.Linear(l5, l6)
        )
        return model


    def obs(self):
        # Get the state matrix
        state_matrix = self.env.getStateMatrixV2()

        # Add the state matrix to the queue
        self.state_matrices.appendleft(state_matrix)

        # Convert the matrices into a flat NumPy array
        flat_state_array = np.concatenate(self.state_matrices).flatten()

        # Create a PyTorch tensor from the flat NumPy array
        obs_tensor = torch.from_numpy(flat_state_array).float()

        return obs_tensor

    def rew(self):
        reward = self.mu * self.env.getSpeedHW() + self.omega * self.env.getNumberVehicleWaitingTL() + self.tau * self.env.getSpeedRamp()

        ### Optional: Calculation of rewards over several simulation steps ###
        #reward = self.simStepReward / self.simulationStepLength
        #self.simStepReward = 0

        return reward

    def done(self):
        # Optional: Since we use max_steps
        return False

    def info(self):
        # Optional: additional log information
        return {}

    def reset(self):
        # Reset the environment and state_matrix
        for _ in range(3):
            state_matrix = [[0 for _ in range(251)] for _ in range(4)]
            self.state_matrices.appendleft(state_matrix)

        self.env.reset()

    def step(self, action):
        # Execute the action in the surrounding area
        for _ in range(self.simulationStepLength):

            ### Optional: Dynamically adjust the flow ###
            self.env.setFlowOnHW(self.interpolate_flow(self.env.getCurrentStep(), self.data_points)[0])
            self.env.setFlowOnRamp(self.interpolate_flow(self.env.getCurrentStep(), self.data_points)[1])
            self.env.doSimulationStep(action)

            ### Optional: Calculation of rewards over several simulation steps ###
            #self.simStepReward = self.mu * self.env.getSpeedHW() + self.omega * self.env.getNumberVehicleWaitingTL()

    def train(self):
        total_step_loss = []
        total_step_rewards = []
        total_steps = 0
        for i in range(self.epochs):
            print("Epoch: ", i)
            epsilon = self.update_epsilon(i)
            # total_loss = 0
            # total_reward = 0 
            update_count = 0
            self.reset()
            state1 = self.obs()
            isDone = False
            mov = 0
            while not isDone:
                total_steps += 1
                mov += 1
                qval = self.model(state1)
                qval_ = qval.data.numpy()
                if random.random() < epsilon:
                    action_ = np.random.randint(0, self.action_space_n)
                else:
                    action_ = np.argmax(qval_)

                self.step(action_)
                state2 = self.obs()
                reward = self.rew()
                total_step_rewards.append(reward)
                # total_reward += reward
                done = self.done()
                exp = (state1, action_, reward, state2, done)
                self.replay.append(exp)
                state1 = state2

                if len(self.replay) > self.batch_size:
                    minibatch = random.sample(self.replay, self.batch_size)
                    state1_batch = torch.cat([s1.unsqueeze(0) for (s1, a, r, s2, d) in minibatch])
                    action_batch = torch.Tensor([a for (s1, a, r, s2, d) in minibatch])
                    reward_batch = torch.Tensor([r for (s1, a, r, s2, d) in minibatch])
                    state2_batch = torch.cat([s2.unsqueeze(0) for (s1, a, r, s2, d) in minibatch])
                    done_batch = torch.Tensor([d for (s1, a, r, s2, d) in minibatch])

                    Q1 = self.model(state1_batch)
                    with torch.no_grad():
                        Q2 = self.target_model(state2_batch)

                    Y = reward_batch + self.gamma * ((1 - done_batch) * torch.max(Q2, dim=1)[0])
                    X = Q1.gather(dim=1, index=action_batch.long().unsqueeze(dim=1)).squeeze()
                    loss = self.loss_fn(X, Y.detach())

                    self.optimizer.zero_grad()
                    loss.backward()
                    # total_loss += loss.item()
                    total_step_loss.append(loss.item())
                    update_count += 1
                    self.optimizer.step()

                    if total_steps % self.sync_freq == 0:
                        self.target_model.load_state_dict(self.model.state_dict())

                if done or mov >= self.max_steps:
                    isDone = True
            
            # total_step_loss.append(total_loss / max(1, update_count))  # Durchschnittlicher Verlust pro Aktualisierung
            # total_step_rewards.append(total_reward)  # Durchschnittlicher Reward pro Bewegung
        print( total_steps)
        return self.model, np.array(total_step_loss), np.array(total_step_rewards), self.epochs, total_steps, self.simulationStepLength, self.mu, self.omega, self.tau, self.epochs, self.batch_size, self.max_steps, self.learning_rate, self.gamma, self.eps_start, self.eps_min, self.eps_decay_factor, self.eps_dec_exp, self.sync_freq, self.mem_size
    
    def update_epsilon(self, current_epoch):
        if self.eps_dec_exp:
            # Exponential decrease
            return self.eps_min + (self.eps_start - self.eps_min) * np.exp(-self.eps_decay_factor * current_epoch)
        else:
            # Linear decrease
            decay_rate = (self.eps_start - self.eps_min) / self.epochs  # Acceptance rate per epoch
            x = max(self.eps_min, self.eps_start - decay_rate * current_epoch)
            return x
        
    def testModel(self, model, gui=True, useModel=True):
        if gui:
            self.env.close()
            self.env = SumoEnv(gui=True, flow_on_HW = self.flow_on_HW, flow_on_Ramp = self.flow_on_Ramp) 
        self.reset()
        state1 = self.obs()
        isDone = False
        mov = 0
        while not isDone:
            mov += 1
            if useModel:
                qval = self.model(state1)
                qval = model(state1)
                qval_ = qval.data.numpy()
                action_ = np.argmax(qval_)
            else:
                action_ = 0
            self.step(action_)
            state1 = self.obs()
            if mov > self.max_steps:
                isDone = True

        self.env.close()
        return self.env.getStatistics() 

    # Function for linear interpolation of the flow data
    def interpolate_flow(self, step, data_points):

        times = [point[0] for point in data_points]
        hw_flows = [point[1] for point in data_points]
        ramp_flows = [point[2] for point in data_points]
        
        hw_flow = np.interp(step, times, hw_flows)
        ramp_flow = np.interp(step, times, ramp_flows)
        
        return int(hw_flow), int(ramp_flow)

# this is the main entry point of this script
if __name__ == "__main__":
    agent = SumoDQNAgent(action_space_n=2, observation_space_n=3012) # 3012 = 3*1004 (3 matrices with 1004 values each (4*251)

    model, total_step_loss, total_step_rewards, epochs, steps, simulationStepLength, mu, omega, tau, epochs, batch_size, max_steps, learning_rate, gamma, eps_start, eps_min, eps_dec_factor, eps_dec_exp, sync_freq, mem_size = agent.train()
    model.save("bodel.bin")

