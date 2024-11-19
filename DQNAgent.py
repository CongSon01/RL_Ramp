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
        self.mu, self.omega, self.tau = 0.05, -0.6, 0.2
        self.epochs, self.batch_size = 20, 32
        self.max_steps = 3600 / self.simulationStepLength
        self.learning_rate, self.gamma = 5e-5, 0.99
        self.eps_start, self.eps_min = 0.8, 0.05
        self.eps_decay_factor, self.sync_freq = 0.05, 5
        self.eps_dec_exp = True

        # Optimizer, loss function, and experience replay
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.loss_fn = nn.MSELoss()
        self.mem_size = 50000
        self.replay = deque(maxlen=self.mem_size)

    def create_model(self):
        l1, l2, l3, l4, l5 = self.observation_space_n, 128, 64, 32, 8
        model = nn.Sequential(
            nn.Linear(l1, l2), nn.ReLU(),
            nn.Linear(l2, l3), nn.ReLU(),
            nn.Linear(l3, l4), nn.ReLU(),
            nn.Linear(l4, l5), nn.ReLU(),
            nn.Linear(l5, 1), nn.Sigmoid()  # Output single continuous action value in [0, 1]
        )
        return model

    def obs(self):
        state_matrix = self.env.getStateMatrixV2()
        self.state_matrices.appendleft(state_matrix)
        flat_state_array = np.concatenate(self.state_matrices).flatten()
        return torch.from_numpy(flat_state_array).float()

    def rew(self):
        return self.mu * self.env.getSpeedHW() + self.omega * self.env.getNumberVehicleWaitingTL() + self.tau * self.env.getSpeedRamp()

    def step(self, action):
        for _ in range(self.simulationStepLength):
            hw_flow, ramp_flow = self.interpolate_flow(self.env.getCurrentStep(), self.data_points)
            self.env.setFlowOnHW(hw_flow)
            self.env.setFlowOnRamp(ramp_flow)
            self.env.doSimulationStep(action)

    def reset(self):
        for _ in range(3):
            state_matrix = [[0 for _ in range(251)] for _ in range(4)]
            self.state_matrices.appendleft(state_matrix)
        self.env.reset()

    def train(self):
        total_step_loss, total_step_rewards, total_steps = [], [], 0
        for i in range(self.epochs):
            print("Epoch:", i)
            epsilon = self.update_epsilon(i)
            update_count, mov = 0, 0
            self.reset()
            state1 = self.obs()
            isDone = False

            while not isDone:
                total_steps += 1
                mov += 1

                # Action selection
                qval = self.model(state1)
                action_ = qval.item() if random.random() >= epsilon else random.uniform(0, 1)
                
                self.step(action_)
                state2 = self.obs()
                reward = self.rew()
                print(f"____________________reward___________________: {reward}")
                total_step_rewards.append(reward)

                done = False
                exp = (state1, action_, reward, state2, done)
                self.replay.append(exp)
                state1 = state2

                # Training with mini-batch
                if len(self.replay) > self.batch_size:
                    minibatch = random.sample(self.replay, self.batch_size)
                    state1_batch = torch.cat([s1.unsqueeze(0) for (s1, a, r, s2, d) in minibatch])
                    action_batch = torch.Tensor([a for (s1, a, r, s2, d) in minibatch])
                    reward_batch = torch.Tensor([r for (s1, a, r, s2, d) in minibatch])
                    state2_batch = torch.cat([s2.unsqueeze(0) for (s1, a, r, s2, d) in minibatch])
                    done_batch = torch.Tensor([d for (s1, a, r, s2, d) in minibatch])

                    Q1 = self.model(state1_batch).squeeze()
                    with torch.no_grad():
                        Q2 = self.target_model(state2_batch).squeeze()

                    Y = reward_batch + self.gamma * ((1 - done_batch) * Q2)
                    loss = self.loss_fn(Q1, Y.detach())

                    self.optimizer.zero_grad()
                    loss.backward()
                    print(f"____________________loss___________________: {loss.item()}")
                    total_step_loss.append(loss.item())
                    update_count += 1
                    self.optimizer.step()

                    if total_steps % self.sync_freq == 0:
                        self.target_model.load_state_dict(self.model.state_dict())

                if done or mov >= self.max_steps:
                    isDone = True
            
        return self.model, np.array(total_step_loss), np.array(total_step_rewards), self.epochs, total_steps, self.simulationStepLength, self.mu, self.omega, self.tau, self.epochs, self.batch_size, self.max_steps, self.learning_rate, self.gamma, self.eps_start, self.eps_min, self.eps_decay_factor, self.eps_dec_exp, self.sync_freq, self.mem_size



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


    def update_epsilon(self, current_epoch):
        if self.eps_dec_exp:
            return self.eps_min + (self.eps_start - self.eps_min) * np.exp(-self.eps_decay_factor * current_epoch)
        else:
            decay_rate = (self.eps_start - self.eps_min) / self.epochs
            return max(self.eps_min, self.eps_start - decay_rate * current_epoch)

    def interpolate_flow(self, step, data_points):
        times, hw_flows, ramp_flows = zip(*data_points)
        hw_flow = np.interp(step, times, hw_flows)
        ramp_flow = np.interp(step, times, ramp_flows)
        return int(hw_flow), int(ramp_flow)

# Main script
if __name__ == "__main__":
    agent = DqnAgent(observation_space_n=3012)
    model, total_step_loss, total_step_rewards, epochs, steps, simulationStepLength, mu, omega, tau, epochs, batch_size, max_steps, learning_rate, gamma, eps_start, eps_min, eps_dec_factor, eps_dec_exp, sync_freq, mem_size = agent.train()

    # Save training results and model
    results = { "agent": agent, "model": model, "total_step_loss": total_step_loss, "total_step_rewards": total_step_rewards, "epochs": epochs, "steps": steps, "simulationStepLength": simulationStepLength, "mu": mu, "omega": omega, "tau": tau, "batch_size": batch_size, "max_steps": max_steps, "learning_rate": learning_rate, "gamma": gamma, "eps_start": eps_start, "eps_min": eps_min, "eps_dec_factor": eps_dec_factor, "eps_dec_exp": eps_dec_exp, "sync_freq": sync_freq, "mem_size": mem_size } 
    with open('training_results_1.pkl', 'wb') as file:
        pickle.dump(results, file)
        print("Training results saved successfully.")
    torch.save(model, 'Models/DynamicModel2.pth')