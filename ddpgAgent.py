import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
import copy
import pickle
import csv
import matplotlib.pyplot as plt
from maps.SumoEnv import SumoEnv

class DdpAgent:
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

        # Define the neural network for the policy (actor) and the value (critic)
        self.actor_model = self.create_model()
        self.target_actor_model = copy.deepcopy(self.actor_model)
        self.critic_model = self.create_critic_model()
        self.target_critic_model = copy.deepcopy(self.critic_model)

        # Apply Xavier initialization
        self.initialize_weights(self.actor_model)
        self.initialize_weights(self.critic_model)

        # Environment setup
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

        # Simulation parameters
        self.simulationStepLength = 60  # Adjustable step length
        self.mu, self.omega, self.tau = 0.05, -0.6, 0.2
        self.epochs, self.batch_size = 20, 32
        self.max_steps = 3600 / self.simulationStepLength
        self.learning_rate = 1e-6
        self.gamma = 0.99
        self.sync_freq = 10  # Increased for better stability
        self.mem_size = 50000

        # Optimizer and loss function
        self.optimizer_actor = optim.Adam(self.actor_model.parameters(), lr=self.learning_rate)
        self.optimizer_critic = optim.Adam(self.critic_model.parameters(), lr=self.learning_rate)
        self.scheduler_actor = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer_actor, patience=5, factor=0.5, verbose=True)
        self.scheduler_critic = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer_critic, patience=5, factor=0.5, verbose=True)
        self.loss_fn = nn.MSELoss()

        # Experience replay
        self.replay = deque(maxlen=self.mem_size)

    def create_model(self):
        l1, l2, l3, l4 = self.observation_space_n, 128, 64, 32
        model = nn.Sequential(
            nn.Linear(l1, l2), nn.ReLU(),
            nn.Linear(l2, l3), nn.ReLU(),
            nn.Linear(l3, l4), nn.ReLU(),
            nn.Linear(l4, 1), nn.Sigmoid()  # Output continuous action in [0, 1]
        )
        return model

    def create_critic_model(self):
        # Critic model for estimating Q-value
        return nn.Sequential(
            nn.Linear(self.observation_space_n + 1, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, 1)  # Output Q-value
        )

    def obs(self):
        # Get the state matrix from the environment and flatten it
        state_matrix = self.env.getStateMatrixV2()
        self.state_matrices.appendleft(state_matrix)
        flat_state_array = np.concatenate(self.state_matrices).flatten()
        return torch.from_numpy(flat_state_array).float()

    def initialize_weights(self, model):
        for layer in model:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)

    def rew(self):
        # Retrieve normalized values for each component
        speed_hw_reward = self.normalize_speed(self.env.getSpeedHW(), max_speed=48)  # max highway speed is 120 km/h = 33.33 m/s
        waiting_penalty = -self.normalize_count(self.env.getNumberVehicleWaitingTL(), max_count=250)  # max waiting vehicles = 250
        ramp_speed_reward = self.normalize_speed(self.env.getSpeedRamp(), max_speed=24)  # Assume max ramp speed is 60 km/h = 16.67 m/s

         # Combine rewards and penalties
        reward = (self.mu * speed_hw_reward +
              self.omega * waiting_penalty +
              self.tau * ramp_speed_reward)

        return reward

    def normalize_speed(self, speed, max_speed):
        """Normalize speed to a value between 0 and 1."""
        return speed / max_speed if max_speed > 0 else 0

    def normalize_count(self, count, max_count):
        """Normalize count to a value between 0 and 1."""
        return count / max_count if max_count > 0 else 0
    

    #def rew(self):
        # Reward based on various traffic parameters
    #    return self.mu * self.env.getSpeedHW() + self.omega * self.env.getNumberVehicleWaitingTL() + self.tau * self.env.getSpeedRamp()

    def get_action(self, state, epsilon=0.1):
        # Select action using actor
        action = self.actor_model(state).item()

        # Add Gaussian noise for exploration
        action += np.random.normal(0, epsilon)

        # Clip action between 0 and 1
        action = np.clip(action, 0, 1)
        return action

    def step(self, action):
        for _ in range(self.simulationStepLength):
            hw_flow, ramp_flow = self.interpolate_flow(self.env.getCurrentStep(), self.data_points)
            self.env.setFlowOnHW(hw_flow)
            self.env.setFlowOnRamp(ramp_flow)
            self.env.doSimulationStep(action)

    def reset(self):
        # Reset the state and environment
        for _ in range(3):
            state_matrix = [[0 for _ in range(251)] for _ in range(4)]
            self.state_matrices.appendleft(state_matrix)
        self.env.reset()

    def train(self):
        total_step_loss, total_step_actor_loss, total_step_rewards = [], [], []
        with open('loss_log.csv', 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['Step', 'Actor Loss', 'Critic Loss'])

        for epoch in range(self.epochs):
            self.reset()
            state1 = self.obs()
            isDone, step = False, 0

            while not isDone:
                step += 1

                # Select action using actor with exploration noise
                action = self.get_action(state1)

                # Perform action in environment
                self.step(action)
                state2 = self.obs()
                reward = self.rew()
                print(f"____________________reward___________________: {reward}")
                   
                total_step_rewards.append(reward)

                # Store experience in replay buffer
                exp = (state1, action, reward, state2)
                self.replay.append(exp)
                state1 = state2

                # Perform training step if enough data in replay buffer
                if len(self.replay) > self.batch_size:
                    minibatch = random.sample(self.replay, self.batch_size)
                    state1_batch = torch.cat([s1.unsqueeze(0) for (s1, _, _, _) in minibatch])
                    action_batch = torch.Tensor([a for (_, a, _, _) in minibatch])
                    reward_batch = torch.Tensor([r for (_, _, r, _) in minibatch])
                    state2_batch = torch.cat([s2.unsqueeze(0) for (_, _, _, s2) in minibatch])

                    # Critic update
                    q_value = self.critic_model(torch.cat([state1_batch, action_batch.unsqueeze(1)], dim=1)).squeeze()
                    next_q_value = self.target_critic_model(torch.cat([state2_batch, action_batch.unsqueeze(1)], dim=1)).squeeze()
                    target_q_value = reward_batch + self.gamma * next_q_value
                    critic_loss = self.loss_fn(q_value, target_q_value.detach())
                    self.optimizer_critic.zero_grad()
                    critic_loss.backward()
                    self.optimizer_critic.step()
                    total_step_loss.append(critic_loss.item())
                    print(f"____________________critic_loss___________________: {critic_loss.item()}")
                   

                    # Actor update
                    actor_loss = -self.critic_model(torch.cat([state1_batch, self.actor_model(state1_batch)], dim=1)).mean()
                    self.optimizer_actor.zero_grad()
                    actor_loss.backward()
                    self.optimizer_actor.step()
                    total_step_actor_loss.append(actor_loss.item())
                    print(f"____________________actor_loss___________________: {actor_loss.item()}")
                   

                    # Log losses
                    with open('loss_log.csv', 'a', newline='') as csvfile:
                        writer = csv.writer(csvfile)
                        writer.writerow([step, actor_loss.item(), critic_loss.item()])

                    # Sync target networks periodically
                    if step % self.sync_freq == 0:
                        self.target_actor_model.load_state_dict(self.actor_model.state_dict())
                        self.target_critic_model.load_state_dict(self.critic_model.state_dict())
                

                # Log rewards per epoch
                avg_reward = np.mean(total_step_rewards)
                print(f'Epoch {epoch + 1}/{self.epochs} - Average Reward: {avg_reward}')
            
                if step >= self.max_steps:
                    isDone = True        
            
        return self.actor_model, total_step_loss, total_step_actor_loss, total_step_rewards

    def interpolate_flow(self, step, data_points):
        times, hw_flows, ramp_flows = zip(*data_points)
        hw_flow = np.interp(step, times, hw_flows)
        ramp_flow = np.interp(step, times, ramp_flows)
        return int(hw_flow), int(ramp_flow)


    

# Main script
if __name__ == "__main__":
  
    # Train the agent and collect training data
    agent = DdpAgent(observation_space_n=3012)
    actor_model, total_step_loss, total_step_actor_loss, total_step_rewards = agent.train()
  # Save training results and model
    results = {
        "agent": agent,
        "actor_model": actor_model,
        "total_step_loss": total_step_loss,
        "total_step_actor_loss": total_step_actor_loss,
        "total_step_rewards": total_step_rewards,
    }
    with open('DDPG_training_results.pkl', 'wb') as file:
        pickle.dump(results, file)
    torch.save(actor_model, 'Models/DDPGActorModel.pth')