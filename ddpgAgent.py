import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
import copy
import pickle
from maps.SumoEnv import SumoEnv 

class ActorNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(ActorNetwork, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, output_dim), nn.Sigmoid()  # Output continuous action in [0, 1]
        )

    def forward(self, x):
        return self.model(x)

class CriticNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(CriticNetwork, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(state_dim + action_dim, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, 1)  # Output Q-value
        )

    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        return self.model(x)

class DDPGAgent:
    def __init__(self, state_dim, action_dim):
        # Device setup
        device = "mps" if torch.backends.mps.is_available() else "cpu"
        print(f"Using device: {device}")

        # Seed for reproducibility
        random_seed = 33
        torch.manual_seed(random_seed)
        np.random.seed(random_seed)
        random.seed(random_seed)

        # Environment setup
        self.env = SumoEnv(gui=False, flow_on_HW=5000, flow_on_Ramp=2000)
        self.state_matrices = deque(maxlen=3)
        for _ in range(3):
            state_matrix = [[0 for _ in range(251)] for _ in range(4)]
            self.state_matrices.appendleft(state_matrix)
        self.data_points = [(t * 60, hw, ramp) for t, hw, ramp in [
            (0, 1000, 500), (10, 2000, 1300), (20, 3200, 1800),
            (30, 2500, 1500), (40, 1500, 1000), (50, 1000, 700), (60, 800, 500)
        ]]

        # Hyperparameters
        self.gamma = 0.99
        self.tau = 0.005
        self.actor_lr = 1e-4
        self.critic_lr = 1e-3
        self.batch_size = 32
        self.max_steps = 3600 // 60
        self.exploration_noise = 0.1
        self.memory = deque(maxlen=50000)
        self.epochs = 5

        # Actor and Critic networks
        self.actor = ActorNetwork(state_dim, action_dim).to(device)
        self.critic = CriticNetwork(state_dim, action_dim).to(device)
        self.target_actor = copy.deepcopy(self.actor)
        self.target_critic = copy.deepcopy(self.critic)

        # Optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.critic_lr)
        self.loss_fn = nn.MSELoss()

    def obs(self):
        state_matrix = self.env.getStateMatrixV2()
        self.state_matrices.appendleft(state_matrix)
        flat_state_array = np.concatenate(self.state_matrices).flatten()
        return torch.from_numpy(flat_state_array).float()

    def rew(self):
        return 0.05 * self.env.getSpeedHW() - 0.5 * self.env.getNumberVehicleWaitingTL() + 0.2 * self.env.getSpeedRamp()

    def step(self, action):
        for _ in range(60):
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
        for epoch in range(self.epochs):
            print("Epoch:", epoch)
            self.reset()
            state = self.obs()
            done = False
            episode_rewards = 0

            while not done:
                total_steps += 1

                # Select action with exploration noise
                action = self.actor(state).detach().item()
                noise = self.exploration_noise * np.random.normal()
                action = np.clip(action + noise, 0, 1)
                action_tensor = torch.Tensor([action])

                # Execute action
                self.step(action)
                next_state = self.obs()
                reward = self.rew()
                episode_rewards += reward
                total_step_rewards.append(reward)
                done = False

                # Store experience in replay buffer
                self.memory.append((state, action_tensor, reward, next_state, done))
                state = next_state

                # Sample and train
                if len(self.memory) > self.batch_size:
                    self.train_step()

                # Soft update target networks
                self.soft_update(self.actor, self.target_actor)
                self.soft_update(self.critic, self.target_critic)

                if done or total_steps >= self.max_steps:
                    done = True

            print(f"Epoch {epoch + 1}/{self.epochs}, Total Reward: {episode_rewards}")

        return self.actor, np.array(total_step_loss), np.array(total_step_rewards), total_steps

    def train_step(self):
        minibatch = random.sample(self.memory, self.batch_size)

        states = torch.cat([s[0].unsqueeze(0) for s in minibatch])
        actions = torch.cat([s[1].unsqueeze(0) for s in minibatch])
        rewards = torch.Tensor([s[2] for s in minibatch])
        next_states = torch.cat([s[3].unsqueeze(0) for s in minibatch])
        dones = torch.Tensor([s[4] for s in minibatch])

        # Critic update
        with torch.no_grad():
            target_actions = self.target_actor(next_states)
            target_q_values = self.target_critic(next_states, target_actions)
            y = rewards + self.gamma * target_q_values * (1 - dones)

        critic_q_values = self.critic(states, actions)
        critic_loss = self.loss_fn(critic_q_values, y.detach())
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Actor update
        actor_loss = -self.critic(states, self.actor(states)).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

    def soft_update(self, local_model, target_model):
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(self.tau * local_param.data + (1.0 - self.tau) * target_param.data)

    def interpolate_flow(self, step, data_points):
        times, hw_flows, ramp_flows = zip(*data_points)
        hw_flow = np.interp(step, times, hw_flows)
        ramp_flow = np.interp(step, times, ramp_flows)
        return int(hw_flow), int(ramp_flow)

# Main script
if __name__ == "__main__":
    agent = DDPGAgent(state_dim=3012, action_dim=1)
    actor_model, total_step_loss, total_step_rewards, steps = agent.train()

    # Save training results and actor model
    results = {"actor_model": actor_model, "total_step_loss": total_step_loss, "total_step_rewards": total_step_rewards, "steps": steps}
    with open('ddpg_training_results.pkl', 'wb') as file:
        pickle.dump(results, file)
        print("DDPG training results saved successfully.")
    torch.save(actor_model, 'Models/DDPGActorModel3.pth')
