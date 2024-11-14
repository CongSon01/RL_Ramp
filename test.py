from maps.SumoEnv import SumoEnv  # Importieren Sie Ihre SumoEnv-Klasse aus der entsprechenden Datei
import torch
import numpy as np
from collections import deque

# Erstellen Sie eine Instanz der Umgebung
flow_on_HW = 5000
flow_on_Ramp= 2000
simulationStepLength = 2 # 2 seconds per simulation step (time that elapses in the simulation per DQN step)
max_steps = 3600 / simulationStepLength # 1 hour traffic simulation
# Führen Sie die Simulation für 3600 Schritte durch
num_steps = 3600
# Traffic flow data for the simulation
data_points = [
            (0, 1000, 500),    # Data point at 7:50 a.m.
            (10, 2000, 1300),  # Data point at 8:00 a.m.
            (20, 3200, 1800),  # Data point at 8:10 a.m.
            (30, 2500, 1500),  # Data point at 8:20 a.m.
            (40, 1500, 1000),  # Data point at 8:30 a.m.
            (50, 1000, 700),   # Data point at 8:40 a.m.
            (60, 800, 500),    # Data point at 8:50 a.m.
]
# Conversion of time from minutes to steps (assumes 1 step = 1 second)

data_points = [(t * 60, hw, ramp) for t, hw, ramp in data_points]
model = torch.load('models/DynamicModel2.pth')

env = SumoEnv(gui=True, flow_on_HW = flow_on_HW, flow_on_Ramp = flow_on_Ramp) 


state_matrices = deque(maxlen=3) # Create queue for the obs matrices (3 DQN steps included)

        ### Optional to collect the reward over several steps ###
        # self.rewardArray = deque(maxlen=3) # Create queue for the rewards (3 DQN steps included)
        # self.rewardArray.extend([0, 0, 0])
        # self.simStepReward = 0 # Reward für einen Simulationsschritt

        # Initialize the deque with state matrices (all zeros)
for _ in range(3):
    state_matrix = [[0 for _ in range(251)] for _ in range(4)]
    state_matrices.appendleft(state_matrix)

# Function for linear interpolation of the flow data
def interpolate_flow(step, data_points):

    times = [point[0] for point in data_points]
    hw_flows = [point[1] for point in data_points]
    ramp_flows = [point[2] for point in data_points]
    
    hw_flow = np.interp(step, times, hw_flows)
    ramp_flow = np.interp(step, times, ramp_flows)
    
    return int(hw_flow), int(ramp_flow)

def obs():
        # Get the state matrix
        state_matrix = env.getStateMatrixV2()

        # Add the state matrix to the queue
        state_matrices.appendleft(state_matrix)

        # Convert the matrices into a flat NumPy array
        flat_state_array = np.concatenate(state_matrices).flatten()

        # Create a PyTorch tensor from the flat NumPy array
        obs_tensor = torch.from_numpy(flat_state_array).float()

        return obs_tensor

def step(action):
    # Execute the action in the surrounding area
    for _ in range(simulationStepLength):
        ### Optional: Dynamically adjust the flow ###
        env.setFlowOnHW(interpolate_flow(env.getCurrentStep(), data_points)[0])
        env.setFlowOnRamp(interpolate_flow(env.getCurrentStep(), data_points)[1])
        env.doSimulationStep(action)

def testModel(model, useModel=True):
    env.reset()
    state1 = obs()
    isDone = False
    mov = 0
    while not isDone:
        mov += 1
        if useModel:
            qval = model(state1)
            qval_ = qval.data.numpy()
            action_ = np.argmax(qval_)
        else:
            action_ = 0
        step(action_)
        state1 = obs()
        if mov > max_steps:
            isDone = True
    env.close()
    return env.getStatistics() 

# for step in range(num_steps):
#     # Führen Sie einen Simulationsschritt aus
#     value = 0
#     # if step % 50 == 0:
#     #     # Wechselt den Wert zwischen 0 und 1 alle 5 Schritte
#     #     value = 0 if (step // 50) % 2 == 0 else 1
#     env.doSimulationStep(value)

testModel(model, useModel=True)

# Schließen Sie die Umgebung am Ende der Simulation
env.close()