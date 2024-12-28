from maps.SumoEnv import SumoEnv  # Import your SumoEnv class from the corresponding file
import torch
import numpy as np
from collections import deque
import pickle
import random
# Constants for traffic simulation
FLOW_ON_HW = 5000  # Highway inflow rate (vehicles per hour)
FLOW_ON_RAMP = 2000  # Ramp inflow rate (vehicles per hour)
SIMULATION_STEP_LENGTH = 2  # Simulation step length in seconds
MAX_STEPS = int(3600 / SIMULATION_STEP_LENGTH)  # Maximum steps for a 1-hour traffic simulation
NUM_STEPS = 3600  # Total number of simulation steps
ACTION_CHANGE_INTERVAL = 60  # Time interval (in seconds) to change action when not using the model

# Traffic flow data points (time in seconds, highway flow, ramp flow)
DATA_POINTS = [
    (0, 1000, 500),
    (600, 2000, 1300),
    (1200, 3200, 1800),
    (1800, 2500, 1500),
    (2400, 1500, 1000),
    (3000, 1000, 700),
    (3600, 800, 500),
]

def obs(env, state_matrices, delay_probability=0.01):
    """
    Collects the observation from the environment and prepares it for the model.
    Simulates delayed observations by randomly hiding some values > 0 in the state matrix.

    Args:
        env (SumoEnv): The simulation environment instance.
        state_matrices (deque): A queue storing the latest state matrices.
        delay_probability (float): Probability of masking a value > 0 to simulate delayed observations.

    Returns:
        torch.Tensor: Flattened tensor of concatenated state matrices for input to the model.
    """
    state_matrix = env.getStateMatrixV2()
    
    # Apply delayed observation by masking some values > 0
    mask = (state_matrix > 0) & (np.random.rand(*state_matrix.shape) > delay_probability)
    delayed_state_matrix = state_matrix * mask

    # Add to the state_matrices queue
    state_matrices.appendleft(delayed_state_matrix)

    # Flatten and return as a torch tensor
    flat_state_array = np.concatenate(state_matrices).flatten()
    return torch.from_numpy(flat_state_array).float()

def interpolate_flow(step, data_points):
    """
    Interpolates traffic flow values for highway and ramp at a given simulation step.

    Args:
        step (int): Current simulation step.
        data_points (list): List of tuples containing time, highway flow, and ramp flow.

    Returns:
        tuple: Interpolated highway and ramp flow rates (int).
    """
    times = [point[0] for point in data_points]
    hw_flows = [point[1] for point in data_points]
    ramp_flows = [point[2] for point in data_points]
    hw_flow = np.interp(step, times, hw_flows)
    ramp_flow = np.interp(step, times, ramp_flows)
    return int(hw_flow), int(ramp_flow)

def step(env, action, data_points):
    """
    Executes the specified action in the environment for the defined simulation step length.

    Args:
        env (SumoEnv): The simulation environment instance.
        action (int): The action to execute in the environment.
        data_points (list): Traffic flow data points for interpolation.
    """
    for _ in range(SIMULATION_STEP_LENGTH):
        current_step = env.getCurrentStep()
        hw_flow, ramp_flow = interpolate_flow(current_step, data_points)
        env.setFlowOnHW(hw_flow)
        env.setFlowOnRamp(ramp_flow)
        env.doSimulationStep(action)

def test_model(env, model, state_matrices, use_model=True):
    """
    Tests the given model in the simulation environment.

    Args:
        env (SumoEnv): The simulation environment instance.
        model (torch.nn.Module): The trained model to test.
        state_matrices (deque): A queue storing the latest state matrices.
        use_model (bool): Whether to use the model's predictions or a default action.

    Returns:
        dict: Simulation statistics collected during the test.
    """
    env.reset()
    state1 = obs(env, state_matrices)
    is_done = False
    steps_taken = 0
    last_action_change = 0
    action_ = 0  # Default action when not using the model

    while not is_done:
        steps_taken += 1
        if use_model:
            qval = model(state1)
            qval_ = qval.data.numpy()
            action_ = np.argmax(qval_)
        else:
            # Change action every ACTION_CHANGE_INTERVAL seconds
            if (steps_taken * SIMULATION_STEP_LENGTH) - last_action_change >= ACTION_CHANGE_INTERVAL:
                action_ = (action_ + 1) % 2  # Toggle between 0 and 1 for example
                last_action_change = steps_taken * SIMULATION_STEP_LENGTH
        step(env, action_, DATA_POINTS)
        state1 = obs(env, state_matrices)
        if steps_taken > MAX_STEPS:
            is_done = True

    env.close()
    return env.getStatistics()

def main():
    """
    Main function to set up the simulation environment and test the trained model.
    """
    # Initialize the simulation environment
    env = SumoEnv(gui=True, flow_on_HW=FLOW_ON_HW, flow_on_Ramp=FLOW_ON_RAMP)

    # Initialize state matrices queue
    state_matrices = deque(maxlen=3)
    for _ in range(3):
        state_matrix = [[0 for _ in range(251)] for _ in range(4)]
        state_matrices.appendleft(state_matrix)

    # Load the trained model
    model = torch.load('DynamicModel.pth')

    # Test the model in the environment
    stats = test_model(env, model, state_matrices, use_model=False)
    with open('./ddpg.pkl', 'wb') as f:
        pickle.dump(stats, f)

    env.close()

if __name__ == "__main__":
    main()
