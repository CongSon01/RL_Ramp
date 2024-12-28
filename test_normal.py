from maps.SumoEnv import SumoEnv  # Import your SumoEnv class from the corresponding file
import torch
import numpy as np
from collections import deque
import pickle

# Constants for traffic simulation
FLOW_ON_HW = 5000  # Highway inflow rate (vehicles per hour)
FLOW_ON_RAMP = 2000  # Ramp inflow rate (vehicles per hour)
SIMULATION_STEP_LENGTH = 2  # Simulation step length in seconds
MAX_STEPS = int(3600 / SIMULATION_STEP_LENGTH)  # Maximum steps for a 1-hour traffic simulation
NUM_STEPS = 3600  # Total number of simulation steps
ACTION_CHANGE_INTERVAL = 60  # Time interval (in seconds) to change action when not using the model

def obs(env, state_matrices):
    """
    Collects the observation from the environment and prepares it for the model.

    Args:
        env (SumoEnv): The simulation environment instance.
        state_matrices (deque): A queue storing the latest state matrices.

    Returns:
        torch.Tensor: Flattened tensor of concatenated state matrices for input to the model.
    """
    state_matrix = env.getStateMatrixV2()
    state_matrices.appendleft(state_matrix)
    flat_state_array = np.concatenate(state_matrices).flatten()
    return torch.from_numpy(flat_state_array).float()

def step(env, action):
    """
    Executes the specified action in the environment for the defined simulation step length.

    Args:
        env (SumoEnv): The simulation environment instance.
        action (int): The action to execute in the environment.
    """
    for _ in range(SIMULATION_STEP_LENGTH):
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
        step(env, action_)
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
    filename = "./ddpg.pkl"
    with open(filename, "wb") as f:
        pickle.dump(stats, f)

    env.close()

if __name__ == "__main__":
    main()
