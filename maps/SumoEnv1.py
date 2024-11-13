import os
import sys

# Set path to SUMO Home and add tools to system path
if "SUMO_HOME" in os.environ:
    tools = os.path.join(os.environ["SUMO_HOME"], "tools")
    sys.path.append(tools)
else:
    sys.exit("Please declare the environment variable 'SUMO_HOME'.")

import traci
from sumolib import checkBinary

class SumoEnv:
    def __init__(self, gui=False, flow_on_HW=4500, flow_on_Ramp=1800):
        """
        Initializes the SUMO environment.
        
        Parameters:
        gui (bool): If True, runs SUMO with GUI; else, without GUI.
        flow_on_HW (int): Traffic flow rate on the highway (vehicles per hour).
        flow_on_Ramp (int): Traffic flow rate on the ramp (vehicles per hour).
        """
        # Define the SUMO binary based on whether GUI is requested
        sumoBinary = checkBinary('sumo-gui') if gui else checkBinary('sumo')
        
        # Define the SUMO configuration file path
        current_dir = os.path.dirname(os.path.abspath(__file__))
        sumocfg_path = os.path.join(current_dir, "HRC.sumocfg")
        
        # Define SUMO command to start simulation
        self.sumoCmd = [sumoBinary, "-c", sumocfg_path]
        
        # Initialize the SUMO connection
        traci.start(self.sumoCmd)
        
        # Simulation parameters
        self.flow_on_HW = flow_on_HW
        self.flow_on_Ramp = flow_on_Ramp
        self.partial_vehicle_accumulator_HW = 0
        self.partial_vehicle_accumulator_Ramp = 0

        # Data storage variables
        self.steps, self.flow_steps = [], []
        self.flows_HW, self.speeds_HW, self.densities_HW = [], [], []
        self.travelTimes_HW, self.flows_Ramp, self.speeds_Ramp = [], [], []
        self.densities_Ramp, self.travelTimes_Ramp, self.travelTimesSystem = [], [], []
        self.trafficLightPhases = []
        
        # Simulation step counter and metrics for HW and Ramp
        self.step = 0
        self.numberOfVehicleHW, self.numberOfVehicleRamp = 0, 0
        self.densityHW, self.densityRamp = 0, 0
        self.speedHW, self.speedRamp = 0, 0
        self.travelTimeHW, self.travelTimeRamp, self.travelTimeSystem = 0, 0, 0
        
        # Vehicle start times for calculating travel times
        self.vehicle_depart_times_HW = {}
        self.vehicle_depart_times_Ramp = {}

    def reset(self):
        """
        Resets the simulation by closing and restarting SUMO.
        Clears all accumulated data and re-initializes variables.
        """
        traci.close()
        # Clear all data and reset counters
        self.steps, self.flow_steps = [], []
        self.flows_HW, self.speeds_HW, self.densities_HW = [], [], []
        self.travelTimes_HW, self.flows_Ramp, self.speeds_Ramp = [], [], []
        self.densities_Ramp, self.travelTimes_Ramp, self.travelTimesSystem = [], [], []
        self.step = 0
        self.numberOfVehicleHW, self.numberOfVehicleRamp = 0, 0
        self.densityHW, self.densityRamp = 0, 0
        self.speedHW, self.speedRamp = 0, 0
        self.travelTimeHW, self.travelTimeRamp, self.travelTimeSystem = 0, 0, 0
        self.partial_vehicle_accumulator_HW = 0
        self.partial_vehicle_accumulator_Ramp = 0
        traci.start(self.sumoCmd)

    def doSimulationStep(self, phase_index):
        """
        Performs a single simulation step in SUMO.
        
        Parameters:
        phase_index (int): Traffic light phase to be set.
        """
        # Set traffic light phase
        traci.trafficlight.setPhase("JRTL1", phase_index)
        self.trafficLightPhases.append(phase_index)
        
        # Generate vehicle flows for HW and Ramp
        self.partial_vehicle_accumulator_HW = self.generateVehicles("r_Highway", "car_truck", self.flow_on_HW, self.step, self.partial_vehicle_accumulator_HW)
        self.partial_vehicle_accumulator_Ramp = self.generateVehicles("r_Ramp", "car_truck", self.flow_on_Ramp, self.step, self.partial_vehicle_accumulator_Ramp)
        
        # Advance simulation by one step
        traci.simulationStep()
        self.step += 1
        
        # Collect and calculate metrics for this step
        self.collectMetrics()
        
        # Record data for visualization every 100 steps
        mesureIntervalLenght = 100
        if self.step % mesureIntervalLenght == 0:
            self.recordFlowData(mesureIntervalLenght)

    def generateVehicles(self, route_id, vehicle_type, flow_rate, step, partial_vehicle_accumulator):
        """
        Generates vehicles based on flow rate, adding them to SUMO simulation.
        
        Parameters:
        route_id (str): Route ID for generated vehicles.
        vehicle_type (str): Type of vehicle to generate.
        flow_rate (int): Flow rate in vehicles per hour.
        step (int): Current simulation step.
        partial_vehicle_accumulator (float): Accumulator for vehicles per second.
        
        Returns:
        float: Updated partial vehicle accumulator after generation.
        """
        vehicles_per_second = flow_rate / 3600
        partial_vehicle_accumulator += vehicles_per_second
        vehicles_to_create = 0

        while partial_vehicle_accumulator >= 1:
            vehicles_to_create += 1
            partial_vehicle_accumulator -= 1

        for veh in range(vehicles_to_create):
            vehicle_id = f"veh_{step}{veh}{route_id}"
            try:
                traci.vehicle.add(vehicle_id, route_id, vehicle_type)
            except traci.TraCIException as e:
                print(f"Vehicle {vehicle_id} could not be created: {e}")
        
        return partial_vehicle_accumulator

    def collectMetrics(self):
        """
        Collects traffic metrics such as vehicle count, speed, density, and travel times for HW and Ramp.
        """
        # Vehicle count on edges
        self.numberOfVehicleHW = traci.edge.getLastStepVehicleNumber("HW_Ramp")
        self.numberOfVehicleRamp = traci.edge.getLastStepVehicleNumber("Ramp_beforeTL")
        
        # Density calculation (vehicles per km per lane)
        self.densityHW = self.numberOfVehicleHW / traci.lane.getLength("HW_Ramp_1") * 1000 / traci.edge.getLaneNumber("HW_Ramp")
        self.densityRamp = self.numberOfVehicleRamp / traci.lane.getLength("Ramp_beforeTL_0") * 1000

        # Speed (mean speed on edges)
        self.speedHW = traci.edge.getLastStepMeanSpeed("HW_Ramp")
        self.speedRamp = traci.edge.getLastStepMeanSpeed("Ramp_beforeTL")
        
        # Travel times for vehicles arriving on HW and Ramp
        # (new travel time approach based on recorded departure and arrival steps)
        self.recordTravelTimes()

    def recordTravelTimes(self):
        """
        Records the travel times of vehicles arriving at their destinations.
        """
        # Identify newly started vehicles on HW and Ramp, recording their start time
        starting_vehicles_HW = set(traci.edge.getLastStepVehicleIDs("HW_beforeRamp"))
        starting_vehicles_Ramp = set(traci.edge.getLastStepVehicleIDs("Ramp_beforeTL"))
        new_starters_HW = starting_vehicles_HW - self.vehicle_depart_times_HW.keys()
        new_starters_Ramp = starting_vehicles_Ramp - self.vehicle_depart_times_Ramp.keys()
        
        for vehicle_id in new_starters_HW:
            self.vehicle_depart_times_HW[vehicle_id] = self.step
        for vehicle_id in new_starters_Ramp:
            self.vehicle_depart_times_Ramp[vehicle_id] = self.step
        
        # Calculate travel times for vehicles arriving on the edge "HW_afterRamp"
        self.updateArrivalTimes("HW_afterRamp")

    def updateArrivalTimes(self, arrival_edge):
        """
        Updates the travel times of vehicles arriving on the specified edge.
        
        Parameters:
        arrival_edge (str): Edge where vehicles arrive, completing their trip.
        """
        arrived_vehicles = set(traci.edge.getLastStepVehicleIDs(arrival_edge))
        finishing_vehicles_HW = arrived_vehicles.intersection(self.vehicle_depart_times_HW.keys())
        finishing_vehicles_Ramp = arrived_vehicles.intersection(self.vehicle_depart_times_Ramp.keys())
        
        travel_times_HW_this_step, travel_times_Ramp_this_step = [], []

        # Calculate travel times for HW vehicles
        for vehicle_id in finishing_vehicles_HW:
            depart_time = self.vehicle_depart_times_HW.pop(vehicle_id, None)
            if depart_time is not None:
                travel_time = self.step - depart_time
                travel_times_HW_this_step.append(travel_time)
        
        # Calculate travel times for Ramp vehicles
        for vehicle_id in finishing_vehicles_Ramp:
            depart_time = self.vehicle_depart_times_Ramp.pop(vehicle_id, None)
            if depart_time is not None:
                travel_time = self.step - depart_time
                travel_times_Ramp_this_step.append(travel_time)

        # Average travel times for this step
        self.travelTimes_HW.append(sum(travel_times_HW_this_step) / len(travel_times_HW_this_step) if travel_times_HW_this_step else 0)
        self.travelTimes_Ramp.append(sum(travel_times_Ramp_this_step) / len(travel_times_Ramp_this_step) if travel_times_Ramp_this_step else 0)

        # Overall system travel time
        all_travel_times = travel_times_HW_this_step + travel_times_Ramp_this_step
        self.travelTimesSystem.append(sum(all_travel_times) / len(all_travel_times) if all_travel_times else 0)
    
    def recordFlowData(self, mesureIntervalLenght):
        """
        Records aggregated data for visualization every specified interval.
        
        Parameters:
        mesureIntervalLenght (int): Interval in steps to record data.
        """
        self.flow_steps.append(self.step)
        self.flows_HW.append(self.numberOfVehicleHW / mesureIntervalLenght)
        self.flows_Ramp.append(self.numberOfVehicleRamp / mesureIntervalLenght)

    def close(self):
        """
        Closes the SUMO connection.
        """
        traci.close()
