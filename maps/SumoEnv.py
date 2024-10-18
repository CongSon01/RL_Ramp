import os
import sys

# Set path to SUMO Home
if "SUMO_HOME" in os.environ:
    tools = os.path.join(os.environ["SUMO_HOME"], "tools")
    sys.path.append(tools)
else:
    sys.exit("please declare environment variable 'SUMO_HOME'")

import traci
from sumolib import checkBinary

class SumoEnv:
    def __init__(self, gui=False, flow_on_HW=4500, flow_on_Ramp=1800):
        
        # SUMO configuration (with or without GUI)
        if gui:
            sumoBinary = checkBinary('sumo-gui')
        else:
            sumoBinary = checkBinary('sumo')

        # Path to the directory of the current script
        current_dir = os.path.dirname(os.path.abspath(__file__))

        # Path to the SUMO configuration file (sumocfg)
        sumocfg_path = os.path.join(current_dir, "HRC.sumocfg")

        self.sumoCmd = [sumoBinary, "-c", sumocfg_path]

        # Initializing the SUMO connection for the first time
        traci.start(self.sumoCmd)

        ### Further initializations ###
        self.flow_on_HW = flow_on_HW
        self.flow_on_Ramp = flow_on_Ramp

        self.partial_vehicle_accumulator_HW = 0
        self.partial_vehicle_accumulator_Ramp = 0

        # Initialization of the lists for data storage
        self.steps, self.flow_steps, self.flows_HW, self.speeds_HW, self.densities_HW, self.travelTimes_HW, self.flows_Ramp, self.speeds_Ramp, self.densities_Ramp, self.travelTimes_Ramp, self.travelTimesSystem, self.trafficLightPhases = [], [], [], [], [], [], [], [], [], [], [], []
        # Initialization of the variables for data storage
        self.step, self.numberOfVehicleHW, self.numberOfVehicleRamp, self.densityHW, self.densityRamp, self.speedHW, self.speedRamp, self.travelTimeHW, self.travelTimeRamp, self.travelTimeSystem = 0, 0, 0, 0, 0, 0, 0, 0, 0, 0

        # Initialization of start and arrival times
        self.vehicle_depart_times_HW = {}  # Start times for HW
        self.vehicle_depart_times_Ramp = {}  # Start times for Ramp
        

    ### Methods to call for the agent ###

    def reset(self):
        traci.close()  
        # Reset the variables and lists
        self.steps, self.flow_steps, self.flows_HW, self.speeds_HW, self.densities_HW, self.travelTimes_HW, self.flows_Ramp, self.speeds_Ramp, self.densities_Ramp, self.travelTimes_Ramp, self.travelTimesSystem = [], [], [], [], [], [], [], [], [], [], []
        self.step, self.numberOfVehicleHW, self.numberOfVehicleRamp, self.densityHW, self.densityRamp, self.speedHW, self.speedRamp, self.travelTimeHW, self.travelTimeRamp, self.travelTimeSystem = 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
        self.partial_vehicle_accumulator_HW = 0
        self.partial_vehicle_accumulator_Ramp = 0
        # Starts a new SUMO session
        traci.start(self.sumoCmd)  

    def doSimulationStep(self, phase_index):
        # Set the phase of the traffic light
        traci.trafficlight.setPhase("JRTL1", phase_index)
        self.trafficLightPhases.append(phase_index)

        # Generate vehicle flows
        self.partial_vehicle_accumulator_HW = self.generateVehicles("r_Highway", "car_truck", self.flow_on_HW, self.step, self.partial_vehicle_accumulator_HW) # Highway
        self.partial_vehicle_accumulator_Ramp = self.generateVehicles("r_Ramp", "car_truck", self.flow_on_Ramp, self.step, self.partial_vehicle_accumulator_Ramp) # Ramp

        # Execute simulation step
        traci.simulationStep()
        self.step += 1

        # Number Veh
        self.numberOfVehicleHW = traci.edge.getLastStepVehicleNumber("HW_Ramp")
        self.numberOfVehicleRamp = traci.edge.getLastStepVehicleNumber("Ramp_beforeTL")

        # Density: (Veh/km/lane)
        self.densityHW = self.numberOfVehicleHW / traci.lane.getLength("HW_Ramp_1")*1000 / traci.edge.getLaneNumber("HW_Ramp") 
        self.densityRamp = self.numberOfVehicleRamp / traci.lane.getLength("Ramp_beforeTL_0")*1000 

        # Speed (mean)
        self.speedHW = traci.edge.getLastStepMeanSpeed("HW_Ramp")
        self.speedRamp = traci.edge.getLastStepMeanSpeed("Ramp_beforeTL")

        # TravelTime (length/mean speed) OLD!!!
        # self.travelTimeHW = traci.edge.getTraveltime("HW_beforeRamp") + traci.edge.getTraveltime("HW_Ramp") + traci.edge.getTraveltime("HW_afterRamp")
        # self.travelTimeRamp =  traci.edge.getTraveltime("Ramp_beforeTL") + traci.edge.getTraveltime("Ramp_afterTL") + traci.edge.getTraveltime("HW_Ramp") + traci.edge.getTraveltime("HW_afterRamp")
        # self.travelTimeSystem = self.travelTimeHW + traci.edge.getTraveltime("Ramp_beforeTL") + traci.edge.getTraveltime("Ramp_afterTL")

        # Real TravelTime (arrival - depart) NEW!!!

        # Recording the vehicles that appear on the starting edges
        starting_vehicles_HW = set(traci.edge.getLastStepVehicleIDs("HW_beforeRamp"))
        starting_vehicles_Ramp = set(traci.edge.getLastStepVehicleIDs("Ramp_beforeTL"))

        # Update the start times, if not yet recorded
        new_starters_HW = starting_vehicles_HW - self.vehicle_depart_times_HW.keys()
        new_starters_Ramp = starting_vehicles_Ramp - self.vehicle_depart_times_Ramp.keys()

        for vehicle_id in new_starters_HW:
            self.vehicle_depart_times_HW[vehicle_id] = self.step

        for vehicle_id in new_starters_Ramp:
            self.vehicle_depart_times_Ramp[vehicle_id] = self.step

        # Record the arrival times and calculate the travel times for this interval
        travel_times_HW_this_step = []
        travel_times_Ramp_this_step = []

        # Capture the vehicles that appear on the trailing edge
        arrived_vehicles = set(traci.edge.getLastStepVehicleIDs("HW_afterRamp"))

        # Determine the vehicles that reach their destination (HW and ramp)
        finishing_vehicles_HW = arrived_vehicles.intersection(self.vehicle_depart_times_HW.keys())
        finishing_vehicles_Ramp = arrived_vehicles.intersection(self.vehicle_depart_times_Ramp.keys())

        # Recording the arrival times for HW
        for vehicle_id in finishing_vehicles_HW:
            depart_time = self.vehicle_depart_times_HW.pop(vehicle_id, None)
            if depart_time is not None:
                travel_time = self.step - depart_time
                travel_times_HW_this_step.append(travel_time)

        # Record arrival times for Ramp
        for vehicle_id in finishing_vehicles_Ramp:
            depart_time = self.vehicle_depart_times_Ramp.pop(vehicle_id, None)
            if depart_time is not None:
                travel_time = self.step - depart_time
                travel_times_Ramp_this_step.append(travel_time)

        # Calculation of the average travel times for HW in this step
        if travel_times_HW_this_step:
            avg_travel_time_HW = sum(travel_times_HW_this_step) / len(travel_times_HW_this_step)
        else:
            avg_travel_time_HW = self.travelTimes_HW[-1] if self.travelTimes_HW else 0
        self.travelTimes_HW.append(avg_travel_time_HW)

        # Calculation of the average travel times for Ramp in this step
        if travel_times_Ramp_this_step:
            avg_travel_time_Ramp = sum(travel_times_Ramp_this_step) / len(travel_times_Ramp_this_step)
        else:
            avg_travel_time_Ramp = self.travelTimes_Ramp[-1] if self.travelTimes_Ramp else 0
        self.travelTimes_Ramp.append(avg_travel_time_Ramp)

        # Calculate the average system travel time and add to the list
        total_travel_times = travel_times_HW_this_step + travel_times_Ramp_this_step
        if total_travel_times:
            avg_travel_time_System = sum(total_travel_times) / len(total_travel_times)
        else:
            avg_travel_time_System = self.travelTimesSystem[-1] if self.travelTimesSystem else 0
        self.travelTimesSystem.append(avg_travel_time_System)

        # Collect data for graphics
        self.steps.append(self.step)
        self.speeds_HW.append(self.speedHW)
        self.densities_HW.append(self.densityHW)
        self.speeds_Ramp.append(self.speedRamp)
        self.densities_Ramp.append(self.densityRamp)

        # Induktionsschleifenmessung Intervall 
        mesureIntervalLenght = 100 # must be the same time as defined in idnuction loop 
        if self.step % mesureIntervalLenght == 0:
            self.flow_steps.append(self.step)

            # Number Veh entered
            numberOfVehicleEnteredHWfromRamp = traci.inductionloop.getLastIntervalVehicleNumber("det_0")
            numberOfVehicleEnteredHW = traci.inductionloop.getLastIntervalVehicleNumber("det_1") + traci.inductionloop.getLastIntervalVehicleNumber("det_2") 
            
            # Flow (Veh/h)
            flowHW = numberOfVehicleEnteredHW/mesureIntervalLenght*3600
            flowRamp = numberOfVehicleEnteredHWfromRamp/mesureIntervalLenght*3600

            # Collect data for graphics
            self.flows_HW.append(flowHW)
            self.flows_Ramp.append(flowRamp)
            
    def generateVehicles(self, route_id, vehicle_type, flow_rate, step, partial_vehicle_accumulator):
        # Calculate the vehicles per second based on the current flow rate
        vehicles_per_second = flow_rate / 3600

        # Add the expected vehicles of this step to the accumulator
        partial_vehicle_accumulator += vehicles_per_second

        # If the accumulator reaches 1 or more, create vehicles and update the accumulator
        vehicles_to_create = 0
        while partial_vehicle_accumulator >= 1:
            vehicles_to_create += 1
            partial_vehicle_accumulator -= 1

        # Create the calculated number of vehicles
        for veh in range(vehicles_to_create):
            vehicle_id = f"veh_{step}{veh}{route_id}"
            try:
                traci.vehicle.add(vehicle_id, route_id, vehicle_type, "now", "free", "free", "speedLimit")
            except traci.TraCIException as e:
                print(f"Vehicle {vehicle_id} could not be created: {e}")

        return partial_vehicle_accumulator

    
    def getCurrentStep(self):
        return self.step
    
    def setFlowOnHW(self, flow_on_HW):
        self.flow_on_HW = flow_on_HW

    def setFlowOnRamp(self, flow_on_Ramp):
        self.flow_on_Ramp = flow_on_Ramp
        
    def setTrafficLight(self, phase_index):
        traci.trafficlight.setPhase("JRTL1", phase_index)

    def getTrafficLightState(self):
        return traci.trafficlight.getRedYellowGreenState("JRTL1")  
    
    def getDensityHW(self):
        return self.densityHW
    
    def getDensityRamp(self):
        return self.densityRamp
    
    def getSpeedHW(self):
        return self.speedHW
    
    def getSpeedRamp(self): 
        return self.speedRamp
    
    def getTravelTimeHW(self):
        return self.travelTimeHW
    
    def getTravelTimeRamp(self):
        return self.travelTimeRamp
    
    def getTravelTimeSystem(self):
        return self.travelTimeSystem
    
    def getNumberVehicleWaitingTL(self):
        return traci.edge.getLastStepHaltingNumber("Ramp_beforeTL")
    
    # without taking into account the length of the vehicles
    def getStateMatrix(self):
        # Read out length and number of lanes
        laneLength = traci.lane.getLength("HW_Ramp_0")
        laneNumber = traci.edge.getLaneNumber("HW_Ramp")

        # Create an array stateMatrix with zeros
        stateMatrix = [[0 for _ in range(int(laneLength) + 1)] for _ in range(int(laneNumber) + 1)]

        # Reading out the vehicles on the road
        vehicleList = traci.edge.getLastStepVehicleIDs("HW_Ramp")
        for vehID in vehicleList:
            
            lanePosition = traci.vehicle.getLanePosition(vehID)
            # Splitting the string at the underscores and converting the last element into an integer
            lane = int(traci.vehicle.getLaneID(vehID).split('_')[-1])
            # Set the 1 at the corresponding position in the stateMatrix
            stateMatrix[lane][int(lanePosition)] = 1

        numberOfVehicleWaitingTF = traci.edge.getLastStepHaltingNumber("Ramp_beforeTL")

        for i in range(numberOfVehicleWaitingTF):
            if i < 250:  
                 stateMatrix[3][249-i] = 1
                
        stateMatrix[3][250] = traci.trafficlight.getPhase("JRTL1")

        return stateMatrix
    
    # taking into account the length of the vehicles
    def getStateMatrixV2(self):
        # Read out length and number of lanes
        laneLength = traci.lane.getLength("HW_Ramp_0")
        laneNumber = traci.edge.getLaneNumber("HW_Ramp")

        # Create an array stateMatrix with zeros
        stateMatrix = [[0 for _ in range(int(laneLength) + 1)] for _ in range(int(laneNumber) + 1)]

        # Reading out the vehicles on the road
        vehicleList = traci.edge.getLastStepVehicleIDs("HW_Ramp")
        for vehID in vehicleList:
            
            lanePosition = traci.vehicle.getLanePosition(vehID)
            vehicleLength = traci.vehicle.getLength(vehID)
            # Splitting the string at the underscores and converting the last element into an integer
            lane = int(traci.vehicle.getLaneID(vehID).split('_')[-1])

            # Set the 1 at the corresponding position in the stateMatrix
            for i in range(int(vehicleLength)):
                if int(lanePosition) - i >= 0:
                    stateMatrix[lane][int(lanePosition) - i] = 1
                else:
                    break

        numberOfVehicleWaitingTF = traci.edge.getLastStepHaltingNumber("Ramp_beforeTL")

        for i in range(numberOfVehicleWaitingTF):
            if i < 250:  
                 stateMatrix[3][249-i] = 1
                
        stateMatrix[3][250] = traci.trafficlight.getPhase("JRTL1")

        return stateMatrix

    def getStatistics(self):
        return (self.steps, self.flow_steps, self.flows_HW, self.speeds_HW, self.densities_HW, self.travelTimes_HW, self.flows_Ramp, self.speeds_Ramp, self.densities_Ramp, self.travelTimes_Ramp, self.travelTimesSystem, self.trafficLightPhases)

    def close(self):
        traci.close()
