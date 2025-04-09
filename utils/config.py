"""
Configuration parameters for the UAV path planning simulation.
"""

# World parameters
WORLD_SIZE = (1000, 1000)  # Size of the world in meters

# UAV parameters
UAV_INITIAL_POSITION = (500, 500)  # Initial position of the UAV
UAV_SPEED = 10.0  # UAV speed in m/s
UAV_INITIAL_ENERGY = 10000.0  # Initial energy in Joules
UAV_HOVER_POWER = 100.0  # Power consumption when hovering in Watts
UAV_MOVE_POWER = 150.0  # Power consumption when moving in Watts
UAV_COMM_POWER = 50.0  # Power consumption when communicating in Watts

# User parameters
NUM_USERS = 20  # Number of users
USER_TASK_PROBABILITY = 0.01  # Probability of a user generating a task in each time step
USER_TASK_DATA_MIN = 10.0  # Minimum data size in MB
USER_TASK_DATA_MAX = 50.0  # Maximum data size in MB
USER_MOBILITY_PROBABILITY = 0.05  # Probability of a user moving in each time step
USER_MOBILITY_DISTANCE = 20.0  # Maximum distance a user can move in meters

# Simulation parameters
TIME_STEP = 1.0  # Time step in seconds
DATA_TRANSFER_RATE = 10.0  # Data transfer rate in MB/s
SERVICE_RANGE = 10.0  # Range within which the UAV can service a user in meters
USER_SERVICE_DISTANCE = 10.0  # Distance within which the UAV can service a user
USER_SERVICE_RATE = 5.0  # Data transfer rate when servicing a user in MB/s

# MCTS parameters
MCTS_ITERATIONS = 100  # Number of iterations for MCTS
MCTS_EXPLORATION_WEIGHT = 1.0  # Exploration weight for UCT
MCTS_ROLLOUT_DEPTH = 20  # Maximum depth for rollout simulation
MCTS_MAX_DEPTH = 50  # Maximum depth of the MCTS tree

# RRT parameters
RRT_MAX_ITERATIONS = 1000  # Maximum number of iterations for tree building
RRT_STEP_SIZE = 20.0  # Step size for extending the tree
RRT_GOAL_SAMPLE_RATE = 0.1  # Probability of sampling the goal position
RRT_CONNECT_CIRCLE_DISTANCE = 20.0  # Maximum distance to connect two nodes