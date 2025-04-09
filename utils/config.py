"""
Configuration parameters for the UAV path planning simulation.
"""

# Environment parameters
WORLD_SIZE = (1000, 1000)  # Size of the world in meters
NUM_USERS = 20  # Number of users in the simulation
USER_DATA_RATE = 2.0  # Data rate of each user in Mbps
USER_TASK_DURATION = 5.0  # Duration of each user task in seconds
USER_TASK_INTERVAL = 30.0  # Average interval between tasks in seconds
USER_TASK_SIZE = 10.0  # Size of each task in MB
USER_SPEED = 1.0  # Speed of users in m/s
USER_MOBILITY_TYPE = 'random_waypoint'  # Mobility model for users

# UAV parameters
UAV_INITIAL_POSITION = (500, 500)  # Initial position of the UAV in meters
UAV_SPEED = 5.0  # Speed of the UAV in m/s
UAV_INITIAL_ENERGY = 10000.0  # Initial energy of the UAV in J
UAV_HOVER_POWER = 100.0  # Power consumption when hovering in W
UAV_MOVE_POWER = 120.0  # Power consumption when moving in W
UAV_COMM_POWER = 50.0  # Power consumption when communicating in W
UAV_SERVICE_TIME = 3.0  # Time required to service a user in seconds
COMM_RANGE = 100.0  # Communication range in meters

# Simulation parameters
SIM_TIME = 300.0  # Simulation time in seconds
TIME_STEP = 0.1  # Time step in seconds
MAX_STEPS = int(SIM_TIME / TIME_STEP)  # Maximum number of steps

# Algorithm parameters
# MCTS parameters
MCTS_ITERATIONS = 100  # Number of iterations for MCTS
MCTS_EXPLORATION_WEIGHT = 1.0  # Exploration weight for UCT
MCTS_ROLLOUT_DEPTH = 20  # Depth of rollout in MCTS
MCTS_MAX_DEPTH = 100  # Maximum depth of tree in MCTS

# RRT parameters
RRT_MAX_ITERATIONS = 1000  # Maximum number of iterations for RRT
RRT_STEP_SIZE = 10.0  # Step size for RRT
RRT_GOAL_SAMPLE_RATE = 0.1  # Goal sample rate for RRT
RRT_CONNECT_CIRCLE_DISTANCE = 20.0  # Connect circle distance for RRT

# A* parameters (future implementation)
ASTAR_GRID_SIZE = 10.0  # Grid size for A* in meters
ASTAR_HEURISTIC = 'euclidean'  # Heuristic function for A*

# Output parameters
RESULT_DIR = 'results/'  # Directory for result files