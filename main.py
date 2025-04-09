"""
Main entry point for the UAV path planning simulation.

This script allows running the simulation from the command line or starting
the web interface for interactive use.
"""

import os
import argparse
import logging
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Import the app for gunicorn
from web.app import app

# Ensure result directory exists
def ensure_result_dir():
    """Ensure the result directory exists."""
    Path('results').mkdir(exist_ok=True)

# Initialize directories
ensure_result_dir()

# Check if run in CLI or web mode
def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='UAV Path Planning Simulation')
    parser.add_argument('--cli', action='store_true', help='Run in command line mode')
    parser.add_argument('--algorithm', type=str, default='mcts', choices=['mcts', 'rrt', 'astar'], help='Algorithm to use')
    parser.add_argument('--num_users', type=int, default=20, help='Number of users')
    parser.add_argument('--sim_time', type=int, default=300, help='Simulation time in seconds')
    parser.add_argument('--compare', action='store_true', help='Compare MCTS, RRT, and A* algorithms')
    return parser.parse_args()

# Start the web server
def start_web_server():
    """Start the web server."""
    app.run(host='0.0.0.0', port=5000, debug=True)

def run_cli_mode(args):
    """Run in CLI mode."""
    logger.info("Running in CLI mode")
    # Import modules needed for CLI here to avoid circular imports
    from algorithms.mcts import MCTSAlgorithm
    from algorithms.rrt import RRTAlgorithm
    from algorithms.astar import AStarAlgorithm
    from simulation.environment import Environment
    from visualization.plotter import plot_trajectory, plot_energy_consumption, plot_comparison_metrics
    
    # Run the specified algorithm or comparison
    if args.compare:
        run_comparison(args, MCTSAlgorithm, RRTAlgorithm, AStarAlgorithm, Environment, plot_comparison_metrics)
    else:
        run_single_algorithm(args, MCTSAlgorithm, RRTAlgorithm, AStarAlgorithm, Environment, plot_trajectory, plot_energy_consumption)

def run_comparison(args, MCTSAlgorithm, RRTAlgorithm, AStarAlgorithm, Environment, plot_comparison_metrics):
    """Run comparison of algorithms."""
    logger.info("Comparing MCTS, RRT, and A* algorithms")
    
    # Create environment
    env = Environment()
    
    # Initialize algorithms
    mcts_algo = MCTSAlgorithm()
    rrt_algo = RRTAlgorithm()
    astar_algo = AStarAlgorithm()
    
    # Setup algorithms with the environment
    mcts_algo.setup(env)
    rrt_algo.setup(env)
    astar_algo.setup(env)
    
    # Run episodes for each algorithm
    mcts_metrics = mcts_algo.run_episode(max_steps=args.sim_time)
    
    # Reset environment for next algorithm
    env.reset()
    rrt_metrics = rrt_algo.run_episode(max_steps=args.sim_time)
    
    # Reset environment for next algorithm
    env.reset()
    astar_metrics = astar_algo.run_episode(max_steps=args.sim_time)
    
    # Prepare results for comparison
    results = {
        'MCTS': mcts_metrics,
        'RRT': rrt_metrics,
        'ASTAR': astar_metrics
    }
    
    # Generate comparison plots
    plot_comparison_metrics(results)
    
    # Log results
    logger.info("Comparison completed. Results saved to results directory.")

def run_single_algorithm(args, MCTSAlgorithm, RRTAlgorithm, AStarAlgorithm, Environment, plot_trajectory, plot_energy_consumption):
    """Run a single algorithm."""
    logger.info(f"Running {args.algorithm} algorithm")
    
    # Create environment
    env = Environment()
    
    # Initialize the selected algorithm
    algo = None
    if args.algorithm == 'mcts':
        algo = MCTSAlgorithm()
    elif args.algorithm == 'rrt':
        algo = RRTAlgorithm()
    elif args.algorithm == 'astar':
        algo = AStarAlgorithm()
    
    # Check if a valid algorithm was selected
    if algo is None:
        logger.error(f"Unknown algorithm: {args.algorithm}")
        return
    
    # Setup algorithm with the environment
    algo.setup(env)
    
    # Run episode
    metrics = algo.run_episode(max_steps=args.sim_time)
    
    # Generate plots
    trajectory = env.get_trajectory()
    energy_log = env.get_energy_log()
    
    plot_trajectory(trajectory, title=f"{args.algorithm.upper()} UAV Trajectory")
    plot_energy_consumption(energy_log, title=f"{args.algorithm.upper()} Energy Consumption")
    
    # Log results
    logger.info(f"Simulation completed. Results saved to results directory.")
    logger.info(f"Metrics: {metrics}")

if __name__ == '__main__':
    args = parse_args()
    
    if args.cli:
        run_cli_mode(args)
    else:
        logger.info("Starting web server")
        start_web_server()