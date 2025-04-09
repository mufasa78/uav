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
    parser.add_argument('--algorithm', type=str, default='mcts', choices=['mcts', 'rrt'], help='Algorithm to use')
    parser.add_argument('--num_users', type=int, default=20, help='Number of users')
    parser.add_argument('--sim_time', type=int, default=300, help='Simulation time in seconds')
    parser.add_argument('--compare', action='store_true', help='Compare MCTS and RRT algorithms')
    return parser.parse_args()

# Start the web server
def start_web_server():
    """Start the web server."""
    app.run(host='0.0.0.0', port=5000, debug=True)

if __name__ == '__main__':
    args = parse_args()
    
    if args.cli:
        logger.info("Running in CLI mode")
        # Import modules needed for CLI here to avoid circular imports
        from algorithms.mcts import MCTSAlgorithm
        from algorithms.rrt import RRTAlgorithm
        from simulation.environment import Environment
        from visualization.plotter import plot_trajectory, plot_energy_consumption, plot_comparison_metrics
        
        # Run the specified algorithm or comparison
        if args.compare:
            logger.info("Comparing MCTS and RRT algorithms")
            # Logic for comparing algorithms
        else:
            logger.info(f"Running {args.algorithm} algorithm")
            # Logic for running a single algorithm
    else:
        logger.info("Starting web server")
        start_web_server()