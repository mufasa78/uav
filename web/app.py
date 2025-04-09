"""
Flask web application for UAV path planning visualization and control.
"""

import os
import io
import base64
import json
import logging
from typing import Dict, List, Tuple, Any, Optional

import numpy as np
from flask import Flask, render_template, request, jsonify
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

from algorithms.base import PathPlanningAlgorithm
from algorithms.mcts import MCTSAlgorithm
from algorithms.rrt import RRTAlgorithm
from simulation.environment import Environment
from visualization.plotter import (
    plot_trajectory, 
    plot_energy_consumption, 
    plot_comparative_trajectories, 
    plot_comparative_energy_consumption,
    plot_comparison_metrics,
    plot_simulation_progress
)
from utils.config import WORLD_SIZE, COMM_RANGE

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Create Flask app
app = Flask(__name__)

# Global state
algorithms = {
    'mcts': MCTSAlgorithm(),
    'rrt': RRTAlgorithm()
}

environments = {
    'mcts': Environment(),
    'rrt': Environment()
}

current_results = {
    'mcts': None,
    'rrt': None
}

def figure_to_base64(fig):
    """
    Convert a matplotlib figure to a base64 encoded string.
    
    Args:
        fig: Matplotlib figure
        
    Returns:
        Base64 encoded string
    """
    output = io.BytesIO()
    FigureCanvas(fig).print_png(output)
    plt.close(fig)
    return base64.b64encode(output.getvalue()).decode('utf-8')

@app.route('/')
def index():
    """Render the homepage."""
    return render_template('index.html')

@app.route('/simulation')
def simulation():
    """Render the simulation page."""
    return render_template('simulation.html')

@app.route('/comparison')
def comparison():
    """Render the comparison page."""
    return render_template('comparison.html')

@app.route('/documentation')
def documentation():
    """Render the documentation page."""
    return render_template('documentation.html')

@app.route('/api/initialize/<algorithm>', methods=['POST'])
def initialize_algorithm(algorithm):
    """
    Initialize an algorithm.
    
    Args:
        algorithm: Name of the algorithm
    """
    if algorithm not in algorithms:
        return jsonify({'error': f'Algorithm {algorithm} not found'}), 404
    
    # Reset the environment
    environments[algorithm].reset()
    
    # Set up the algorithm
    algorithms[algorithm].setup(environments[algorithm])
    
    return jsonify({'success': True})

@app.route('/api/step/<algorithm>', methods=['POST'])
def step_algorithm(algorithm):
    """
    Step an algorithm.
    
    Args:
        algorithm: Name of the algorithm
    """
    if algorithm not in algorithms:
        return jsonify({'error': f'Algorithm {algorithm} not found'}), 404
    
    # Get the state
    env = environments[algorithm]
    state = env.get_state()
    
    # Compute the action
    target_position, user_id = algorithms[algorithm].compute_action(state)
    
    # Set the user to service if specified
    if user_id is not None:
        env.set_service_user(user_id)
    
    # Take a step
    env.step(target_position)
    
    # Get the updated state
    state = env.get_state()
    
    # Render the state
    fig = plot_simulation_progress(
        state,
        env.get_trajectory(),
        COMM_RANGE,
        env.current_step,
        env.max_steps,
        title=f"{algorithm.upper()} Simulation Progress"
    )
    
    # Check if done
    done = env.is_done()
    
    # If done, calculate metrics
    if done:
        metrics = env.get_metrics()
        metrics['algorithm'] = algorithms[algorithm].name
        current_results[algorithm] = {
            'metrics': metrics,
            'trajectory': env.get_trajectory(),
            'energy_log': env.get_energy_log(),
            'time_steps': [i * env.time_step for i in range(len(env.get_energy_log()))]
        }
    
    response = {
        'state': state,
        'plot': figure_to_base64(fig),
        'done': done
    }
    
    return jsonify(response)

@app.route('/api/run/<algorithm>', methods=['POST'])
def run_algorithm(algorithm):
    """
    Run an algorithm to completion.
    
    Args:
        algorithm: Name of the algorithm
    """
    if algorithm not in algorithms:
        return jsonify({'error': f'Algorithm {algorithm} not found'}), 404
    
    # Reset the environment
    environments[algorithm].reset()
    
    # Set up the algorithm
    algorithms[algorithm].setup(environments[algorithm])
    
    # Run the episode
    metrics = algorithms[algorithm].run_episode()
    
    # Store the results
    current_results[algorithm] = {
        'metrics': metrics,
        'trajectory': environments[algorithm].get_trajectory(),
        'energy_log': environments[algorithm].get_energy_log(),
        'time_steps': [i * environments[algorithm].time_step for i in range(len(environments[algorithm].get_energy_log()))]
    }
    
    # Plot the trajectory
    trajectory_fig = plot_trajectory(
        environments[algorithm].get_trajectory(),
        WORLD_SIZE,
        title=f"{algorithm.upper()} Trajectory"
    )
    
    # Plot the energy consumption
    energy_fig = plot_energy_consumption(
        environments[algorithm].get_energy_log(),
        [i * environments[algorithm].time_step for i in range(len(environments[algorithm].get_energy_log()))],
        title=f"{algorithm.upper()} Energy Consumption"
    )
    
    response = {
        'metrics': metrics,
        'trajectory_plot': figure_to_base64(trajectory_fig),
        'energy_plot': figure_to_base64(energy_fig)
    }
    
    return jsonify(response)

@app.route('/api/compare', methods=['POST'])
def compare_algorithms():
    """
    Compare algorithms.
    """
    # Check if all algorithms have results
    for algorithm in algorithms:
        if current_results[algorithm] is None:
            return jsonify({'error': f'No results for {algorithm}'}), 400
    
    # Plot comparative trajectories
    trajectories = {alg: current_results[alg]['trajectory'] for alg in algorithms}
    comparative_trajectories_fig = plot_comparative_trajectories(
        trajectories,
        WORLD_SIZE,
        title="Comparative Trajectories"
    )
    
    # Plot comparative energy consumption
    energy_logs = {alg: current_results[alg]['energy_log'] for alg in algorithms}
    time_steps = {alg: current_results[alg]['time_steps'] for alg in algorithms}
    comparative_energy_fig = plot_comparative_energy_consumption(
        energy_logs,
        time_steps,
        title="Comparative Energy Consumption"
    )
    
    # Plot comparative metrics
    metrics = {alg: current_results[alg]['metrics'] for alg in algorithms}
    metrics_to_compare = [
        'total_flight_distance',
        'energy_consumed',
        'serviced_tasks',
        'energy_efficiency'
    ]
    comparative_metrics_fig = plot_comparison_metrics(
        metrics,
        metrics_to_compare,
        title="Algorithm Performance Comparison"
    )
    
    response = {
        'trajectories_plot': figure_to_base64(comparative_trajectories_fig),
        'energy_plot': figure_to_base64(comparative_energy_fig),
        'metrics_plot': figure_to_base64(comparative_metrics_fig),
        'metrics': metrics
    }
    
    return jsonify(response)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)