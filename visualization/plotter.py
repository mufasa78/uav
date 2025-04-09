"""
Plotting utilities for the UAV path planning simulation.
"""

import base64
from io import BytesIO
from typing import Dict, List, Tuple, Any, Optional, Union

import numpy as np
import matplotlib.pyplot as plt

def plot_trajectory(
    trajectory: Union[List[Tuple[float, float]], Dict[str, List[Tuple[float, float]]]],
    world_size: Tuple[float, float] = (1000, 1000),
    title: str = "UAV Trajectory"
) -> str:
    """
    Plot the UAV trajectory.
    
    Args:
        trajectory: Either a list of (x, y) positions or a dictionary with algorithm names as keys and trajectories as values
        world_size: Size of the world in meters
        title: Title of the plot
        
    Returns:
        Base64 encoded PNG image
    """
    fig = plt.figure(figsize=(10, 8))
    
    # Check if trajectory is a dictionary for comparison
    if isinstance(trajectory, dict):
        # Define colors for different algorithms
        colors = {'MCTS': 'blue', 'RRT': 'red', 'ASTAR': 'green'}
        
        # Plot each trajectory
        for alg_name, traj in trajectory.items():
            if not traj:
                continue
                
            x = [pos[0] for pos in traj]
            y = [pos[1] for pos in traj]
            plt.plot(x, y, '-', label=alg_name, color=colors.get(alg_name, 'gray'), linewidth=2)
            
            # Plot start and end points
            plt.scatter(x[0], y[0], color=colors.get(alg_name, 'gray'), marker='o', s=100, label=f"{alg_name} Start")
            plt.scatter(x[-1], y[-1], color=colors.get(alg_name, 'gray'), marker='x', s=100, label=f"{alg_name} End")
    else:
        # Plot single trajectory
        if trajectory:
            x = [pos[0] for pos in trajectory]
            y = [pos[1] for pos in trajectory]
            plt.plot(x, y, 'b-', linewidth=2)
            
            # Plot start and end points
            plt.scatter(x[0], y[0], color='green', marker='o', s=100, label="Start")
            plt.scatter(x[-1], y[-1], color='red', marker='x', s=100, label="End")
    
    # Set plot limits
    plt.xlim(0, world_size[0])
    plt.ylim(0, world_size[1])
    
    # Add grid and labels
    plt.grid(True)
    plt.xlabel('X (m)')
    plt.ylabel('Y (m)')
    plt.title(title)
    plt.legend()
    
    # Convert plot to base64 string
    buf = BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plot_base64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)
    
    return plot_base64

def plot_energy_consumption(
    energy_log: Union[List[float], Dict[str, List[float]]],
    time_steps: Optional[List[float]] = None,
    title: str = "UAV Energy Consumption"
) -> str:
    """
    Plot the UAV energy consumption.
    
    Args:
        energy_log: Either a list of energy values or a dictionary with algorithm names as keys and energy logs as values
        time_steps: List of time steps
        title: Title of the plot
        
    Returns:
        Base64 encoded PNG image
    """
    fig = plt.figure(figsize=(10, 6))
    
    # Check if energy_log is a dictionary for comparison
    if isinstance(energy_log, dict):
        # Define colors for different algorithms
        colors = {'MCTS': 'blue', 'RRT': 'red', 'ASTAR': 'green'}
        
        # Plot each energy log
        for alg_name, log in energy_log.items():
            if not log:
                continue
                
            if time_steps and len(time_steps) == len(log):
                x = time_steps
            else:
                x = range(len(log))
                
            plt.plot(x, log, '-', label=alg_name, color=colors.get(alg_name, 'gray'), linewidth=2)
    else:
        # Plot single energy log
        if energy_log:
            if time_steps and len(time_steps) == len(energy_log):
                x = time_steps
            else:
                x = range(len(energy_log))
                
            plt.plot(x, energy_log, 'b-', linewidth=2)
    
    # Add grid and labels
    plt.grid(True)
    plt.xlabel('Time Step')
    plt.ylabel('Energy (J)')
    plt.title(title)
    
    if isinstance(energy_log, dict):
        plt.legend()
    
    # Convert plot to base64 string
    buf = BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plot_base64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)
    
    return plot_base64

def plot_comparison_metrics(
    results: Dict[str, Dict[str, Any]],
    metrics_to_compare: List[str] = ['serviced_tasks', 'data_processed', 'total_flight_distance', 'energy_consumed', 'energy_efficiency'],
    title: str = "Algorithm Performance Comparison"
) -> str:
    """
    Plot comparison of algorithm metrics.
    
    Args:
        results: Dictionary with algorithm names as keys and metrics dictionaries as values
        metrics_to_compare: List of metric names to compare
        title: Title of the plot
        
    Returns:
        Base64 encoded PNG image
    """
    # Filter metrics to only include those we want to compare
    filtered_metrics = {}
    for alg_name, metrics in results.items():
        filtered_metrics[alg_name] = {k: v for k, v in metrics.items() if k in metrics_to_compare and isinstance(v, (int, float))}
    
    # Number of metrics to compare
    n_metrics = len(metrics_to_compare)
    if n_metrics == 0:
        return ""
    
    # Create figure
    fig, axes = plt.subplots(nrows=1, ncols=n_metrics, figsize=(n_metrics * 4, 6))
    
    # Handle case with only one metric
    if n_metrics == 1:
        axes = [axes]
    
    # Define colors for different algorithms
    colors = {'MCTS': 'royalblue', 'RRT': 'firebrick', 'ASTAR': 'forestgreen'}
    
    # Plot each metric
    for i, metric in enumerate(metrics_to_compare):
        ax = axes[i]
        
        # Extract values for this metric
        alg_names = []
        values = []
        
        for alg_name, metrics in filtered_metrics.items():
            if metric in metrics:
                alg_names.append(alg_name)
                values.append(metrics[metric])
        
        # Skip if no values
        if not alg_names:
            continue
        
        # Create bar plot
        bars = ax.bar(alg_names, values, color=[colors.get(name, 'gray') for name in alg_names])
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01 * max(values),
                    f'{height:.2f}', ha='center', va='bottom', rotation=0)
        
        # Set title and labels
        ax.set_title(metric.replace('_', ' ').title())
        ax.set_ylim(0, max(values) * 1.2)
        
        # Adjust x-axis labels
        ax.set_xticklabels(alg_names, rotation=45)
        
        # Add grid
        ax.grid(True, axis='y', linestyle='--', alpha=0.7)
    
    # Add overall title
    fig.suptitle(title, fontsize=16)
    
    # Adjust layout
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    # Convert plot to base64 string
    buf = BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plot_base64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)
    
    return plot_base64

def plot_trajectory_with_users(
    trajectory: List[Tuple[float, float]],
    user_positions: List[List[Tuple[float, float]]],
    world_size: Tuple[float, float] = (1000, 1000),
    service_positions: Optional[List[Tuple[float, float]]] = None,
    title: str = "UAV Trajectory with Users"
) -> str:
    """
    Plot the UAV trajectory with user positions.
    
    Args:
        trajectory: List of UAV (x, y) positions
        user_positions: List of lists of user (x, y) positions at each time step
        world_size: Size of the world in meters
        service_positions: List of positions where the UAV serviced users
        title: Title of the plot
        
    Returns:
        Base64 encoded PNG image
    """
    fig = plt.figure(figsize=(10, 8))
    
    # Plot UAV trajectory
    if trajectory:
        x = [pos[0] for pos in trajectory]
        y = [pos[1] for pos in trajectory]
        plt.plot(x, y, 'b-', linewidth=2, label="UAV Path")
        
        # Plot start and end points
        plt.scatter(x[0], y[0], color='green', marker='o', s=100, label="Start")
        plt.scatter(x[-1], y[-1], color='red', marker='x', s=100, label="End")
    
    # Plot user positions at final time step
    if user_positions and user_positions[-1]:
        for i, pos in enumerate(user_positions[-1]):
            plt.scatter(pos[0], pos[1], color='gray', marker='o', s=30)
            plt.text(pos[0] + 5, pos[1] + 5, f"User {i}")
    
    # Plot service positions
    if service_positions:
        for pos in service_positions:
            plt.scatter(pos[0], pos[1], color='purple', marker='*', s=150, label="Service")
    
    # Set plot limits
    plt.xlim(0, world_size[0])
    plt.ylim(0, world_size[1])
    
    # Add grid and labels
    plt.grid(True)
    plt.xlabel('X (m)')
    plt.ylabel('Y (m)')
    plt.title(title)
    plt.legend()
    
    # Convert plot to base64 string
    buf = BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plot_base64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)
    
    return plot_base64