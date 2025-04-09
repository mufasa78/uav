"""
Visualization utilities for UAV path planning simulation.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Any

def plot_trajectory(trajectory, world_size, users=None, connection_range=None, title="UAV Trajectory"):
    """
    Plot the trajectory of the UAV.
    
    Args:
        trajectory: List of (x, y) positions
        world_size: Tuple of (width, height)
        users: Optional list of user positions
        connection_range: Optional communication range
        title: Title for the plot
        
    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Plot the trajectory
    x = [pos[0] for pos in trajectory]
    y = [pos[1] for pos in trajectory]
    ax.plot(x, y, 'b-', linewidth=2, alpha=0.7, label='UAV Path')
    
    # Plot start and end points
    ax.plot(x[0], y[0], 'go', markersize=10, label='Start')
    ax.plot(x[-1], y[-1], 'ro', markersize=10, label='End')
    
    # Plot users if provided
    if users:
        user_x = [user[0] for user in users]
        user_y = [user[1] for user in users]
        ax.scatter(user_x, user_y, c='purple', marker='^', s=100, label='Users')
    
    # Plot connection range if provided
    if connection_range and len(trajectory) > 0:
        circle = plt.Circle(trajectory[-1], connection_range, color='g', fill=False, linestyle='--', alpha=0.5)
        ax.add_artist(circle)
        ax.text(trajectory[-1][0], trajectory[-1][1] + connection_range/10, f"Range: {connection_range}m", 
                ha='center', va='bottom', color='g')
    
    # Set plot limits and labels
    ax.set_xlim(0, world_size[0])
    ax.set_ylim(0, world_size[1])
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_title(title)
    ax.grid(True)
    ax.legend()
    
    # Make plot look better in dark mode
    fig.patch.set_facecolor('#222222')
    ax.set_facecolor('#333333')
    ax.tick_params(colors='white')
    ax.xaxis.label.set_color('white')
    ax.yaxis.label.set_color('white')
    ax.title.set_color('white')
    ax.spines['bottom'].set_color('white')
    ax.spines['top'].set_color('white')
    ax.spines['left'].set_color('white')
    ax.spines['right'].set_color('white')
    
    fig.tight_layout()
    return fig

def plot_energy_consumption(energy_log, time_steps, title="Energy Consumption"):
    """
    Plot the energy consumption over time.
    
    Args:
        energy_log: List of energy values
        time_steps: List of time steps
        title: Title for the plot
        
    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot energy consumption
    ax.plot(time_steps, energy_log, 'r-', linewidth=2)
    
    # Plot initial and final energy
    ax.plot(time_steps[0], energy_log[0], 'go', markersize=8, label='Initial Energy')
    ax.plot(time_steps[-1], energy_log[-1], 'bo', markersize=8, label='Final Energy')
    
    # Add energy consumption
    energy_consumed = energy_log[0] - energy_log[-1]
    ax.text(0.05, 0.05, f"Energy Consumed: {energy_consumed:.2f} J", 
            transform=ax.transAxes, ha='left', va='bottom', 
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Set plot limits and labels
    ax.set_xlim(0, time_steps[-1])
    ax.set_ylim(0, energy_log[0] * 1.1)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Energy (J)')
    ax.set_title(title)
    ax.grid(True)
    ax.legend()
    
    # Make plot look better in dark mode
    fig.patch.set_facecolor('#222222')
    ax.set_facecolor('#333333')
    ax.tick_params(colors='white')
    ax.xaxis.label.set_color('white')
    ax.yaxis.label.set_color('white')
    ax.title.set_color('white')
    ax.spines['bottom'].set_color('white')
    ax.spines['top'].set_color('white')
    ax.spines['left'].set_color('white')
    ax.spines['right'].set_color('white')
    
    fig.tight_layout()
    return fig

def plot_comparative_trajectories(trajectories, world_size, title="Comparative Trajectories"):
    """
    Plot multiple trajectories for comparison.
    
    Args:
        trajectories: Dictionary mapping algorithm names to lists of positions
        world_size: Tuple of (width, height)
        title: Title for the plot
        
    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Colors for different algorithms
    colors = {
        'mcts': 'blue',
        'rrt': 'green',
        'astar': 'red'
    }
    
    # Plot each trajectory
    for alg, path in trajectories.items():
        x = [pos[0] for pos in path]
        y = [pos[1] for pos in path]
        color = colors.get(alg.lower(), 'purple')
        ax.plot(x, y, color=color, linewidth=2, alpha=0.7, label=f"{alg}")
        
        # Plot start and end points
        ax.plot(x[0], y[0], 'o', color=color, markersize=8)
        ax.plot(x[-1], y[-1], 's', color=color, markersize=8)
    
    # Set plot limits and labels
    ax.set_xlim(0, world_size[0])
    ax.set_ylim(0, world_size[1])
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_title(title)
    ax.grid(True)
    ax.legend()
    
    # Make plot look better in dark mode
    fig.patch.set_facecolor('#222222')
    ax.set_facecolor('#333333')
    ax.tick_params(colors='white')
    ax.xaxis.label.set_color('white')
    ax.yaxis.label.set_color('white')
    ax.title.set_color('white')
    ax.spines['bottom'].set_color('white')
    ax.spines['top'].set_color('white')
    ax.spines['left'].set_color('white')
    ax.spines['right'].set_color('white')
    
    fig.tight_layout()
    return fig

def plot_comparative_energy_consumption(energy_logs, time_steps, title="Comparative Energy Consumption"):
    """
    Plot multiple energy consumption logs for comparison.
    
    Args:
        energy_logs: Dictionary mapping algorithm names to lists of energy values
        time_steps: Dictionary mapping algorithm names to lists of time steps
        title: Title for the plot
        
    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Colors for different algorithms
    colors = {
        'mcts': 'blue',
        'rrt': 'green',
        'astar': 'red'
    }
    
    # Plot each energy log
    for alg, energy in energy_logs.items():
        color = colors.get(alg.lower(), 'purple')
        time = time_steps[alg]
        
        # Make sure the time steps and energy log are the same length
        min_len = min(len(time), len(energy))
        time = time[:min_len]
        energy = energy[:min_len]
        
        ax.plot(time, energy, color=color, linewidth=2, label=f"{alg}")
    
    # Set plot limits and labels
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Energy (J)')
    ax.set_title(title)
    ax.grid(True)
    ax.legend()
    
    # Make plot look better in dark mode
    fig.patch.set_facecolor('#222222')
    ax.set_facecolor('#333333')
    ax.tick_params(colors='white')
    ax.xaxis.label.set_color('white')
    ax.yaxis.label.set_color('white')
    ax.title.set_color('white')
    ax.spines['bottom'].set_color('white')
    ax.spines['top'].set_color('white')
    ax.spines['left'].set_color('white')
    ax.spines['right'].set_color('white')
    
    fig.tight_layout()
    return fig

def plot_comparison_metrics(metrics, metrics_to_compare, title="Algorithm Performance Comparison"):
    """
    Plot a bar chart comparing metrics across algorithms.
    
    Args:
        metrics: Dictionary mapping algorithm names to dictionaries of metrics
        metrics_to_compare: List of metrics to compare
        title: Title for the plot
        
    Returns:
        Matplotlib figure
    """
    algorithms = list(metrics.keys())
    n_metrics = len(metrics_to_compare)
    
    # Create a figure with a subplot for each metric
    fig, axes = plt.subplots(n_metrics, 1, figsize=(10, 3 * n_metrics))
    
    # Make axes iterable if there's only one subplot
    if n_metrics == 1:
        axes = [axes]
    
    # Labels for metrics
    metric_labels = {
        'total_flight_distance': 'Total Flight Distance (m)',
        'energy_consumed': 'Energy Consumed (J)',
        'serviced_tasks': 'Serviced Tasks',
        'avg_task_delay': 'Average Task Delay (s)',
        'energy_efficiency': 'Energy Efficiency (bits/J)'
    }
    
    # Colors for different algorithms
    colors = {
        'mcts': 'blue',
        'rrt': 'green',
        'astar': 'red'
    }
    
    # Plot each metric
    for i, metric in enumerate(metrics_to_compare):
        ax = axes[i]
        
        # Get values for this metric across algorithms
        values = [metrics[alg][metric] for alg in algorithms]
        
        # Create bars
        bar_positions = range(len(algorithms))
        bars = ax.bar(bar_positions, values, width=0.6)
        
        # Color the bars
        for j, bar in enumerate(bars):
            bar.set_color(colors.get(algorithms[j].lower(), 'purple'))
        
        # Add value labels on top of bars
        for j, v in enumerate(values):
            ax.text(j, v, f"{v:.2f}", ha='center', va='bottom', fontsize=9)
        
        # Set axis labels and title
        ax.set_ylabel(metric_labels.get(metric, metric))
        ax.set_xticks(bar_positions)
        ax.set_xticklabels(algorithms)
        ax.set_title(f"{metric_labels.get(metric, metric)}")
        ax.grid(True, axis='y')
        
        # Make plot look better in dark mode
        ax.set_facecolor('#333333')
        ax.tick_params(colors='white')
        ax.xaxis.label.set_color('white')
        ax.yaxis.label.set_color('white')
        ax.title.set_color('white')
        ax.spines['bottom'].set_color('white')
        ax.spines['top'].set_color('white')
        ax.spines['left'].set_color('white')
        ax.spines['right'].set_color('white')
    
    # Set a main title
    fig.suptitle(title, fontsize=16, color='white')
    
    # Make the figure look better in dark mode
    fig.patch.set_facecolor('#222222')
    
    fig.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust for the main title
    return fig

def plot_simulation_progress(state, trajectory, comm_range, current_step, max_steps, title="Simulation Progress"):
    """
    Plot the current state of the simulation.
    
    Args:
        state: Dictionary with the current state of the environment
        trajectory: List of (x, y) positions
        comm_range: Communication range
        current_step: Current time step
        max_steps: Maximum number of steps
        title: Title for the plot
        
    Returns:
        Matplotlib figure
    """
    # Create a figure with a specified size
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Extract data from state
    if 'uav_position' in state:
        uav_position = state['uav_position']
    elif len(trajectory) > 0:
        uav_position = trajectory[-1]
    else:
        uav_position = (0, 0)
    
    # Get user positions and status
    users_with_tasks = state.get('users_with_tasks', [])
    all_users = state.get('all_users', [])
    current_user = state.get('current_user', None)
    
    # Plot all users
    if all_users:
        user_positions = [user[1] for user in all_users]
        for i, pos in enumerate(user_positions):
            if i in users_with_tasks:
                # User with an active task
                ax.scatter(pos[0], pos[1], c='orange', marker='^', s=100)
                ax.text(pos[0], pos[1] - 20, f"User {i}", fontsize=8, ha='center', color='white')
            else:
                # User without an active task
                ax.scatter(pos[0], pos[1], c='gray', marker='^', s=80, alpha=0.5)
                ax.text(pos[0], pos[1] - 20, f"User {i}", fontsize=8, ha='center', color='gray')
            
            # Highlight the current user being serviced
            if i == current_user:
                ax.add_artist(plt.Circle(pos, 30, color='yellow', fill=False, linestyle='-', linewidth=2))
                ax.text(pos[0], pos[1] + 30, "Servicing", fontsize=10, ha='center', color='yellow')
    
    # Plot the UAV trajectory
    if trajectory:
        x = [pos[0] for pos in trajectory]
        y = [pos[1] for pos in trajectory]
        ax.plot(x, y, 'b-', linewidth=1.5, alpha=0.7)
    
    # Plot the UAV's current position and communication range
    ax.scatter(uav_position[0], uav_position[1], c='red', marker='o', s=120)
    ax.text(uav_position[0], uav_position[1] + 20, "UAV", fontsize=10, ha='center', color='white')
    
    # Add communication range circle
    circle = plt.Circle(uav_position, comm_range, color='green', fill=False, linestyle='--', alpha=0.7)
    ax.add_artist(circle)
    
    # Add progress information
    progress_text = f"Step: {current_step}/{max_steps} ({current_step/max_steps*100:.1f}%)"
    energy_text = f"Energy: {state.get('uav_energy', 0):.1f} J"
    tasks_text = f"Tasks Serviced: {state.get('serviced_tasks', 0)}"
    
    ax.text(0.02, 0.98, progress_text, transform=ax.transAxes, fontsize=10, va='top', color='white',
            bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))
    ax.text(0.02, 0.93, energy_text, transform=ax.transAxes, fontsize=10, va='top', color='white',
            bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))
    ax.text(0.02, 0.88, tasks_text, transform=ax.transAxes, fontsize=10, va='top', color='white',
            bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))
    
    # Set axis limits, labels, and title
    world_size = (1000, 1000)  # Default world size
    ax.set_xlim(0, world_size[0])
    ax.set_ylim(0, world_size[1])
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_title(title)
    ax.grid(True)
    
    # Make plot look better in dark mode
    fig.patch.set_facecolor('#222222')
    ax.set_facecolor('#333333')
    ax.tick_params(colors='white')
    ax.xaxis.label.set_color('white')
    ax.yaxis.label.set_color('white')
    ax.title.set_color('white')
    ax.spines['bottom'].set_color('white')
    ax.spines['top'].set_color('white')
    ax.spines['left'].set_color('white')
    ax.spines['right'].set_color('white')
    
    fig.tight_layout()
    return fig