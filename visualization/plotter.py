"""
Plotting utilities for the UAV path planning simulation.
"""

import base64
from io import BytesIO
from typing import Dict, List, Tuple, Any, Optional, Union

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

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

def plot_comparison_metrics(
    results: Dict[str, Dict[str, Any]],
    metrics: List[str] = ['average_reward', 'energy_consumed', 'serviced_tasks'],
    title: str = "Algorithm Performance Comparison"
) -> str:
    """
    Plot comparative performance metrics for different algorithms.

    Args:
        results: Dictionary with algorithm names as keys and their metrics as values
        metrics: List of metric names to plot
        title: Title of the plot

    Returns:
        Base64 encoded PNG image
    """
    fig, axes = plt.subplots(len(metrics), 1, figsize=(12, 4*len(metrics)), squeeze=False)
    colors = {'MCTS': 'blue', 'RRT': 'red', 'ASTAR': 'green'}

    for idx, metric in enumerate(metrics):
        ax = axes[idx, 0]

        # Extract metric values for each algorithm
        for alg_name, alg_results in results.items():
            if metric in alg_results:
                values = alg_results[metric] if isinstance(alg_results[metric], list) else [alg_results[metric]]
                steps = range(len(values))
                ax.plot(steps, values, '-', label=alg_name, color=colors.get(alg_name, 'gray'), linewidth=2)

        ax.grid(True)
        ax.set_xlabel('Time Steps')
        ax.set_ylabel(metric.replace('_', ' ').title())
        ax.legend()

    plt.tight_layout()

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

def plot_comprehensive_comparison(
    results: Dict[str, Dict[str, Any]],
    metrics_to_compare: List[str] = ['serviced_tasks', 'data_processed', 'total_distance', 'energy_consumed', 'energy_efficiency', 'task_completion_rate', 'avg_service_latency', 'performance_score'],
    title: str = "Comprehensive Algorithm Performance Comparison"
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

def plot_advanced_comparison(
    results: Dict[str, Dict[str, Any]],
    trajectories: Dict[str, List[Tuple[float, float]]],
    energy_logs: Dict[str, List[float]],
    world_size: Tuple[float, float] = (1000, 1000),
    title: str = "Advanced Algorithm Comparison"
) -> str:
    """
    Create an advanced visualization comparing multiple algorithms with trajectories and metrics.

    Args:
        results: Dictionary with algorithm names as keys and metrics dictionaries as values
        trajectories: Dictionary with algorithm names as keys and trajectory lists as values
        energy_logs: Dictionary with algorithm names as keys and energy log lists as values
        world_size: Size of the world in meters
        title: Title of the plot

    Returns:
        Base64 encoded PNG image
    """
    # Define key metrics to show
    key_metrics = [
        'serviced_tasks', 'data_processed', 'energy_consumed',
        'energy_efficiency', 'task_completion_rate', 'performance_score'
    ]

    # Create figure with grid layout
    fig = plt.figure(figsize=(15, 12))
    gs = GridSpec(3, 2, figure=fig, height_ratios=[2, 1, 1])

    # Plot trajectories in the top left
    ax_traj = fig.add_subplot(gs[0, 0])
    colors = {'MCTS': 'blue', 'QL': 'red', 'DQN': 'green'}

    for alg_name, traj in trajectories.items():
        if not traj:
            continue

        x = [pos[0] for pos in traj]
        y = [pos[1] for pos in traj]
        ax_traj.plot(x, y, '-', label=alg_name, color=colors.get(alg_name, 'gray'), linewidth=2)

        # Plot start and end points
        ax_traj.scatter(x[0], y[0], color=colors.get(alg_name, 'gray'), marker='o', s=80)
        ax_traj.scatter(x[-1], y[-1], color=colors.get(alg_name, 'gray'), marker='x', s=80)

    ax_traj.set_xlim(0, world_size[0])
    ax_traj.set_ylim(0, world_size[1])
    ax_traj.set_title('UAV Trajectories')
    ax_traj.grid(True)
    ax_traj.legend()

    # Plot energy consumption over time in the top right
    ax_energy = fig.add_subplot(gs[0, 1])

    for alg_name, energy_log in energy_logs.items():
        if not energy_log:
            continue

        x = range(len(energy_log))
        ax_energy.plot(x, energy_log, '-', label=alg_name, color=colors.get(alg_name, 'gray'), linewidth=2)

    ax_energy.set_title('Energy Consumption Over Time')
    ax_energy.set_xlabel('Time Steps')
    ax_energy.set_ylabel('Energy (J)')
    ax_energy.grid(True)
    ax_energy.legend()

    # Plot key metrics as bar charts in the bottom row
    alg_names = list(results.keys())

    # Create 2x3 grid for metrics in the bottom two rows
    for i, metric in enumerate(key_metrics):
        row = 1 + i // 2
        col = i % 2
        ax = fig.add_subplot(gs[row, col])

        # Extract values for this metric
        values = [results[alg].get(metric, 0) for alg in alg_names]

        # Create bar chart
        bars = ax.bar(alg_names, values, color=[colors.get(alg, 'gray') for alg in alg_names])

        # Add value labels on top of bars
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01 * max(values),
                    f'{value:.2f}', ha='center', va='bottom', fontsize=9)

        # Set title and labels
        ax.set_title(metric.replace('_', ' ').title())
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
    user_positions: List[Tuple[float, float]],  # Changed to accept static user positions
    user_tasks: Dict[int, bool] = None,  # Added parameter for user task status
    world_size: Tuple[float, float] = (1000, 1000),
    service_positions: Optional[List[Tuple[float, float]]] = None,
    title: str = "UAV Trajectory with Users"
) -> str:
    """
    Plot the UAV trajectory with user positions in the new two-row layout.

    Args:
        trajectory: List of UAV (x, y) positions
        user_positions: Dictionary of user positions {user_id: (x, y)}
        user_tasks: Dictionary of user task status {user_id: has_task}
        world_size: Size of the world in meters
        service_positions: List of positions where the UAV serviced users
        title: Title of the plot

    Returns:
        Base64 encoded PNG image
    """
    fig = plt.figure(figsize=(12, 9))

    # Plot UAV trajectory
    if trajectory:
        x = [pos[0] for pos in trajectory]
        y = [pos[1] for pos in trajectory]
        plt.plot(x, y, 'b-', linewidth=2, label="UAV Path")

        # Plot start and end points
        plt.scatter(x[0], y[0], color='green', marker='o', s=100, label="Start")
        plt.scatter(x[-1], y[-1], color='red', marker='x', s=100, label="End")

    # Plot user positions
    if user_positions:
        for user_id, pos in user_positions.items():
            # Determine color based on task status
            has_task = user_tasks.get(user_id, False) if user_tasks else False
            color = 'red' if has_task else 'blue'

            plt.scatter(pos[0], pos[1], color=color, marker='o', s=80, alpha=0.7)
            plt.text(pos[0] + 5, pos[1] + 5, f"User {user_id}", fontsize=8)

    # Plot service positions
    if service_positions:
        for pos in service_positions:
            plt.scatter(pos[0], pos[1], color='purple', marker='*', s=150)

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