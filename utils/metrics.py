"""
Metrics utilities for the UAV path planning simulation.
"""

from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import matplotlib.pyplot as plt

def calculate_distance(pos1: Tuple[float, float], pos2: Tuple[float, float]) -> float:
    """
    Calculate the Euclidean distance between two positions.
    
    Args:
        pos1: First position (x, y)
        pos2: Second position (x, y)
        
    Returns:
        Euclidean distance
    """
    return np.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)

def calculate_path_length(trajectory: List[Tuple[float, float]]) -> float:
    """
    Calculate the total length of a path.
    
    Args:
        trajectory: List of positions (x, y)
        
    Returns:
        Total path length
    """
    if not trajectory or len(trajectory) < 2:
        return 0.0
    
    total_length = 0.0
    for i in range(1, len(trajectory)):
        total_length += calculate_distance(trajectory[i-1], trajectory[i])
    
    return total_length

def calculate_average_speed(trajectory: List[Tuple[float, float]], time_steps: List[float]) -> float:
    """
    Calculate the average speed along a trajectory.
    
    Args:
        trajectory: List of positions (x, y)
        time_steps: List of time steps
        
    Returns:
        Average speed
    """
    if not trajectory or len(trajectory) < 2 or len(trajectory) != len(time_steps):
        return 0.0
    
    total_distance = calculate_path_length(trajectory)
    total_time = time_steps[-1] - time_steps[0]
    
    if total_time <= 0:
        return 0.0
    
    return total_distance / total_time

def calculate_energy_efficiency(energy_consumed: float, data_processed: float) -> float:
    """
    Calculate the energy efficiency in terms of data processed per unit of energy.
    
    Args:
        energy_consumed: Energy consumed in Joules
        data_processed: Data processed in bits
        
    Returns:
        Energy efficiency in bits per Joule
    """
    if energy_consumed <= 0:
        return 0.0
    
    return data_processed / energy_consumed

def calculate_service_rate(serviced_tasks: int, total_tasks: int) -> float:
    """
    Calculate the service rate as the ratio of serviced tasks to total tasks.
    
    Args:
        serviced_tasks: Number of serviced tasks
        total_tasks: Total number of tasks
        
    Returns:
        Service rate (0.0 to 1.0)
    """
    if total_tasks <= 0:
        return 0.0
    
    return serviced_tasks / total_tasks

def calculate_average_task_delay(task_delays: List[float]) -> float:
    """
    Calculate the average delay between task arrival and service.
    
    Args:
        task_delays: List of task delays
        
    Returns:
        Average task delay
    """
    if not task_delays:
        return 0.0
    
    return sum(task_delays) / len(task_delays)

def calculate_energy_per_distance(energy_consumed: float, total_distance: float) -> float:
    """
    Calculate the energy consumed per unit distance.
    
    Args:
        energy_consumed: Energy consumed in Joules
        total_distance: Total distance traveled in meters
        
    Returns:
        Energy per distance in Joules per meter
    """
    if total_distance <= 0:
        return 0.0
    
    return energy_consumed / total_distance

def calculate_energy_per_task(energy_consumed: float, serviced_tasks: int) -> float:
    """
    Calculate the energy consumed per task serviced.
    
    Args:
        energy_consumed: Energy consumed in Joules
        serviced_tasks: Number of serviced tasks
        
    Returns:
        Energy per task in Joules per task
    """
    if serviced_tasks <= 0:
        return 0.0
    
    return energy_consumed / serviced_tasks

def calculate_metrics_from_simulation(stats_log: List[Dict[str, Any]], trajectory: List[Tuple[float, float]], energy_log: List[float]) -> Dict[str, float]:
    """
    Calculate metrics from simulation logs.
    
    Args:
        stats_log: List of statistics at each time step
        trajectory: List of UAV positions
        energy_log: List of UAV energy values
        
    Returns:
        Dictionary with calculated metrics
    """
    metrics = {}
    
    # Basic metrics
    total_flight_distance = calculate_path_length(trajectory)
    metrics['total_flight_distance'] = total_flight_distance
    
    # Energy metrics
    initial_energy = energy_log[0] if energy_log else 0.0
    final_energy = energy_log[-1] if energy_log else 0.0
    energy_consumed = initial_energy - final_energy
    metrics['energy_consumed'] = energy_consumed
    
    # Task metrics
    serviced_tasks = 0
    data_processed = 0.0
    task_delays = []
    
    for stats in stats_log:
        if 'serviced_task' in stats and stats['serviced_task']:
            serviced_tasks += 1
            
        if 'data_processed' in stats:
            data_processed += stats.get('data_processed', 0.0)
            
        if 'task_delay' in stats and stats['task_delay'] is not None:
            task_delays.append(stats['task_delay'])
    
    metrics['serviced_tasks'] = serviced_tasks
    metrics['data_processed'] = data_processed
    
    # Calculate derived metrics
    if energy_consumed > 0:
        # Convert data from MB to bits for energy efficiency
        metrics['energy_efficiency'] = calculate_energy_efficiency(energy_consumed, data_processed * 8 * 1024 * 1024)
    else:
        metrics['energy_efficiency'] = 0.0
        
    if task_delays:
        metrics['avg_task_delay'] = calculate_average_task_delay(task_delays)
    else:
        metrics['avg_task_delay'] = 0.0
        
    if serviced_tasks > 0:
        metrics['energy_per_task'] = calculate_energy_per_task(energy_consumed, serviced_tasks)
    else:
        metrics['energy_per_task'] = 0.0
        
    if total_flight_distance > 0:
        metrics['energy_per_distance'] = calculate_energy_per_distance(energy_consumed, total_flight_distance)
    else:
        metrics['energy_per_distance'] = 0.0
    
    return metrics

def compare_algorithms(results: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """
    Compare metrics from multiple algorithms.
    
    Args:
        results: Dictionary with algorithm names as keys and metrics dictionaries as values
        
    Returns:
        Dictionary with comparison results
    """
    comparison = {}
    
    # List of metrics to compare
    metrics_to_compare = [
        'serviced_tasks',
        'data_processed',
        'total_flight_distance',
        'energy_consumed',
        'energy_efficiency',
        'avg_task_delay'
    ]
    
    # Collect metric values for each algorithm
    metric_values = {}
    for metric in metrics_to_compare:
        metric_values[metric] = {alg: results[alg].get(metric, 0.0) for alg in results}
    
    # Calculate best algorithm for each metric
    best_algorithm = {}
    for metric in metrics_to_compare:
        values = {alg: metric_values[metric][alg] for alg in results}
        
        # For metrics where lower is better
        if metric in ['total_flight_distance', 'energy_consumed', 'avg_task_delay']:
            best_alg = min(values, key=lambda k: values[k])
        # For metrics where higher is better
        else:
            best_alg = max(values, key=lambda k: values[k])
            
        best_algorithm[metric] = best_alg
    
    # Calculate improvement percentages
    improvements = {}
    for metric in metrics_to_compare:
        improvements[metric] = {}
        
        for alg1 in results:
            improvements[metric][alg1] = {}
            val1 = metric_values[metric][alg1]
            
            for alg2 in results:
                if alg1 == alg2:
                    improvements[metric][alg1][alg2] = 0.0
                    continue
                
                val2 = metric_values[metric][alg2]
                
                if val2 == 0:
                    improvements[metric][alg1][alg2] = 0.0
                    continue
                
                # For metrics where lower is better
                if metric in ['total_flight_distance', 'energy_consumed', 'avg_task_delay']:
                    improvement = (val2 - val1) / val2 * 100.0
                # For metrics where higher is better
                else:
                    improvement = (val1 - val2) / val2 * 100.0
                
                improvements[metric][alg1][alg2] = improvement
    
    comparison['metric_values'] = metric_values
    comparison['best_algorithm'] = best_algorithm
    comparison['improvements'] = improvements
    
    return comparison