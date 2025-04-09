"""
Utilities for calculating and comparing metrics in UAV path planning.
"""

import numpy as np
import matplotlib.pyplot as plt

def calculate_path_length(path):
    """
    Calculate the total length of a path.
    
    Args:
        path: List of (x, y) positions
        
    Returns:
        Total path length
    """
    if len(path) < 2:
        return 0.0
    
    total_length = 0.0
    for i in range(1, len(path)):
        x1, y1 = path[i-1]
        x2, y2 = path[i]
        segment_length = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        total_length += segment_length
    
    return total_length

def calculate_energy_consumption(path, hover_power, move_power, comm_power, time_step, service_times):
    """
    Calculate the energy consumption for a UAV path.
    
    Args:
        path: List of (x, y) positions
        hover_power: Power consumption when hovering in W
        move_power: Power consumption when moving in W
        comm_power: Power consumption when communicating in W
        time_step: Time step in seconds
        service_times: List of times when the UAV is servicing a user
        
    Returns:
        Total energy consumption in J
    """
    total_energy = 0.0
    
    for i in range(1, len(path)):
        # Check if UAV is servicing a user at this time step
        t = i * time_step
        is_servicing = any(abs(t - service_time) < time_step for service_time in service_times)
        
        # Calculate power consumption
        x1, y1 = path[i-1]
        x2, y2 = path[i]
        distance = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        
        if distance < 0.1:  # Basically hovering
            power = hover_power
        else:
            power = move_power
        
        # Add communication power if servicing
        if is_servicing:
            power += comm_power
        
        # Calculate energy for this time step
        energy = power * time_step
        total_energy += energy
    
    return total_energy

def calculate_average_delay(task_arrival_times, task_service_times):
    """
    Calculate the average delay between task arrival and service.
    
    Args:
        task_arrival_times: List of task arrival times
        task_service_times: List of task service times
        
    Returns:
        Average delay in seconds
    """
    if not task_arrival_times or not task_service_times:
        return 0.0
    
    delays = []
    for arrival, service in zip(task_arrival_times, task_service_times):
        delay = service - arrival
        delays.append(delay)
    
    return np.mean(delays)

def calculate_energy_efficiency(data_processed, energy_consumed):
    """
    Calculate the energy efficiency in bits per Joule.
    
    Args:
        data_processed: Amount of data processed in bits
        energy_consumed: Energy consumed in Joules
        
    Returns:
        Energy efficiency in bits per Joule
    """
    if energy_consumed == 0:
        return 0.0
    
    return data_processed / energy_consumed

def compare_algorithms(results_dict):
    """
    Compare the performance of different algorithms.
    
    Args:
        results_dict: Dictionary with algorithm names as keys and results as values
        
    Returns:
        Dictionary with comparison metrics
    """
    comparison = {}
    
    # Extract metrics for each algorithm
    metrics = {}
    for alg, results in results_dict.items():
        metrics[alg] = {
            'total_flight_distance': results['total_flight_distance'],
            'energy_consumed': results['energy_consumed'],
            'serviced_tasks': results['serviced_tasks'],
            'data_processed': results['data_processed'],
            'avg_task_delay': results['avg_task_delay'],
            'energy_efficiency': results['energy_efficiency']
        }
    
    comparison['metrics'] = metrics
    
    # Determine which algorithm performs better for each metric
    if len(metrics) > 1:
        better_algorithm = {}
        
        # Compare total flight distance (lower is better)
        distances = {alg: metrics[alg]['total_flight_distance'] for alg in metrics}
        better_algorithm['total_flight_distance'] = min(distances, key=distances.get)
        
        # Compare energy consumed (lower is better)
        energy = {alg: metrics[alg]['energy_consumed'] for alg in metrics}
        better_algorithm['energy_consumed'] = min(energy, key=energy.get)
        
        # Compare serviced tasks (higher is better)
        tasks = {alg: metrics[alg]['serviced_tasks'] for alg in metrics}
        better_algorithm['serviced_tasks'] = max(tasks, key=tasks.get)
        
        # Compare average task delay (lower is better)
        delay = {alg: metrics[alg]['avg_task_delay'] for alg in metrics}
        better_algorithm['avg_task_delay'] = min(delay, key=delay.get)
        
        # Compare energy efficiency (higher is better)
        efficiency = {alg: metrics[alg]['energy_efficiency'] for alg in metrics}
        better_algorithm['energy_efficiency'] = max(efficiency, key=efficiency.get)
        
        comparison['better_algorithm'] = better_algorithm
    
    return comparison