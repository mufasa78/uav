"""
Environment for UAV path planning simulation.
"""

import math
import random
import numpy as np
from typing import Dict, List, Tuple, Any, Optional

from simulation.uav import UAV
from utils.config import (
    WORLD_SIZE,
    NUM_USERS,
    USER_DATA_RATE,
    USER_TASK_DURATION,
    USER_TASK_INTERVAL,
    USER_TASK_SIZE,
    USER_SPEED,
    USER_MOBILITY_TYPE,
    UAV_SERVICE_TIME,
    COMM_RANGE,
    TIME_STEP,
    MAX_STEPS
)

class Environment:
    """
    Environment for UAV path planning simulation.
    """
    
    def __init__(self):
        """
        Initialize the environment.
        """
        self.uav = UAV()
        self.world_size = WORLD_SIZE
        self.time_step = TIME_STEP
        self.max_steps = MAX_STEPS
        
        # State variables
        self.current_time = 0.0
        self.current_step = 0
        self.users = []  # List of (user_id, position, velocity, next_task_time)
        self.users_with_tasks = []  # List of user IDs with active tasks
        self.user_task_durations = {}  # Mapping of user ID to task duration
        self.user_data_sizes = {}  # Mapping of user ID to data size
        self.current_service_user = None  # Current user being serviced
        self.service_start_time = None  # Time when the current service started
        self.serviced_tasks = 0  # Number of tasks serviced
        self.data_processed = 0.0  # Amount of data processed in MB
        self.total_flight_distance = 0.0  # Total flight distance in meters
        
        # Performance tracking
        self.trajectory = []  # List of UAV positions
        self.energy_log = []  # List of UAV energy values
        self.stats_log = []  # List of performance metrics at each time step
    
    def reset(self) -> None:
        """
        Reset the environment.
        """
        self.uav = UAV()
        self.current_time = 0.0
        self.current_step = 0
        self.users = []
        self.users_with_tasks = []
        self.user_task_durations = {}
        self.user_data_sizes = {}
        self.current_service_user = None
        self.service_start_time = None
        self.serviced_tasks = 0
        self.data_processed = 0.0
        self.total_flight_distance = 0.0
        self.trajectory = []
        self.energy_log = []
        self.stats_log = []
        
        # Initialize users
        self._init_users()
        
        # Log initial state
        self._log_state()
    
    def _init_users(self) -> None:
        """
        Initialize the users.
        """
        for i in range(NUM_USERS):
            # Random position within the world
            x = random.uniform(0, self.world_size[0])
            y = random.uniform(0, self.world_size[1])
            position = (x, y)
            
            # Random velocity
            if USER_MOBILITY_TYPE == 'random_waypoint':
                angle = random.uniform(0, 2 * math.pi)
                vx = USER_SPEED * math.cos(angle)
                vy = USER_SPEED * math.sin(angle)
                velocity = (vx, vy)
            else:
                velocity = (0, 0)  # Static users
            
            # Random time for the next task
            next_task_time = random.uniform(0, USER_TASK_INTERVAL)
            
            # Add user
            self.users.append((i, position, velocity, next_task_time))
    
    def _log_state(self) -> None:
        """
        Log the current state of the environment.
        """
        # Log UAV position
        self.trajectory.append(self.uav.get_position())
        
        # Log UAV energy
        self.energy_log.append(self.uav.get_energy())
        
        # Log statistics
        stats = {
            'time': self.current_time,
            'uav_position': self.uav.get_position(),
            'uav_energy': self.uav.get_energy(),
            'serviced_tasks': self.serviced_tasks,
            'data_processed': self.data_processed,
            'users_with_tasks': len(self.users_with_tasks),
            'total_flight_distance': self.uav.get_total_distance()
        }
        
        self.stats_log.append(stats)
    
    def _update_users(self) -> None:
        """
        Update user positions and generate new tasks.
        """
        updated_users = []
        
        for user_id, position, velocity, next_task_time in self.users:
            # Update position based on velocity
            if USER_MOBILITY_TYPE == 'random_waypoint':
                new_x = position[0] + velocity[0] * self.time_step
                new_y = position[1] + velocity[1] * self.time_step
                
                # Check if we hit the boundary
                if new_x < 0 or new_x > self.world_size[0] or new_y < 0 or new_y > self.world_size[1]:
                    # Reflect the velocity
                    if new_x < 0 or new_x > self.world_size[0]:
                        velocity = (-velocity[0], velocity[1])
                    if new_y < 0 or new_y > self.world_size[1]:
                        velocity = (velocity[0], -velocity[1])
                    
                    # Recalculate position
                    new_x = max(0, min(self.world_size[0], new_x))
                    new_y = max(0, min(self.world_size[1], new_y))
                
                new_position = (new_x, new_y)
            else:
                new_position = position
            
            # Check if it's time for a new task
            if next_task_time <= self.current_time and user_id not in self.users_with_tasks:
                self.users_with_tasks.append(user_id)
                
                # Generate task parameters
                task_duration = random.uniform(0.5, USER_TASK_DURATION)
                data_size = random.uniform(0.5, USER_TASK_SIZE)
                
                self.user_task_durations[user_id] = task_duration
                self.user_data_sizes[user_id] = data_size
                
                # Set the next task time
                next_task_time = self.current_time + random.uniform(0.5, USER_TASK_INTERVAL)
            
            updated_users.append((user_id, new_position, velocity, next_task_time))
        
        self.users = updated_users
    
    def get_state(self) -> Dict[str, Any]:
        """
        Get the current state of the environment.
        
        Returns:
            Dictionary with the current state
        """
        return {
            'time': self.current_time,
            'uav_position': self.uav.get_position(),
            'uav_energy': self.uav.get_energy(),
            'all_users': self.users,
            'users_with_tasks': self.users_with_tasks,
            'current_service_user': self.current_service_user,
            'serviced_tasks': self.serviced_tasks,
            'data_processed': self.data_processed,
            'total_flight_distance': self.uav.get_total_distance()
        }
    
    def set_service_user(self, user_id: int) -> None:
        """
        Set the user to service.
        
        Args:
            user_id: ID of the user to service
        """
        if user_id in self.users_with_tasks:
            self.current_service_user = user_id
            self.service_start_time = self.current_time
            self.uav.set_servicing(True)
    
    def _check_service_completion(self) -> None:
        """
        Check if the current service is completed.
        """
        if self.current_service_user is not None and self.service_start_time is not None:
            # Check if the service duration is reached
            service_duration = self.current_time - self.service_start_time
            
            if service_duration >= UAV_SERVICE_TIME:
                # Service is completed
                self.serviced_tasks += 1
                self.data_processed += self.user_data_sizes.get(self.current_service_user, 0)
                
                # Remove the user from the list of users with tasks
                if self.current_service_user in self.users_with_tasks:
                    self.users_with_tasks.remove(self.current_service_user)
                
                # Reset service state
                self.current_service_user = None
                self.service_start_time = None
                self.uav.set_servicing(False)
    
    def step(self, target_position: Optional[Tuple[float, float]]) -> None:
        """
        Step the environment forward in time.
        
        Args:
            target_position: Target position for the UAV
        """
        # Update time
        self.current_time += self.time_step
        self.current_step += 1
        
        # Move the UAV
        if target_position is not None:
            self.uav.move_towards(target_position, self.time_step)
        
        # Update users
        self._update_users()
        
        # Check if service is completed
        self._check_service_completion()
        
        # Log state
        self._log_state()
    
    def is_done(self) -> bool:
        """
        Check if the simulation is done.
        
        Returns:
            True if done, False otherwise
        """
        # Check if maximum number of steps is reached
        if self.current_step >= self.max_steps:
            return True
        
        # Check if UAV is out of energy
        if self.uav.get_energy() <= 0:
            return True
        
        return False
    
    def get_metrics(self) -> Dict[str, float]:
        """
        Get the metrics of the simulation.
        
        Returns:
            Dictionary with metrics
        """
        # Calculate energy consumed
        energy_consumed = self.uav.get_energy() - self.energy_log[0]
        
        # Calculate average task delay
        
        # Calculate energy efficiency (in bits per Joule)
        energy_efficiency = 0.0
        if energy_consumed > 0:
            energy_efficiency = (self.data_processed * 8 * 1e6) / abs(energy_consumed)
        
        return {
            'serviced_tasks': self.serviced_tasks,
            'data_processed': self.data_processed,
            'total_flight_distance': self.uav.get_total_distance(),
            'energy_consumed': abs(energy_consumed),
            'energy_efficiency': energy_efficiency,
            'avg_task_delay': 0.0  # Placeholder for now
        }
    
    def get_trajectory(self) -> List[Tuple[float, float]]:
        """
        Get the trajectory of the UAV.
        
        Returns:
            List of UAV positions
        """
        return self.trajectory
    
    def get_energy_log(self) -> List[float]:
        """
        Get the energy log of the UAV.
        
        Returns:
            List of UAV energy values
        """
        return self.energy_log
    
    def get_stats_log(self) -> List[Dict[str, Any]]:
        """
        Get the statistics log.
        
        Returns:
            List of statistics at each time step
        """
        return self.stats_log