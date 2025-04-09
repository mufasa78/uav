"""
Environment for UAV path planning simulation.
无人机路径规划模拟环境。
"""

import random
import math
import logging
from typing import Dict, List, Tuple, Any, Optional

from simulation.uav import UAV
from utils.config import (
    WORLD_SIZE,
    TIME_STEP,
    NUM_USERS,
    USER_TASK_PROBABILITY,
    USER_TASK_DATA_MIN,
    USER_TASK_DATA_MAX,
    USER_SERVICE_DISTANCE,
    USER_SERVICE_RATE
)

# Configure logging
logger = logging.getLogger(__name__)

class Environment:
    """
    Environment for UAV path planning simulation.
    无人机路径规划模拟环境。
    """
    
    def __init__(self):
        """
        Initialize the environment.
        初始化环境。
        """
        # UAV
        self.uav = UAV()
        
        # World size
        self.world_size = WORLD_SIZE
        
        # Simulation step
        self.current_step = 0
        
        # Users dictionary: { user_id: { position, has_task, task_data, ... } }
        self.users = {}
        
        # Service state
        self.current_service_user_id = None
        self.current_service_task_data_remaining = 0.0
        
        # Statistics
        self.serviced_tasks = 0
        self.data_processed = 0.0
        self.energy_consumed = 0.0
        
        # Logs
        self.trajectory = []
        self.energy_log = []
        self.stats_log = []
        
        # Initialize users
        self._init_users()
    
    def reset(self) -> None:
        """
        Reset the environment.
        重置环境。
        """
        # Reset UAV
        self.uav = UAV()
        
        # Reset simulation step
        self.current_step = 0
        
        # Reset users
        self.users = {}
        self._init_users()
        
        # Reset service state
        self.current_service_user_id = None
        self.current_service_task_data_remaining = 0.0
        
        # Reset statistics
        self.serviced_tasks = 0
        self.data_processed = 0.0
        self.energy_consumed = 0.0
        
        # Reset logs
        self.trajectory = []
        self.energy_log = []
        self.stats_log = []
    
    def _init_users(self) -> None:
        """
        Initialize the users.
        初始化用户。
        """
        for i in range(NUM_USERS):
            # Random position within the world
            position = (
                random.uniform(0, self.world_size[0]),
                random.uniform(0, self.world_size[1])
            )
            
            # Initialize user without task
            self.users[i] = {
                'position': position,
                'has_task': False,
                'task_data': 0.0  # Size of the task in MB
            }
    
    def _log_state(self) -> None:
        """
        Log the current state of the environment.
        """
        # Log UAV position
        self.trajectory.append(self.uav.get_position())
        
        # Log energy
        self.energy_log.append(self.uav.get_energy())
        
        # Log statistics
        stats = {
            'step': self.current_step,
            'uav_position': self.uav.get_position(),
            'uav_energy': self.uav.get_energy(),
            'servicing_user': self.current_service_user_id,
            'serviced_tasks': self.serviced_tasks,
            'data_processed': self.data_processed,
            'energy_consumed': self.energy_consumed
        }
        self.stats_log.append(stats)
    
    def _update_users(self) -> None:
        """
        Update user positions and generate new tasks.
        """
        for user_id in self.users:
            # Update position (users are stationary for now)
            
            # Generate new task with probability
            if not self.users[user_id]['has_task'] and random.random() < USER_TASK_PROBABILITY:
                self._generate_task(user_id)
    
    def _generate_task(self, user_id: int) -> None:
        """
        Generate a task for a user.
        
        Args:
            user_id: ID of the user
        """
        # Generate random task size
        task_data = random.uniform(USER_TASK_DATA_MIN, USER_TASK_DATA_MAX)
        
        # Update user
        self.users[user_id]['has_task'] = True
        self.users[user_id]['task_data'] = task_data
    
    def get_state(self) -> Dict[str, Any]:
        """
        Get the current state of the environment.
        
        Returns:
            Dictionary with the current state
        """
        return {
            'current_step': self.current_step,
            'world_size': self.world_size,
            'uav_position': self.uav.get_position(),
            'uav_energy': self.uav.get_energy(),
            'users': self.users.copy(),
            'current_service_user_id': self.current_service_user_id
        }
    
    def set_service_user(self, user_id: int) -> None:
        """
        Set the user to service.
        
        Args:
            user_id: ID of the user to service
        """
        # Check if valid user
        if user_id not in self.users:
            return
        
        # Check if user has task
        if not self.users[user_id]['has_task']:
            return
        
        # Check if already servicing this user
        if self.current_service_user_id == user_id:
            return
        
        # Check if UAV is close enough to the user
        uav_position = self.uav.get_position()
        user_position = self.users[user_id]['position']
        distance = math.sqrt((uav_position[0] - user_position[0]) ** 2 + (uav_position[1] - user_position[1]) ** 2)
        
        if distance > USER_SERVICE_DISTANCE:
            return
        
        # Start servicing
        self.current_service_user_id = user_id
        self.current_service_task_data_remaining = self.users[user_id]['task_data']
        self.uav.set_servicing(True)
    
    def _check_service_completion(self) -> None:
        """
        Check if the current service is completed.
        """
        if self.current_service_user_id is None:
            return
        
        # Check if UAV is still close enough to the user
        uav_position = self.uav.get_position()
        user_position = self.users[self.current_service_user_id]['position']
        distance = math.sqrt((uav_position[0] - user_position[0]) ** 2 + (uav_position[1] - user_position[1]) ** 2)
        
        if distance > USER_SERVICE_DISTANCE:
            # Stop servicing
            self.current_service_user_id = None
            self.current_service_task_data_remaining = 0.0
            self.uav.set_servicing(False)
            return
        
        # Process task
        data_processed = min(USER_SERVICE_RATE * TIME_STEP, self.current_service_task_data_remaining)
        self.current_service_task_data_remaining -= data_processed
        self.data_processed += data_processed
        
        # Check if task completed
        if self.current_service_task_data_remaining <= 0:
            # Mark task as completed
            self.users[self.current_service_user_id]['has_task'] = False
            self.users[self.current_service_user_id]['task_data'] = 0.0
            
            # Update statistics
            self.serviced_tasks += 1
            
            # Stop servicing
            self.current_service_user_id = None
            self.uav.set_servicing(False)
    
    def step(self, target_position: Optional[Tuple[float, float]]) -> None:
        """
        Step the environment forward in time.
        环境向前推进一个时间步长。
        
        Args:
            target_position: Target position for the UAV
                           无人机的目标位置
        """
        # Move UAV if target_position is provided
        if target_position is not None:
            moved, _ = self.uav.move_towards(target_position, TIME_STEP)
            
            # Calculate energy consumption for movement or hovering
            if moved:
                energy_consumed = self.uav.move_power * TIME_STEP
            else:
                energy_consumed = self.uav.hover_power * TIME_STEP
            
            # Additional energy for communication if servicing
            if self.uav.get_servicing_state():
                energy_consumed += self.uav.comm_power * TIME_STEP
            
            # Update energy
            self.uav.set_energy(self.uav.get_energy() - energy_consumed)
            self.energy_consumed += energy_consumed
        
        # Check service completion
        self._check_service_completion()
        
        # Update users
        self._update_users()
        
        # Log state
        self._log_state()
        
        # Increment step
        self.current_step += 1
    
    def is_done(self) -> bool:
        """
        Check if the simulation is done.
        
        Returns:
            True if done, False otherwise
        """
        # Done if UAV has no energy
        if self.uav.get_energy() <= 0:
            return True
        
        return False
    
    def get_metrics(self) -> Dict[str, float]:
        """
        Get the metrics of the simulation.
        获取模拟的度量指标。
        
        Returns:
            Dictionary with metrics 包含度量指标的字典
        """
        return {
            'serviced_tasks': self.serviced_tasks,
            'data_processed': self.data_processed,
            'energy_consumed': self.energy_consumed,
            'total_distance': self.uav.get_total_distance(),
            'remaining_energy': self.uav.get_energy()
        }
    
    def get_trajectory(self) -> List[Tuple[float, float]]:
        """
        Get the trajectory of the UAV.
        
        Returns:
            List of UAV positions
        """
        return self.trajectory.copy()
    
    def get_energy_log(self) -> List[float]:
        """
        Get the energy log of the UAV.
        
        Returns:
            List of UAV energy values
        """
        return self.energy_log.copy()
    
    def get_stats_log(self) -> List[Dict[str, Any]]:
        """
        Get the statistics log.
        
        Returns:
            List of statistics at each time step
        """
        return self.stats_log.copy()