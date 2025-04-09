"""
User module for UAV path planning simulation.

Implements the ground users with Random Waypoint (RWP) mobility model and task generation.
"""

import numpy as np
from typing import Tuple, List, Optional


class User:
    """
    Ground user with Random Waypoint mobility model.
    
    The user generates tasks at random intervals and moves according to the RWP model.
    """
    
    def __init__(
        self, 
        user_id: int, 
        position: Tuple[float, float], 
        world_size: Tuple[float, float],
        speed_range: Tuple[float, float],
        pause_time_range: Tuple[float, float],
        task_size_range: Tuple[float, float],
        task_interval_range: Tuple[float, float]
    ):
        """
        Initialize a ground user.
        
        Args:
            user_id: Unique identifier for the user
            position: Initial position (x, y)
            world_size: Size of the environment (width, height)
            speed_range: Min and max speed of the user
            pause_time_range: Min and max pause time at waypoints
            task_size_range: Min and max task size in megabits
            task_interval_range: Min and max interval between tasks in seconds
        """
        self.id = user_id
        self.position = np.array(position, dtype=float)
        self.world_size = np.array(world_size, dtype=float)
        self.speed_range = speed_range
        self.pause_time_range = pause_time_range
        self.task_size_range = task_size_range
        self.task_interval_range = task_interval_range
        
        # RWP model variables
        self.target_position = self._generate_random_waypoint()
        self.speed = self._generate_random_speed()
        self.direction = self._calculate_direction()
        self.pause_time = 0.0
        self.paused = False
        
        # Task variables
        self.task_active = False
        self.task_size = 0.0
        self.task_remaining = 0.0
        self.task_generated_time = 0.0
        self.next_task_time = self._generate_task_interval()
        self.tasks_history = []  # List of (time, size, completion_time) tuples
        
    def _generate_random_waypoint(self) -> np.ndarray:
        """Generate a random waypoint within the world boundaries."""
        return np.random.rand(2) * self.world_size
    
    def _generate_random_speed(self) -> float:
        """Generate a random speed within the specified range."""
        return np.random.uniform(self.speed_range[0], self.speed_range[1])
    
    def _generate_task_interval(self) -> float:
        """Generate a random interval for the next task."""
        return np.random.uniform(self.task_interval_range[0], self.task_interval_range[1])
    
    def _generate_task_size(self) -> float:
        """Generate a random task size within the specified range."""
        return np.random.uniform(self.task_size_range[0], self.task_size_range[1])
    
    def _calculate_direction(self) -> np.ndarray:
        """Calculate the normalized direction vector to the target position."""
        vector = self.target_position - self.position
        distance = np.linalg.norm(vector)
        if distance > 0:
            return vector / distance
        return np.zeros(2)
    
    def update(self, time_step: float, current_time: float) -> None:
        """
        Update the user's position and task status.
        
        Args:
            time_step: Time step for the update in seconds
            current_time: Current simulation time
        """
        # Update task status
        if not self.task_active:
            if current_time >= self.next_task_time:
                self.task_active = True
                self.task_size = self._generate_task_size()
                self.task_remaining = self.task_size
                self.task_generated_time = current_time
        
        # Update movement based on RWP model
        if self.paused:
            self.pause_time -= time_step
            if self.pause_time <= 0:
                self.paused = False
                self.target_position = self._generate_random_waypoint()
                self.speed = self._generate_random_speed()
                self.direction = self._calculate_direction()
        else:
            # Calculate the distance to move in this time step
            distance_to_move = self.speed * time_step
            
            # Calculate the distance to the target
            vector_to_target = self.target_position - self.position
            distance_to_target = np.linalg.norm(vector_to_target)
            
            if distance_to_move >= distance_to_target:
                # Reached the target, update position to target
                self.position = self.target_position.copy()
                
                # Pause at the target
                self.pause_time = np.random.uniform(self.pause_time_range[0], self.pause_time_range[1])
                self.paused = (self.pause_time > 0)
            else:
                # Move towards the target
                self.position += self.direction * distance_to_move
    
    def process_task(self, data_rate: float, time_step: float, current_time: float) -> float:
        """
        Process the active task.
        
        Args:
            data_rate: Data rate in megabits per second
            time_step: Time step for the update in seconds
            current_time: Current simulation time
            
        Returns:
            The amount of data processed in this time step
        """
        if not self.task_active:
            return 0.0
        
        data_processed = min(data_rate * time_step, self.task_remaining)
        self.task_remaining -= data_processed
        
        if self.task_remaining <= 0:
            self.task_active = False
            completion_time = current_time
            delay = completion_time - self.task_generated_time
            self.tasks_history.append((self.task_generated_time, self.task_size, delay))
            self.next_task_time = current_time + self._generate_task_interval()
        
        return data_processed
    
    def get_position(self) -> Tuple[float, float]:
        """Return the current position as a tuple."""
        return tuple(self.position)
    
    def get_task_status(self) -> Tuple[bool, float]:
        """Return the current task status (active, remaining)."""
        return self.task_active, self.task_remaining
    
    def get_delay_stats(self) -> Tuple[float, float]:
        """Return the average and maximum task delay."""
        if not self.tasks_history:
            return 0.0, 0.0
        
        delays = [task[2] for task in self.tasks_history]
        return np.mean(delays), np.max(delays)


class UserGroup:
    """
    Group of users with RWP mobility pattern.
    """
    
    def __init__(
        self,
        num_users: int,
        world_size: Tuple[float, float],
        speed_range: Tuple[float, float],
        pause_time_range: Tuple[float, float],
        task_size_range: Tuple[float, float],
        task_interval_range: Tuple[float, float]
    ):
        """
        Initialize a group of users.
        
        Args:
            num_users: Number of users
            world_size: Size of the environment (width, height)
            speed_range: Min and max speed of users
            pause_time_range: Min and max pause time at waypoints
            task_size_range: Min and max task size in megabits
            task_interval_range: Min and max interval between tasks in seconds
        """
        self.users = []
        
        for i in range(num_users):
            # Generate random initial position
            position = (
                np.random.uniform(0, world_size[0]),
                np.random.uniform(0, world_size[1])
            )
            
            user = User(
                user_id=i,
                position=position,
                world_size=world_size,
                speed_range=speed_range,
                pause_time_range=pause_time_range,
                task_size_range=task_size_range,
                task_interval_range=task_interval_range
            )
            
            self.users.append(user)
    
    def update(self, time_step: float, current_time: float) -> None:
        """
        Update all users.
        
        Args:
            time_step: Time step for the update in seconds
            current_time: Current simulation time
        """
        for user in self.users:
            user.update(time_step, current_time)
    
    def get_positions(self) -> List[Tuple[float, float]]:
        """Return the positions of all users."""
        return [user.get_position() for user in self.users]
    
    def get_users_with_tasks(self) -> List[int]:
        """Return the IDs of users with active tasks."""
        return [user.id for user in self.users if user.task_active]
    
    def get_user_positions_with_tasks(self) -> List[Tuple[int, Tuple[float, float], float]]:
        """Return the positions of users with active tasks."""
        result = []
        for user in self.users:
            if user.task_active:
                result.append((user.id, user.get_position(), user.task_remaining))
        return result
    
    def get_user_by_id(self, user_id: int) -> Optional[User]:
        """Get a user by ID."""
        for user in self.users:
            if user.id == user_id:
                return user
        return None
    
    def process_task(self, user_id: int, data_rate: float, time_step: float, current_time: float) -> float:
        """
        Process a task for a specific user.
        
        Args:
            user_id: ID of the user
            data_rate: Data rate in megabits per second
            time_step: Time step for the update in seconds
            current_time: Current simulation time
            
        Returns:
            The amount of data processed in this time step
        """
        user = self.get_user_by_id(user_id)
        if user:
            return user.process_task(data_rate, time_step, current_time)
        return 0.0
    
    def get_avg_task_delay(self) -> float:
        """Return the average task delay across all users."""
        total_delay = 0.0
        total_tasks = 0
        
        for user in self.users:
            if user.tasks_history:
                delays = [task[2] for task in user.tasks_history]
                total_delay += sum(delays)
                total_tasks += len(delays)
        
        if total_tasks > 0:
            return total_delay / total_tasks
        return 0.0
