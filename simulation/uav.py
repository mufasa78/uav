"""
UAV model for path planning simulation.
"""

import math
from typing import Tuple, Optional

from utils.config import (
    UAV_INITIAL_POSITION,
    UAV_SPEED,
    UAV_INITIAL_ENERGY,
    UAV_HOVER_POWER,
    UAV_MOVE_POWER,
    UAV_COMM_POWER
)

class UAV:
    """
    Unmanned Aerial Vehicle model for path planning simulation.
    """
    
    def __init__(
        self, 
        initial_position: Tuple[float, float] = UAV_INITIAL_POSITION, 
        speed: float = UAV_SPEED,
        initial_energy: float = UAV_INITIAL_ENERGY,
        hover_power: float = UAV_HOVER_POWER,
        move_power: float = UAV_MOVE_POWER,
        comm_power: float = UAV_COMM_POWER
    ):
        """
        Initialize the UAV.
        
        Args:
            initial_position: Initial position (x, y) in meters
            speed: Speed in m/s
            initial_energy: Initial energy in Joules
            hover_power: Power consumption when hovering in Watts
            move_power: Power consumption when moving in Watts
            comm_power: Power consumption when communicating in Watts
        """
        self.position = initial_position
        self.speed = speed
        self.energy = initial_energy
        self.initial_energy = initial_energy
        self.hover_power = hover_power
        self.move_power = move_power
        self.comm_power = comm_power
        
        # State variables
        self.servicing_state = False
        self.total_distance = 0.0
    
    def get_position(self) -> Tuple[float, float]:
        """
        Get the current position.
        
        Returns:
            Current position (x, y)
        """
        return self.position
    
    def set_position(self, position: Tuple[float, float]) -> None:
        """
        Set the position directly.
        
        Args:
            position: New position (x, y)
        """
        self.position = position
    
    def get_speed(self) -> float:
        """
        Get the speed.
        
        Returns:
            Speed in m/s
        """
        return self.speed
    
    def set_speed(self, speed: float) -> None:
        """
        Set the speed.
        
        Args:
            speed: New speed in m/s
        """
        self.speed = speed
    
    def get_energy(self) -> float:
        """
        Get the current energy.
        
        Returns:
            Current energy in Joules
        """
        return self.energy
    
    def set_energy(self, energy: float) -> None:
        """
        Set the energy directly.
        
        Args:
            energy: New energy in Joules
        """
        self.energy = max(0.0, energy)
    
    def get_servicing_state(self) -> bool:
        """
        Check if the UAV is currently servicing a user.
        
        Returns:
            True if servicing, False otherwise
        """
        return self.servicing_state
    
    def set_servicing(self, servicing_state: bool) -> None:
        """
        Set the servicing state.
        
        Args:
            servicing_state: True if servicing, False otherwise
        """
        self.servicing_state = servicing_state
    
    def get_total_distance(self) -> float:
        """
        Get the total distance traveled.
        
        Returns:
            Total distance in meters
        """
        return self.total_distance
    
    def move_towards(self, target_position: Tuple[float, float], time_step: float) -> Tuple[bool, bool]:
        """
        Move towards a target position.
        
        Args:
            target_position: Target position (x, y)
            time_step: Time step in seconds
        
        Returns:
            A tuple of (moved, reached_target) booleans
        """
        # Calculate direction and distance to target
        dx = target_position[0] - self.position[0]
        dy = target_position[1] - self.position[1]
        distance = self._calculate_distance(self.position, target_position)
        
        # Check if already at target
        if distance < 0.1:
            return False, True
        
        # Calculate maximum distance to move in this time step
        max_distance = self.speed * time_step
        
        # Move towards target
        if distance <= max_distance:
            # Reached target
            new_position = target_position
            self.total_distance += distance
            self.position = new_position
            return True, True
        else:
            # Move towards target
            direction_x = dx / distance
            direction_y = dy / distance
            
            new_position = (
                self.position[0] + direction_x * max_distance,
                self.position[1] + direction_y * max_distance
            )
            
            self.total_distance += max_distance
            self.position = new_position
            return True, False
    
    def _calculate_distance(self, pos1: Tuple[float, float], pos2: Tuple[float, float]) -> float:
        """
        Calculate the Euclidean distance between two positions.
        
        Args:
            pos1: First position (x, y)
            pos2: Second position (x, y)
            
        Returns:
            Euclidean distance
        """
        return math.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)