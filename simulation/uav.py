"""
UAV model for path planning simulation.
"""

import math
from typing import Tuple

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
        self._position = initial_position
        self._speed = speed
        self._energy = initial_energy
        self._hover_power = hover_power
        self._move_power = move_power
        self._comm_power = comm_power
        self._is_servicing = False
        
        # Tracking movement
        self._last_position = initial_position
        self._total_distance = 0.0
    
    def get_position(self) -> Tuple[float, float]:
        """
        Get the current position.
        
        Returns:
            Current position (x, y)
        """
        return self._position
    
    def set_position(self, position: Tuple[float, float]) -> None:
        """
        Set the position directly.
        
        Args:
            position: New position (x, y)
        """
        self._last_position = self._position
        self._position = position
        
        # Update total distance
        self._total_distance += self._calculate_distance(self._last_position, self._position)
    
    def get_speed(self) -> float:
        """
        Get the speed.
        
        Returns:
            Speed in m/s
        """
        return self._speed
    
    def set_speed(self, speed: float) -> None:
        """
        Set the speed.
        
        Args:
            speed: New speed in m/s
        """
        self._speed = speed
    
    def get_energy(self) -> float:
        """
        Get the current energy.
        
        Returns:
            Current energy in Joules
        """
        return self._energy
    
    def set_energy(self, energy: float) -> None:
        """
        Set the energy directly.
        
        Args:
            energy: New energy in Joules
        """
        self._energy = energy
    
    def is_servicing(self) -> bool:
        """
        Check if the UAV is currently servicing a user.
        
        Returns:
            True if servicing, False otherwise
        """
        return self._is_servicing
    
    def set_servicing(self, is_servicing: bool) -> None:
        """
        Set the servicing state.
        
        Args:
            is_servicing: True if servicing, False otherwise
        """
        self._is_servicing = is_servicing
    
    def get_total_distance(self) -> float:
        """
        Get the total distance traveled.
        
        Returns:
            Total distance in meters
        """
        return self._total_distance
    
    def move_towards(self, target_position: Tuple[float, float], time_step: float) -> None:
        """
        Move towards a target position.
        
        Args:
            target_position: Target position (x, y)
            time_step: Time step in seconds
        
        Returns:
            A tuple of (moved, reached_target) booleans
        """
        # Store last position
        self._last_position = self._position
        
        # Calculate distance to target
        dx = target_position[0] - self._position[0]
        dy = target_position[1] - self._position[1]
        distance = math.sqrt(dx * dx + dy * dy)
        
        # Calculate maximum distance that can be traveled in the time step
        max_distance = self._speed * time_step
        
        # If we can reach the target, move directly to it
        if distance <= max_distance:
            self._position = target_position
            moved_distance = distance
        else:
            # Otherwise, move as far as possible towards the target
            # Calculate unit vector in direction of target
            direction_x = dx / distance
            direction_y = dy / distance
            
            # Calculate new position
            new_x = self._position[0] + direction_x * max_distance
            new_y = self._position[1] + direction_y * max_distance
            self._position = (new_x, new_y)
            moved_distance = max_distance
        
        # Update total distance
        self._total_distance += moved_distance
        
        # Update energy
        # If moving, use move power; if not, use hover power
        if moved_distance > 0.01:  # Small threshold to determine if we actually moved
            power = self._move_power
        else:
            power = self._hover_power
        
        # Add communication power if servicing
        if self._is_servicing:
            power += self._comm_power
        
        # Calculate energy consumption
        energy_consumed = power * time_step
        self._energy -= energy_consumed
    
    def _calculate_distance(self, pos1: Tuple[float, float], pos2: Tuple[float, float]) -> float:
        """
        Calculate the Euclidean distance between two positions.
        
        Args:
            pos1: First position (x, y)
            pos2: Second position (x, y)
            
        Returns:
            Euclidean distance
        """
        dx = pos2[0] - pos1[0]
        dy = pos2[1] - pos1[1]
        return math.sqrt(dx * dx + dy * dy)