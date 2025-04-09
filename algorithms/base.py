"""
Base algorithm interface for UAV path planning.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Any, Optional

class PathPlanningAlgorithm(ABC):
    """
    Abstract base class for path planning algorithms.
    """
    
    def __init__(self, name: str):
        """
        Initialize the algorithm.
        
        Args:
            name: Name of the algorithm
        """
        self.name = name
        self.env = None
    
    @abstractmethod
    def setup(self, env) -> None:
        """
        Set up the algorithm with the environment.
        
        Args:
            env: Simulation environment
        """
        self.env = env
    
    @abstractmethod
    def compute_action(self, state: Dict[str, Any]) -> Tuple[Optional[Tuple[float, float]], Optional[int]]:
        """
        Compute the next action based on the current state.
        
        Args:
            state: Current state of the environment
            
        Returns:
            Tuple of (target_position, user_id_to_service) where both can be None
        """
        pass
    
    def run_episode(self, max_steps: int = 1000) -> Dict[str, Any]:
        """
        Run a full episode with the algorithm.
        
        Args:
            max_steps: Maximum number of steps
            
        Returns:
            Dictionary with the metrics of the episode
        """
        if not self.env:
            raise ValueError("Environment not set up. Call setup() first.")
        
        # Reset the environment
        self.env.reset()
        
        # Run episode
        step = 0
        while not self.env.is_done() and step < max_steps:
            # Get current state
            state = self.env.get_state()
            
            # Compute action
            target_position, user_id = self.compute_action(state)
            
            # Set service user if specified
            if user_id is not None:
                self.env.set_service_user(user_id)
            
            # Step environment
            self.env.step(target_position)
            
            step += 1
        
        # Get metrics
        metrics = self.env.get_metrics()
        
        # Add algorithm name
        metrics['algorithm'] = self.name
        
        # Add episode data
        metrics['trajectory'] = self.env.get_trajectory()
        metrics['energy_log'] = self.env.get_energy_log()
        metrics['stats_log'] = self.env.get_stats_log()
        
        return metrics