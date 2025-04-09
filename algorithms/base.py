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
        if self.env is None:
            raise ValueError("Environment not set up. Call setup() first.")
        
        # Reset the environment
        self.env.reset()
        
        # Run the episode
        for _ in range(max_steps):
            state = self.env.get_state()
            
            # Compute action
            target_position, user_id = self.compute_action(state)
            
            # Set service user if needed
            if user_id is not None:
                self.env.set_service_user(user_id)
            
            # Take a step
            self.env.step(target_position)
            
            # Check if done
            if self.env.is_done():
                break
        
        # Get final metrics
        metrics = self.env.get_metrics()
        metrics['algorithm'] = self.name
        metrics['steps'] = self.env.current_step
        metrics['trajectory'] = self.env.get_trajectory()
        metrics['energy_log'] = self.env.get_energy_log()
        metrics['stats_log'] = self.env.get_stats_log()
        
        return metrics