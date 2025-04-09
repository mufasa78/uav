"""
Rapidly-exploring Random Tree (RRT) implementation for UAV path planning.
"""

import math
import random
import numpy as np
from typing import Dict, List, Tuple, Any, Optional

from algorithms.base import PathPlanningAlgorithm
from utils.config import (
    RRT_MAX_ITERATIONS,
    RRT_STEP_SIZE,
    RRT_GOAL_SAMPLE_RATE,
    RRT_CONNECT_CIRCLE_DISTANCE,
    COMM_RANGE
)

class RRTNode:
    """
    Node in the RRT.
    """
    
    def __init__(self, position: Tuple[float, float]):
        """
        Initialize a node.
        
        Args:
            position: Position of the node (x, y)
        """
        self.position = position
        self.parent = None
        self.cost = 0.0  # Cost from the start node

class RRTAlgorithm(PathPlanningAlgorithm):
    """
    Rapidly-exploring Random Tree algorithm for UAV path planning.
    """
    
    def __init__(
        self,
        max_iterations: int = RRT_MAX_ITERATIONS,
        step_size: float = RRT_STEP_SIZE,
        goal_sample_rate: float = RRT_GOAL_SAMPLE_RATE,
        connect_circle_distance: float = RRT_CONNECT_CIRCLE_DISTANCE
    ):
        """
        Initialize the RRT algorithm.
        
        Args:
            max_iterations: Maximum number of iterations for tree building
            step_size: Step size for extending the tree
            goal_sample_rate: Probability of sampling the goal position
            connect_circle_distance: Maximum distance to connect two nodes
        """
        super().__init__(name="Rapidly-exploring Random Tree")
        self.max_iterations = max_iterations
        self.step_size = step_size
        self.goal_sample_rate = goal_sample_rate
        self.connect_circle_distance = connect_circle_distance
        
        # Tree structure
        self.nodes = []
        self.path = []
        self.target_position = None
        self.current_user_id = None
    
    def setup(self, env) -> None:
        """
        Set up the algorithm with the environment.
        
        Args:
            env: Simulation environment
        """
        super().setup(env)
        self.world_size = (1000, 1000)  # Default world size
    
    def compute_action(self, state: Dict[str, Any]) -> Tuple[Optional[Tuple[float, float]], Optional[int]]:
        """
        Compute the next action using RRT.
        
        Args:
            state: Current state of the environment
            
        Returns:
            Tuple of (target_position, user_id_to_service)
        """
        # Get the current UAV position
        uav_position = state.get('uav_position', (0, 0))
        
        # Get users with tasks
        users_with_tasks = state.get('users_with_tasks', [])
        all_users = state.get('all_users', [])
        
        # Check if there's a user in communication range to service
        for user_id in users_with_tasks:
            if user_id < len(all_users):
                user_position = all_users[user_id][1]
                
                # Calculate distance to user
                distance = math.sqrt(
                    (user_position[0] - uav_position[0]) ** 2 + 
                    (user_position[1] - uav_position[1]) ** 2
                )
                
                if distance <= COMM_RANGE:
                    # If a user is in range, service them
                    return (user_position, user_id)
        
        # If we have a path to follow, continue following it
        if self.path and len(self.path) > 1:
            next_position = self.path.pop(0)
            return (next_position, None)
        
        # Otherwise, plan a new path to a user
        closest_user_id = None
        closest_distance = float('inf')
        closest_position = None
        
        for user_id in users_with_tasks:
            if user_id < len(all_users):
                user_position = all_users[user_id][1]
                
                # Calculate distance to user
                distance = math.sqrt(
                    (user_position[0] - uav_position[0]) ** 2 + 
                    (user_position[1] - uav_position[1]) ** 2
                )
                
                if distance < closest_distance:
                    closest_distance = distance
                    closest_user_id = user_id
                    closest_position = user_position
        
        # If found a user, plan a path to them
        if closest_position:
            self.plan_path(uav_position, closest_position)
            
            # If a path is found, return the first waypoint
            if self.path and len(self.path) > 0:
                next_position = self.path.pop(0)
                return (next_position, None)
        
        # If no path is found or no users, stay in place or explore randomly
        if not users_with_tasks:
            # Random exploration
            random_x = random.uniform(0, self.world_size[0])
            random_y = random.uniform(0, self.world_size[1])
            return ((random_x, random_y), None)
        
        # Default: no movement, no service
        return (None, None)
    
    def plan_path(self, start: Tuple[float, float], goal: Tuple[float, float]) -> None:
        """
        Plan a path using RRT.
        
        Args:
            start: Start position
            goal: Goal position
        """
        # Reset the tree
        self.nodes = []
        self.path = []
        
        # Create start node
        start_node = RRTNode(start)
        self.nodes.append(start_node)
        
        # RRT algorithm
        for _ in range(self.max_iterations):
            # Sample a random position (with bias towards the goal)
            if random.random() < self.goal_sample_rate:
                random_position = goal
            else:
                random_x = random.uniform(0, self.world_size[0])
                random_y = random.uniform(0, self.world_size[1])
                random_position = (random_x, random_y)
            
            # Find the nearest node
            nearest_node_idx = self._get_nearest_node_idx(np.array(random_position))
            nearest_node = self.nodes[nearest_node_idx]
            
            # Steer towards the random position
            new_node = self._steer(nearest_node, np.array(random_position))
            
            # Add the new node to the tree
            new_node.parent = nearest_node_idx
            new_node.cost = nearest_node.cost + self.step_size  # Simple cost function
            self.nodes.append(new_node)
            
            # Check if we're close enough to the goal
            dist_to_goal = math.sqrt(
                (new_node.position[0] - goal[0]) ** 2 + 
                (new_node.position[1] - goal[1]) ** 2
            )
            
            if dist_to_goal <= self.connect_circle_distance:
                # Create a goal node
                goal_node = RRTNode(goal)
                goal_node.parent = len(self.nodes) - 1  # Parent is the new node
                goal_node.cost = new_node.cost + dist_to_goal
                self.nodes.append(goal_node)
                
                # Extract the path
                self.path = self._extract_path(goal_node)
                break
    
    def _get_nearest_node_idx(self, position: np.ndarray) -> int:
        """
        Find the index of the nearest node to the given position.
        
        Args:
            position: Position to find nearest node to
            
        Returns:
            Index of the nearest node
        """
        distances = []
        for node in self.nodes:
            node_pos = np.array(node.position)
            distance = np.linalg.norm(node_pos - position)
            distances.append(distance)
        
        return np.argmin(distances)
    
    def _steer(self, from_node: RRTNode, to_position: np.ndarray) -> RRTNode:
        """
        Create a new node by steering from a node towards a position.
        
        Args:
            from_node: Node to steer from
            to_position: Position to steer towards
            
        Returns:
            New node
        """
        from_position = np.array(from_node.position)
        direction = to_position - from_position
        dist = np.linalg.norm(direction)
        
        # If distance is less than step size, go directly to the position
        if dist <= self.step_size:
            new_position = to_position
        else:
            # Otherwise, move in the direction with step size
            direction = direction / dist * self.step_size
            new_position = from_position + direction
        
        # Ensure position is within bounds
        new_position[0] = max(0, min(self.world_size[0], new_position[0]))
        new_position[1] = max(0, min(self.world_size[1], new_position[1]))
        
        # Create a new node
        new_node = RRTNode((new_position[0], new_position[1]))
        
        return new_node
    
    def _extract_path(self, end_node: RRTNode) -> List[Tuple[float, float]]:
        """
        Extract path from the tree.
        
        Args:
            end_node: End node of the path
            
        Returns:
            List of waypoints
        """
        path = [end_node.position]
        current_node = end_node
        
        # Traverse the tree backwards from the end node to the start node
        while current_node.parent is not None:
            current_node = self.nodes[current_node.parent]
            path.append(current_node.position)
        
        # Reverse the path to get it from start to end
        path.reverse()
        
        return path