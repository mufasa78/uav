"""
Rapidly-exploring Random Tree (RRT) implementation for UAV path planning.
"""

import random
import math
from typing import Dict, List, Tuple, Any, Optional

# Removed numpy dependency

from algorithms.base import PathPlanningAlgorithm
from utils.config import (
    RRT_MAX_ITERATIONS,
    RRT_STEP_SIZE,
    RRT_GOAL_SAMPLE_RATE,
    RRT_CONNECT_CIRCLE_DISTANCE
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
        self.parent: Optional['RRTNode'] = None
        self.cost = 0.0  # Cost from start to this node

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
        super().__init__("RRT")
        self.max_iterations = max_iterations
        self.step_size = step_size
        self.goal_sample_rate = goal_sample_rate
        self.connect_circle_distance = connect_circle_distance
        
        # Additional attributes
        self.nodes = []
        self.path = []
        self.goal_node = None
        self.nearest_node_to_goal = None
        self.current_path_index = 0
        self.current_goal_index = 0
        self.goals = []
        self.planning_in_progress = False
    
    def setup(self, env) -> None:
        """
        Set up the algorithm with the environment.
        
        Args:
            env: Simulation environment
        """
        super().setup(env)
        
        # Reset the tree
        self.nodes = []
        self.path = []
        self.goal_node = None
        self.nearest_node_to_goal = None
        self.current_path_index = 0
        self.current_goal_index = 0
        self.goals = []
        self.planning_in_progress = False
    
    def compute_action(self, state: Dict[str, Any]) -> Tuple[Optional[Tuple[float, float]], Optional[int]]:
        """
        Compute the next action using RRT.
        
        Args:
            state: Current state of the environment
            
        Returns:
            Tuple of (target_position, user_id_to_service)
        """
        # Get current UAV position
        uav_position = state.get('uav_position', (0, 0))
        
        # Check if we need to plan a new path
        users_with_tasks = []
        for user_id, user in state.get('users', {}).items():
            if user.get('has_task', False):
                users_with_tasks.append((user_id, user.get('position', (0, 0))))
        
        # Sort users by distance to UAV
        users_with_tasks.sort(key=lambda u: math.sqrt((u[1][0] - uav_position[0]) ** 2 + (u[1][1] - uav_position[1]) ** 2))
        
        # If there are users with tasks and we're not currently servicing or planning
        if users_with_tasks and not state.get('current_service_user_id') and not self.planning_in_progress:
            # Get the closest user with a task
            user_id, goal = users_with_tasks[0]
            
            # Plan a path to this user
            self.plan_path(uav_position, goal)
            
            # Set the goals to first reach the user, then service
            self.goals = [(goal, None), (None, user_id)]
            self.current_goal_index = 0
            
            # Mark planning as in progress
            self.planning_in_progress = True
        
        # If we have a path, follow it
        if self.path and self.current_path_index < len(self.path):
            # Get the next waypoint
            next_position = self.path[self.current_path_index]
            
            # Check if we reached the waypoint
            distance = math.sqrt((next_position[0] - uav_position[0]) ** 2 + (next_position[1] - uav_position[1]) ** 2)
            if distance < self.connect_circle_distance:
                # Move to next waypoint
                self.current_path_index += 1
                
                # If we reached the end of the path, reset
                if self.current_path_index >= len(self.path):
                    # Move to next goal
                    self.current_goal_index += 1
                    
                    # If we have more goals, plan a path to the next goal
                    if self.current_goal_index < len(self.goals) and self.goals[self.current_goal_index][0] is not None:
                        self.plan_path(uav_position, self.goals[self.current_goal_index][0])
                        self.current_path_index = 0
                    else:
                        # Reset planning if we have no more goals
                        self.planning_in_progress = False
                        
                        # Return the service action if needed
                        if self.current_goal_index < len(self.goals) and self.goals[self.current_goal_index][1] is not None:
                            return (None, self.goals[self.current_goal_index][1])
                        
                        # Reset goals
                        self.goals = []
                        self.current_goal_index = 0
            
            # Return the next position to follow the path
            if self.current_path_index < len(self.path):
                return (self.path[self.current_path_index], None)
        
        # Default behavior - stay in place
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
        
        # Create goal node
        self.goal_node = RRTNode(goal)
        
        # Initialize nearest node to goal
        self.nearest_node_to_goal = start_node
        nearest_dist = float('inf')
        
        # Build the tree
        for i in range(self.max_iterations):
            # Sample a random point
            if random.random() < self.goal_sample_rate:
                # Sample the goal directly
                random_point = goal
            else:
                # Sample a random point in the world
                world_size = (1000, 1000)
                if self.env:
                    state = self.env.get_state()
                    if state:
                        world_size = state.get('world_size', (1000, 1000))
                random_point = (
                    random.uniform(0, world_size[0]),
                    random.uniform(0, world_size[1])
                )
            
            # Find the nearest node
            nearest_idx = self._get_nearest_node_idx(random_point)
            nearest_node = self.nodes[nearest_idx]
            
            # Steer towards the random point
            new_node = self._steer(nearest_node, random_point)
            
            # Check for collision - simplified, just check distance from obstacles
            has_collision = False
            obstacles = []
            if self.env:
                state = self.env.get_state()
                if state:
                    obstacles = state.get('obstacles', [])
            for obstacle in obstacles:
                obstacle_pos = obstacle.get('position', (0, 0))
                obstacle_radius = obstacle.get('radius', 0)
                
                # Calculate distance to obstacle
                distance = math.sqrt((new_node.position[0] - obstacle_pos[0]) ** 2 + (new_node.position[1] - obstacle_pos[1]) ** 2)
                
                # Check if collision
                if distance < obstacle_radius + 5.0:  # Add some buffer
                    has_collision = True
                    break
            
            # Add the node if no collision
            if not has_collision:
                new_node.parent = nearest_node
                new_node.cost = nearest_node.cost + math.sqrt(
                    (new_node.position[0] - nearest_node.position[0]) ** 2 + 
                    (new_node.position[1] - nearest_node.position[1]) ** 2
                )
                self.nodes.append(new_node)
                
                # Check if this node is closer to the goal
                dist_to_goal = math.sqrt(
                    (new_node.position[0] - goal[0]) ** 2 + 
                    (new_node.position[1] - goal[1]) ** 2
                )
                
                if dist_to_goal < nearest_dist:
                    nearest_dist = dist_to_goal
                    self.nearest_node_to_goal = new_node
                
                # Check if the goal is reached
                if dist_to_goal <= self.connect_circle_distance:
                    # Connect to goal
                    goal_node = RRTNode(goal)
                    goal_node.parent = new_node
                    goal_node.cost = new_node.cost + dist_to_goal
                    self.nodes.append(goal_node)
                    self.nearest_node_to_goal = goal_node
                    break
        
        # Extract the path
        self.path = self._extract_path(self.nearest_node_to_goal)
        self.current_path_index = 0
    
    def _get_nearest_node_idx(self, position: Tuple[float, float]) -> int:
        """
        Find the index of the nearest node to the given position.
        
        Args:
            position: Position to find nearest node to
            
        Returns:
            Index of the nearest node
        """
        dists = [
            math.sqrt((node.position[0] - position[0]) ** 2 + (node.position[1] - position[1]) ** 2)
            for node in self.nodes
        ]
        return dists.index(min(dists))
    
    def _steer(self, from_node: RRTNode, to_position: Tuple[float, float]) -> RRTNode:
        """
        Create a new node by steering from a node towards a position.
        
        Args:
            from_node: Node to steer from
            to_position: Position to steer towards
            
        Returns:
            New node
        """
        # Get direction
        dir_x = to_position[0] - from_node.position[0]
        dir_y = to_position[1] - from_node.position[1]
        
        # Get distance
        dist = math.sqrt(dir_x ** 2 + dir_y ** 2)
        
        # Normalize direction
        if dist > 0:
            dir_x /= dist
            dir_y /= dist
        
        # Get new position
        if dist > self.step_size:
            new_x = from_node.position[0] + dir_x * self.step_size
            new_y = from_node.position[1] + dir_y * self.step_size
        else:
            new_x = to_position[0]
            new_y = to_position[1]
        
        # Create new node
        new_node = RRTNode((new_x, new_y))
        
        return new_node
    
    def _extract_path(self, end_node: RRTNode) -> List[Tuple[float, float]]:
        """
        Extract path from the tree.
        
        Args:
            end_node: End node of the path
            
        Returns:
            List of waypoints
        """
        path = []
        current = end_node
        
        # Traverse from end node to start node
        while current.parent is not None:
            path.append(current.position)
            current = current.parent
        
        # Add start node
        path.append(current.position)
        
        # Reverse to get path from start to end
        path.reverse()
        
        return path