"""
Monte Carlo Tree Search implementation for UAV path planning.
"""

import random
import math
import logging
from typing import Dict, List, Tuple, Any, Optional

import numpy as np

from algorithms.base import PathPlanningAlgorithm
from utils.config import (
    MCTS_ITERATIONS,
    MCTS_EXPLORATION_WEIGHT,
    MCTS_ROLLOUT_DEPTH,
    MCTS_MAX_DEPTH
)

# Configure logging
logger = logging.getLogger(__name__)

class Node:
    """
    Node in the Monte Carlo Tree Search.
    
    Each node represents a state in the simulation.
    """
    
    def __init__(
        self, 
        parent=None, 
        action: Optional[Tuple[Tuple[float, float], Optional[int]]] = None,
        state: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize a node.
        
        Args:
            parent: Parent node
            action: Action that led to this node (target_position, service_user_id)
            state: State of the environment
        """
        self.parent = parent
        self.action = action
        self.state = state
        self.children = []
        self.visits = 0
        self.value = 0.0
        self.untried_actions = []
    
    def add_child(self, action: Tuple[Tuple[float, float], Optional[int]], state: Dict[str, Any]) -> 'Node':
        """
        Add a child node.
        
        Args:
            action: Action that leads to the child
            state: State of the child
            
        Returns:
            The new child node
        """
        child = Node(parent=self, action=action, state=state)
        self.children.append(child)
        return child
    
    def update(self, result: float) -> None:
        """
        Update the node's value and visit count.
        
        Args:
            result: Result of the simulation
        """
        self.visits += 1
        self.value += (result - self.value) / self.visits
    
    def is_fully_expanded(self) -> bool:
        """Check if all possible actions have been tried."""
        return len(self.untried_actions) == 0
    
    def best_child(self, exploration_weight: float) -> 'Node':
        """
        Select the best child node using UCT.
        
        Args:
            exploration_weight: Weight for exploration term
            
        Returns:
            Best child node
        """
        # If no children, return self
        if not self.children:
            return self
        
        # Find best child using UCT
        def uct(node):
            """Upper Confidence Bound for Trees."""
            if node.visits == 0:
                return float("inf")
            
            # UCT formula: value + exploration_weight * sqrt(log(parent_visits) / visits)
            exploitation = node.value
            exploration = exploration_weight * math.sqrt(2 * math.log(self.visits) / node.visits)
            return exploitation + exploration
        
        return max(self.children, key=uct)
    
    def rollout_policy(self, possible_actions: List[Tuple[Tuple[float, float], Optional[int]]]) -> Tuple[Tuple[float, float], Optional[int]]:
        """
        Select an action according to the rollout policy.
        
        Args:
            possible_actions: List of possible actions
            
        Returns:
            Selected action
        """
        # Simple random policy
        return random.choice(possible_actions)

class MCTSAlgorithm(PathPlanningAlgorithm):
    """
    Monte Carlo Tree Search algorithm for UAV path planning.
    """
    
    def __init__(
        self,
        iterations: int = MCTS_ITERATIONS,
        exploration_weight: float = MCTS_EXPLORATION_WEIGHT,
        rollout_depth: int = MCTS_ROLLOUT_DEPTH,
        max_depth: int = MCTS_MAX_DEPTH
    ):
        """
        Initialize the MCTS algorithm.
        
        Args:
            iterations: Number of iterations for each action computation
            exploration_weight: Weight for exploration term in UCT
            rollout_depth: Maximum depth for rollout simulation
            max_depth: Maximum depth of the tree
        """
        super().__init__("MCTS")
        self.iterations = iterations
        self.exploration_weight = exploration_weight
        self.rollout_depth = rollout_depth
        self.max_depth = max_depth
        
        # Additional attributes
        self.root = None
    
    def setup(self, env) -> None:
        """
        Set up the algorithm with the environment.
        
        Args:
            env: Simulation environment
        """
        super().setup(env)
        
        # Reset the root node
        self.root = None
    
    def compute_action(self, state: Dict[str, Any]) -> Tuple[Optional[Tuple[float, float]], Optional[int]]:
        """
        Compute the next action using MCTS.
        
        Args:
            state: Current state of the environment
            
        Returns:
            Tuple of (target_position, user_id_to_service)
        """
        # Create new root node with current state
        self.root = Node(state=state)
        
        # Get possible actions
        self.root.untried_actions = self._get_possible_actions(state)
        
        # Simplified version for demo
        if not self.root.untried_actions:
            return None, None
        
        # Run iterations
        for _ in range(min(self.iterations, 10)):  # Limit to 10 iterations for demo
            # Selection and expansion
            node = self._tree_policy(self.root)
            
            # Rollout
            reward = self._rollout(node)
            
            # Backpropagation
            self._backpropagate(node, reward)
        
        # Select best child
        best_child = self.root.best_child(exploration_weight=0.0)  # No exploration for final selection
        
        # Return the action that leads to the best child
        return best_child.action if best_child.action else (None, None)
    
    def _tree_policy(self, node: Node) -> Node:
        """
        Select a node to run the rollout from.
        
        Args:
            node: Starting node
            
        Returns:
            Selected node
        """
        # Check if maximum depth reached
        depth = 0
        current = node
        while current.parent:
            current = current.parent
            depth += 1
        
        if depth >= self.max_depth:
            return node
        
        # Loop until we find a node to expand
        current = node
        while self.env and not self.env.is_done():
            if not current.is_fully_expanded():
                return self._expand(current)
            else:
                current = current.best_child(self.exploration_weight)
        
        return current
    
    def _expand(self, node: Node) -> Node:
        """
        Expand a node by adding a child.
        
        Args:
            node: Node to expand
            
        Returns:
            New child node
        """
        # Check if there are untried actions
        if not node.untried_actions:
            return node
        
        # Choose a random untried action
        action = random.choice(node.untried_actions)
        node.untried_actions.remove(action)
        
        # Create a copy of the environment
        env_copy = self._create_env_copy(node.state)
        
        # Apply the action
        target_position, user_id = action
        if user_id is not None:
            env_copy.set_service_user(user_id)
        env_copy.step(target_position)
        
        # Get new state
        new_state = env_copy.get_state()
        
        # Create new child node
        child = node.add_child(action, new_state)
        
        # Set untried actions for the child
        child.untried_actions = self._get_possible_actions(new_state)
        
        return child
    
    def _rollout(self, node: Node) -> float:
        """
        Perform a rollout from the node.
        
        Args:
            node: Starting node for rollout
            
        Returns:
            Reward from the rollout
        """
        # Create a copy of the environment
        env_copy = self._create_env_copy(node.state)
        
        # Simplified rollout - just take random actions for rollout_depth steps
        for _ in range(self.rollout_depth):
            if env_copy.is_done():
                break
            
            # Get current state
            current_state = env_copy.get_state()
            
            # Get possible actions
            possible_actions = self._get_possible_actions(current_state)
            
            if not possible_actions:
                break
            
            # Choose action according to rollout policy
            action = node.rollout_policy(possible_actions)
            
            # Apply the action
            target_position, user_id = action
            if user_id is not None:
                env_copy.set_service_user(user_id)
            env_copy.step(target_position)
        
        # Calculate reward from final state
        metrics = env_copy.get_metrics()
        reward = metrics.get('serviced_tasks', 0) * 10.0  # Reward for serviced tasks
        reward -= metrics.get('energy_consumed', 0) / 1000.0  # Penalty for energy consumption
        
        return reward
    
    def _backpropagate(self, node: Node, reward: float) -> None:
        """
        Backpropagate the reward up the tree.
        
        Args:
            node: Leaf node
            reward: Reward from rollout
        """
        # Update all nodes from the leaf to the root
        current = node
        while current:
            current.update(reward)
            current = current.parent
    
    def _get_possible_actions(self, state: Dict[str, Any]) -> List[Tuple[Tuple[float, float], Optional[int]]]:
        """
        Get possible actions from the current state.
        
        Args:
            state: Current state
            
        Returns:
            List of possible actions (target_position, service_user_id)
        """
        possible_actions = []
        
        # Get current UAV position
        uav_position = state.get('uav_position', (0, 0))
        
        # Add movement actions in 8 directions
        directions = [
            (1, 0), (1, 1), (0, 1), (-1, 1),
            (-1, 0), (-1, -1), (0, -1), (1, -1)
        ]
        
        world_size = state.get('world_size', (1000, 1000))
        step_size = 50.0  # Step size in meters
        
        for dx, dy in directions:
            # Calculate new position
            new_x = max(0, min(world_size[0], uav_position[0] + dx * step_size))
            new_y = max(0, min(world_size[1], uav_position[1] + dy * step_size))
            
            # Add movement action
            possible_actions.append(((new_x, new_y), None))
        
        # Add service actions for users with tasks
        for user_id, user in state.get('users', {}).items():
            if user.get('has_task', False):
                # Add service action for this user
                possible_actions.append((None, user_id))
        
        return possible_actions
    
    def _create_env_copy(self, state: Optional[Dict[str, Any]]):
        """
        Create a copy of the environment with the given state.
        
        This is a simplified version that creates a new environment
        and manually copies the relevant state information.
        
        Args:
            state: State to copy
            
        Returns:
            Copy of the environment
        """
        # Import here to avoid circular import
        from simulation.environment import Environment
        
        # Create new environment
        env_copy = Environment()
        env_copy.reset()
        
        if state is None:
            return env_copy
            
        # Set UAV position and energy
        env_copy.uav.set_position(state.get('uav_position', (0, 0)))
        env_copy.uav.set_energy(state.get('uav_energy', 10000.0))
        
        # Current step
        env_copy.current_step = state.get('current_step', 0)
        
        # Copy users
        env_copy.users = state.get('users', {}).copy()
        
        # Copy service state
        env_copy.current_service_user_id = state.get('current_service_user_id')
        
        return env_copy