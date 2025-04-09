"""
Monte Carlo Tree Search implementation for UAV path planning.
"""

import math
import random
import numpy as np
from typing import Dict, List, Tuple, Any, Optional

from algorithms.base import PathPlanningAlgorithm
from utils.config import (
    MCTS_ITERATIONS, 
    MCTS_EXPLORATION_WEIGHT, 
    MCTS_ROLLOUT_DEPTH,
    MCTS_MAX_DEPTH,
    COMM_RANGE
)

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
        if not self.children:
            raise ValueError("Node has no children")
        
        def uct(node):
            """Upper Confidence Bound for Trees."""
            exploitation = node.value
            exploration = exploration_weight * math.sqrt(math.log(self.visits) / node.visits)
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
        super().__init__(name="Monte Carlo Tree Search")
        self.iterations = iterations
        self.exploration_weight = exploration_weight
        self.rollout_depth = rollout_depth
        self.max_depth = max_depth
    
    def setup(self, env) -> None:
        """
        Set up the algorithm with the environment.
        
        Args:
            env: Simulation environment
        """
        super().setup(env)
    
    def compute_action(self, state: Dict[str, Any]) -> Tuple[Optional[Tuple[float, float]], Optional[int]]:
        """
        Compute the next action using MCTS.
        
        Args:
            state: Current state of the environment
            
        Returns:
            Tuple of (target_position, user_id_to_service)
        """
        # Create the root node
        root = Node(state=state)
        
        # Generate possible actions for the root
        root.untried_actions = self._get_possible_actions(state)
        
        # Run the MCTS algorithm
        for _ in range(self.iterations):
            # Selection and expansion
            node = self._tree_policy(root)
            
            # Simulation
            reward = self._rollout(node)
            
            # Backpropagation
            self._backpropagate(node, reward)
        
        # Choose the best action
        best_child = max(root.children, key=lambda c: c.visits)
        return best_child.action
    
    def _tree_policy(self, node: Node) -> Node:
        """
        Select a node to run the rollout from.
        
        Args:
            node: Starting node
            
        Returns:
            Selected node
        """
        current_depth = 0
        
        while current_depth < self.max_depth:
            # If node is not fully expanded, expand it
            if not node.is_fully_expanded() and node.untried_actions:
                return self._expand(node)
            
            # If node is a terminal node, return it
            if not node.children:
                return node
            
            # Otherwise, select the best child
            node = node.best_child(self.exploration_weight)
            current_depth += 1
        
        return node
    
    def _expand(self, node: Node) -> Node:
        """
        Expand a node by adding a child.
        
        Args:
            node: Node to expand
            
        Returns:
            New child node
        """
        # Get an untried action
        action = node.untried_actions.pop()
        
        # Create a copy of the environment
        env_copy = self._create_env_copy(node.state)
        
        # Apply the action
        if action[1] is not None:  # If there's a user to service
            env_copy.set_service_user(action[1])
        
        # Take a step
        env_copy.step(action[0])
        
        # Get the new state
        new_state = env_copy.get_state()
        
        # Create a new child node
        child = node.add_child(action, new_state)
        
        # Generate possible actions for the child
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
        
        # Simulate until reaching the rollout depth
        for _ in range(self.rollout_depth):
            # Check if the simulation is done
            if env_copy.is_done():
                break
            
            # Get possible actions
            possible_actions = self._get_possible_actions(env_copy.get_state())
            if not possible_actions:
                break
            
            # Choose an action according to the rollout policy
            action = random.choice(possible_actions)
            
            # Apply the action
            if action[1] is not None:  # If there's a user to service
                env_copy.set_service_user(action[1])
            
            # Take a step
            env_copy.step(action[0])
        
        # Get the final state metrics and compute the reward
        metrics = env_copy.get_metrics()
        reward = (
            metrics.get('serviced_tasks', 0) * 100 +  # Prioritize servicing tasks
            metrics.get('data_processed', 0) / 1e6 -  # Reward for processing data
            metrics.get('energy_consumed', 0) / 1000  # Penalize for energy consumption
        )
        
        return reward
    
    def _backpropagate(self, node: Node, reward: float) -> None:
        """
        Backpropagate the reward up the tree.
        
        Args:
            node: Leaf node
            reward: Reward from rollout
        """
        while node is not None:
            node.update(reward)
            node = node.parent
    
    def _get_possible_actions(self, state: Dict[str, Any]) -> List[Tuple[Tuple[float, float], Optional[int]]]:
        """
        Get possible actions from the current state.
        
        Args:
            state: Current state
            
        Returns:
            List of possible actions (target_position, service_user_id)
        """
        actions = []
        
        # Get the current UAV position
        uav_position = state.get('uav_position', (0, 0))
        
        # Get users with tasks
        users_with_tasks = state.get('users_with_tasks', [])
        all_users = state.get('all_users', [])
        
        # Generate movements to users with tasks
        for user_id in users_with_tasks:
            if user_id < len(all_users):
                user_position = all_users[user_id][1]
                
                # Check if the user is within comm range
                distance = math.sqrt(
                    (user_position[0] - uav_position[0]) ** 2 + 
                    (user_position[1] - uav_position[1]) ** 2
                )
                
                if distance <= COMM_RANGE:
                    # User is in range, service them
                    actions.append((user_position, user_id))
                else:
                    # Move towards the user
                    actions.append((user_position, None))
        
        # Add some random exploration points if there are few users
        if len(actions) < 5:
            world_size = (1000, 1000)  # Default world size
            for _ in range(5 - len(actions)):
                random_x = random.uniform(0, world_size[0])
                random_y = random.uniform(0, world_size[1])
                actions.append(((random_x, random_y), None))
        
        return actions
    
    def _create_env_copy(self, state: Dict[str, Any]):
        """
        Create a copy of the environment with the given state.
        
        This is a simplified version that creates a new environment
        and manually copies the relevant state information.
        
        Args:
            state: State to copy
            
        Returns:
            Copy of the environment
        """
        # Import here to avoid circular imports
        from simulation.environment import Environment
        
        # Create a new environment
        env_copy = Environment()
        
        # Reset the environment
        env_copy.reset()
        
        # Copy the state
        env_copy.current_time = state.get('time', 0)
        env_copy.uav.set_position(state.get('uav_position', (0, 0)))
        env_copy.uav.set_energy(state.get('uav_energy', 10000))
        env_copy.serviced_tasks = state.get('serviced_tasks', 0)
        env_copy.data_processed = state.get('data_processed', 0)
        env_copy.total_flight_distance = state.get('total_flight_distance', 0)
        
        # Copy users and tasks
        env_copy.users = state.get('all_users', [])
        env_copy.users_with_tasks = state.get('users_with_tasks', [])
        
        return env_copy