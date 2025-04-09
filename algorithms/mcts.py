"""
Monte Carlo Tree Search implementation for UAV path planning.
蒙特卡洛树搜索算法在无人机路径规划中的实现。
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
    蒙特卡洛树搜索中的节点。
    
    Each node represents a state in the simulation.
    每个节点代表模拟中的一种状态。
    """
    
    def __init__(
        self, 
        parent=None, 
        action: Optional[Tuple[Tuple[float, float], Optional[int]]] = None,
        state: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize a node.
        初始化节点。
        
        Args:
            parent: Parent node (父节点)
            action: Action that led to this node (target_position, service_user_id)
                   导致此节点的操作（目标位置，要服务的用户ID）
            state: State of the environment (环境状态)
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
    蒙特卡洛树搜索算法用于无人机路径规划。
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
        初始化MCTS算法。
        
        Args:
            iterations: Number of iterations for each action computation
                       每次计算动作的迭代次数
            exploration_weight: Weight for exploration term in UCT
                              UCT中探索项的权重
            rollout_depth: Maximum depth for rollout simulation
                         展开模拟的最大深度
            max_depth: Maximum depth of the tree
                     树的最大深度
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
        使用MCTS计算下一步动作。
        
        Args:
            state: Current state of the environment
                 环境的当前状态
            
        Returns:
            Tuple of (target_position, user_id_to_service)
            返回一个元组：(目标位置, 要服务的用户ID)
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
        选择一个节点来进行展开模拟。
        
        Args:
            node: Starting node
                 起始节点
            
        Returns:
            Selected node
            选择的节点
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
        通过添加子节点来扩展节点。
        
        Args:
            node: Node to expand
                 要扩展的节点
            
        Returns:
            New child node
            新的子节点
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
        从节点开始执行展开模拟。
        
        Args:
            node: Starting node for rollout
                 展开模拟的起始节点
            
        Returns:
            Reward from the rollout
            展开模拟的奖励值
        """
        # Create a copy of the environment
        env_copy = self._create_env_copy(node.state)
        
        # Track rewards at each step
        accumulated_reward = 0.0
        discount_factor = 0.95  # Discount factor for future rewards
        
        # Improved rollout - take actions according to the rollout policy for rollout_depth steps
        for step in range(self.rollout_depth):
            if env_copy.is_done():
                break
            
            # Get current state
            current_state = env_copy.get_state()
            
            # Get possible actions
            possible_actions = self._get_possible_actions(current_state)
            
            if not possible_actions:
                break
            
            # Choose action according to improved rollout policy (heuristic-based)
            action = self._improved_rollout_policy(current_state, possible_actions)
            
            # Apply the action
            target_position, user_id = action
            if user_id is not None:
                env_copy.set_service_user(user_id)
            env_copy.step(target_position)
            
            # Get immediate reward
            immediate_reward = self._calculate_immediate_reward(current_state, env_copy.get_state())
            
            # Accumulate discounted reward
            accumulated_reward += (discount_factor ** step) * immediate_reward
        
        # Add final state evaluation
        final_metrics = env_copy.get_metrics()
        final_reward = (
            final_metrics.get('serviced_tasks', 0) * 15.0 +      # Higher reward for serviced tasks
            final_metrics.get('data_processed', 0) * 0.5 -       # Reward for processed data
            final_metrics.get('energy_consumed', 0) / 800.0      # Adjusted penalty for energy consumption
        )
        
        # Add reward for remaining energy proportional to the mission completion
        if final_metrics.get('serviced_tasks', 0) > 0:
            energy_efficiency = final_metrics.get('remaining_energy', 0) / self.env.uav.get_energy() if self.env else 0
            final_reward += energy_efficiency * 5.0
        
        return accumulated_reward + final_reward
    
    def _improved_rollout_policy(self, state: Dict[str, Any], possible_actions: List[Tuple[Tuple[float, float], Optional[int]]]) -> Tuple[Tuple[float, float], Optional[int]]:
        """
        An improved policy for selecting actions during rollout.
        
        Args:
            state: Current state
            possible_actions: List of possible actions
            
        Returns:
            Selected action
        """
        # If there's a user to service in range, prioritize that
        uav_position = state.get('uav_position', (0, 0))
        
        # Filter service actions (actions where user_id is not None)
        service_actions = [action for action in possible_actions if action[1] is not None]
        
        # If there are users to service, choose the closest one with the most data
        if service_actions:
            # Sort by distance and task data size (weighted combination)
            def service_score(action):
                user_id = action[1]
                user = state.get('users', {}).get(user_id, {})
                user_position = user.get('position', (0, 0))
                distance = math.sqrt((uav_position[0] - user_position[0])**2 + (uav_position[1] - user_position[1])**2)
                task_data = user.get('task_data', 0)
                # Higher score for closer users with more data
                return task_data / (distance + 1)
            
            # With 70% probability, select the best service action
            if random.random() < 0.7:
                return max(service_actions, key=service_score)
        
        # For movement actions, prefer moves towards users with tasks
        movement_actions = [action for action in possible_actions if action[0] is not None]
        
        if movement_actions:
            # Find users with tasks
            users_with_tasks = []
            for user_id, user in state.get('users', {}).items():
                if user.get('has_task', False):
                    users_with_tasks.append((user_id, user.get('position', (0, 0))))
            
            if users_with_tasks:
                # Sort by distance to UAV
                users_with_tasks.sort(key=lambda u: math.sqrt((u[1][0] - uav_position[0])**2 + (u[1][1] - uav_position[1])**2))
                
                # Target the closest user with a task
                target_user_position = users_with_tasks[0][1]
                
                # Find the action that moves closest to the target user
                def movement_score(action):
                    target_position = action[0]
                    distance_to_target = math.sqrt(
                        (target_position[0] - target_user_position[0])**2 + 
                        (target_position[1] - target_user_position[1])**2
                    )
                    return -distance_to_target  # Negative because we want to minimize distance
                
                # With 70% probability, select the best movement action
                if random.random() < 0.7:
                    return max(movement_actions, key=movement_score)
        
        # Fall back to random selection for exploration
        return random.choice(possible_actions)
    
    def _calculate_immediate_reward(self, old_state: Dict[str, Any], new_state: Dict[str, Any]) -> float:
        """
        Calculate immediate reward for transitioning from old state to new state.
        
        Args:
            old_state: State before action
            new_state: State after action
            
        Returns:
            Immediate reward value
        """
        reward = 0.0
        
        # Reward for servicing users
        old_serviced = old_state.get('serviced_tasks', 0)
        new_serviced = new_state.get('serviced_tasks', 0)
        if new_serviced > old_serviced:
            reward += 10.0  # Significant reward for completing a service
        
        # Reward for making progress on a service
        if old_state.get('current_service_user_id') is not None:
            old_remaining = old_state.get('current_service_task_data_remaining', 0)
            new_remaining = new_state.get('current_service_task_data_remaining', 0)
            if old_remaining > new_remaining:
                reward += (old_remaining - new_remaining) * 0.2  # Small reward for progress
        
        # Penalty for energy consumption
        old_energy = old_state.get('uav_energy', 0)
        new_energy = new_state.get('uav_energy', 0)
        energy_used = old_energy - new_energy
        reward -= energy_used / 1000.0  # Small penalty proportional to energy used
        
        # Reward for getting closer to users with tasks
        uav_old_pos = old_state.get('uav_position', (0, 0))
        uav_new_pos = new_state.get('uav_position', (0, 0))
        
        # Only calculate distance improvement if UAV moved
        if uav_old_pos != uav_new_pos:
            distance_improvement = 0.0
            user_count = 0
            
            for user_id, user in new_state.get('users', {}).items():
                if user.get('has_task', False):
                    user_pos = user.get('position', (0, 0))
                    
                    old_distance = math.sqrt((uav_old_pos[0] - user_pos[0])**2 + (uav_old_pos[1] - user_pos[1])**2)
                    new_distance = math.sqrt((uav_new_pos[0] - user_pos[0])**2 + (uav_new_pos[1] - user_pos[1])**2)
                    
                    # Reward for getting closer, penalty for moving away
                    distance_improvement += (old_distance - new_distance)
                    user_count += 1
            
            if user_count > 0:
                reward += (distance_improvement / user_count) * 0.05
        
        return reward
    
    def _backpropagate(self, node: Node, reward: float) -> None:
        """
        Backpropagate the reward up the tree.
        将奖励值反向传播到树的上层节点。
        
        Args:
            node: Leaf node
                 叶子节点
            reward: Reward from rollout
                   展开模拟的奖励值
        """
        # Update all nodes from the leaf to the root
        current = node
        while current:
            current.update(reward)
            current = current.parent
    
    def _get_possible_actions(self, state: Dict[str, Any]) -> List[Tuple[Tuple[float, float], Optional[int]]]:
        """
        Get possible actions from the current state.
        从当前状态获取可能的动作。
        
        Args:
            state: Current state
                  当前状态
            
        Returns:
            List of possible actions (target_position, service_user_id)
            可能动作的列表（目标位置，要服务的用户ID）
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
        使用给定状态创建环境的副本。
        
        This is a simplified version that creates a new environment
        and manually copies the relevant state information.
        这是一个简化版本，创建一个新的环境并手动复制相关的状态信息。
        
        Args:
            state: State to copy
                  要复制的状态
            
        Returns:
            Copy of the environment
            环境的副本
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