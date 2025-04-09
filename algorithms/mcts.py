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