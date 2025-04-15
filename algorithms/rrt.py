"""
Rapidly-exploring Random Tree (RRT) implementation for UAV path planning.
快速探索随机树算法在无人机路径规划中的实现。
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
    RRT中的节点。
    """

    def __init__(self, position: Tuple[float, float]):
        """
        Initialize a node.
        初始化节点。

        Args:
            position: Position of the node (x, y)
                     节点的位置坐标 (x, y)
        """
        self.position = position
        self.parent: Optional['RRTNode'] = None
        self.cost = 0.0  # Cost from start to this node

class RRTAlgorithm(PathPlanningAlgorithm):
    """
    Rapidly-exploring Random Tree algorithm for UAV path planning.
    快速探索随机树算法用于无人机路径规划。
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
        初始化RRT算法。

        Args:
            max_iterations: Maximum number of iterations for tree building
                           树构建的最大迭代次数
            step_size: Step size for extending the tree
                     扩展树的步长
            goal_sample_rate: Probability of sampling the goal position
                             采样目标位置的概率
            connect_circle_distance: Maximum distance to connect two nodes
                                    连接两个节点的最大距离
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
        使用RRT计算下一步动作。

        Args:
            state: Current state of the environment
                  环境的当前状态

        Returns:
            Tuple of (target_position, user_id_to_service)
            返回一个元组：(目标位置, 要服务的用户ID)
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

    def plan_path(self, start: Tuple[float, float], goal: Tuple[float, float]) -> List[Tuple[float, float]]:
        """
        Plan a path using RRT* (an improved version of RRT).
        使用RRT*（RRT的改进版本）规划路径。

        Args:
            start: Start position
                  起始位置
            goal: Goal position
                 目标位置

        Returns:
            List of waypoints
            航点列表
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

        # Get world size
        world_size = (1000, 1000)
        if self.env:
            state = self.env.get_state()
            if state:
                world_size = state.get('world_size', (1000, 1000))

        # No obstacles in the new implementation
        obstacles = []

        # Calculate neighborhood radius for rewiring
        # The neighborhood radius is proportional to the size of the space and inversely proportional to the cube root of the number of nodes
        neighborhood_radius = min(world_size) * math.sqrt(math.log(self.max_iterations + 1) / self.max_iterations)

        # Build the tree
        for i in range(self.max_iterations):
            # Sample a random point with increasing bias towards the goal
            goal_sample_rate = self.goal_sample_rate
            if i > self.max_iterations * 0.7:  # Increase goal sampling rate in the later iterations
                goal_sample_rate = min(0.5, goal_sample_rate + 0.1)

            if random.random() < goal_sample_rate:
                # Sample the goal directly
                random_point = goal
            else:
                # Sample a random point in the world
                # Bias sampling towards areas of interest (around the start, goal, and midpoint)
                bias = random.random()
                if bias < 0.1:  # Sample near start
                    random_point = (
                        start[0] + random.uniform(-100, 100),
                        start[1] + random.uniform(-100, 100)
                    )
                elif bias < 0.2:  # Sample near goal
                    random_point = (
                        goal[0] + random.uniform(-100, 100),
                        goal[1] + random.uniform(-100, 100)
                    )
                elif bias < 0.3:  # Sample near midpoint
                    midpoint = ((start[0] + goal[0]) / 2, (start[1] + goal[1]) / 2)
                    random_point = (
                        midpoint[0] + random.uniform(-100, 100),
                        midpoint[1] + random.uniform(-100, 100)
                    )
                else:  # Sample randomly in the world
                    random_point = (
                        random.uniform(0, world_size[0]),
                        random.uniform(0, world_size[1])
                    )

                # Clamp to world boundaries
                random_point = (
                    max(0, min(world_size[0], random_point[0])),
                    max(0, min(world_size[1], random_point[1]))
                )

            # Find the nearest node
            nearest_idx = self._get_nearest_node_idx(random_point)
            nearest_node = self.nodes[nearest_idx]

            # Steer towards the random point
            new_node = self._steer(nearest_node, random_point)

            # Check for collision - check path from nearest to new node
            if self._check_collision_free(nearest_node.position, new_node.position, obstacles):
                # Add the new node to the tree
                new_node.parent = nearest_node
                new_node.cost = nearest_node.cost + math.sqrt(
                    (new_node.position[0] - nearest_node.position[0]) ** 2 +
                    (new_node.position[1] - nearest_node.position[1]) ** 2
                )
                self.nodes.append(new_node)

                # Find all nodes near the new node for rewiring
                near_indices = self._get_near_indices(new_node, neighborhood_radius)

                # Connect new node to lowest cost parent
                self._choose_parent(new_node, near_indices)

                # Rewire near nodes through new node if it provides lower cost
                self._rewire(new_node, near_indices)

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
                    # Try to connect to goal directly
                    if self._check_collision_free(new_node.position, goal, obstacles):
                        goal_node = RRTNode(goal)
                        goal_node.parent = new_node
                        goal_node.cost = new_node.cost + dist_to_goal
                        self.nodes.append(goal_node)

                        # Find all nodes near the goal for rewiring
                        goal_near_indices = self._get_near_indices(goal_node, neighborhood_radius)

                        # Connect goal to lowest cost parent
                        self._choose_parent(goal_node, goal_near_indices)

                        self.nearest_node_to_goal = goal_node
                        nearest_dist = 0

        # Extract the path
        path = self._extract_path(self.nearest_node_to_goal)
        self.current_path_index = 0
        self.path = path

        # Post-process path for smoothness
        smoothed_path = self._smooth_path(path, obstacles)
        self.path = smoothed_path

        return smoothed_path

    def _check_collision_free(self, from_pos: Tuple[float, float], to_pos: Tuple[float, float], obstacles: List[Dict[str, Any]]) -> bool:
        """
        Check if the path from from_pos to to_pos is valid.
        检查从from_pos到to_pos的路径是否有效。

        Args:
            from_pos: Starting position
                     起始位置
            to_pos: Ending position
                   终止位置
            obstacles: List of obstacles (not used in this implementation)
                      障碍物列表（在此实现中未使用）

        Returns:
            Always returns True as obstacles are removed
            始终返回True，因为障碍物已被移除
        """
        # In the new implementation, we don't check for obstacles
        return True

    def _get_near_indices(self, node: RRTNode, radius: float) -> List[int]:
        """
        Find indices of nodes near the given node.
        查找给定节点附近的节点索引。

        Args:
            node: Node to find neighbors of
                 要查找邻居的节点
            radius: Neighborhood radius
                   邻域半径

        Returns:
            List of indices of nearby nodes
            附近节点的索引列表
        """
        indices = []

        for i, other_node in enumerate(self.nodes):
            if other_node == node:
                continue

            # Calculate distance
            distance = math.sqrt(
                (node.position[0] - other_node.position[0]) ** 2 +
                (node.position[1] - other_node.position[1]) ** 2
            )

            # Check if within radius
            if distance <= radius:
                indices.append(i)

        return indices

    def _choose_parent(self, node: RRTNode, near_indices: List[int]) -> None:
        """
        Choose the parent that results in the lowest cost path to the given node.
        选择到给定节点的成本最低的路径的父节点。

        Args:
            node: Node to choose parent for
                 要选择父节点的节点
            near_indices: List of indices of nearby nodes
                         附近节点的索引列表
        """
        if not near_indices:
            return

        # No obstacles in the new implementation
        obstacles = []

        # Find minimum cost parent
        min_cost = node.cost
        min_node = node.parent

        for idx in near_indices:
            near_node = self.nodes[idx]

            # Calculate potential cost
            edge_cost = math.sqrt(
                (node.position[0] - near_node.position[0]) ** 2 +
                (node.position[1] - near_node.position[1]) ** 2
            )
            potential_cost = near_node.cost + edge_cost

            # If lower cost and collision-free
            if potential_cost < min_cost and self._check_collision_free(near_node.position, node.position, obstacles):
                min_cost = potential_cost
                min_node = near_node

        # Set new parent and cost
        if min_node != node.parent:
            node.parent = min_node
            node.cost = min_cost

    def _rewire(self, node: RRTNode, near_indices: List[int]) -> None:
        """
        Rewire near nodes through the given node if it provides lower cost.
        如果通过给定节点提供更低的成本，则重新连接附近的节点。

        Args:
            node: Node to rewire through
                 要通过其重新连接的节点
            near_indices: List of indices of nearby nodes
                         附近节点的索引列表
        """
        # No obstacles in the new implementation
        obstacles = []

        # Rewire
        for idx in near_indices:
            near_node = self.nodes[idx]

            # Calculate potential cost
            edge_cost = math.sqrt(
                (node.position[0] - near_node.position[0]) ** 2 +
                (node.position[1] - near_node.position[1]) ** 2
            )
            potential_cost = node.cost + edge_cost

            # If lower cost and collision-free
            if potential_cost < near_node.cost and self._check_collision_free(node.position, near_node.position, obstacles):
                # Rewire
                near_node.parent = node
                near_node.cost = potential_cost

    def _smooth_path(self, path: List[Tuple[float, float]], obstacles: List[Dict[str, Any]]) -> List[Tuple[float, float]]:
        """
        Smooth the path to remove unnecessary waypoints.
        平滑路径以移除不必要的航点。

        Args:
            path: Original path
                 原始路径
            obstacles: List of obstacles (not used in this implementation)
                      障碍物列表（在此实现中未使用）

        Returns:
            Smoothed path
            平滑后的路径
        """
        if len(path) <= 2:
            return path

        # Initialize smoothed path with first point
        smoothed_path = [path[0]]

        # In the simplified implementation, we can just go directly from start to end
        smoothed_path.append(path[-1])

        return smoothed_path

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