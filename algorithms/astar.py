"""
A* path planning algorithm implementation for UAV path planning.
A*路径规划算法在无人机路径规划中的实现。
"""

import random
import math
import heapq
from typing import Dict, List, Tuple, Any, Optional, Set

from algorithms.base import PathPlanningAlgorithm
from utils.config import (
    WORLD_SIZE,
    SERVICE_RANGE
)

class AStarNode:
    """
    Node in the A* algorithm.
    A*算法中的节点。
    """
    
    def __init__(self, position: Tuple[float, float], g_cost: float = 0.0, h_cost: float = 0.0):
        """
        Initialize a node.
        初始化节点。
        
        Args:
            position: Position of the node (x, y)
                     节点的位置坐标 (x, y)
            g_cost: Cost from start to this node
                    从起点到该节点的成本
            h_cost: Heuristic cost from this node to goal
                    从该节点到目标的启发式成本
        """
        self.position = position
        self.g_cost = g_cost
        self.h_cost = h_cost
        self.f_cost = g_cost + h_cost
        self.parent: Optional['AStarNode'] = None
        
    def __lt__(self, other: 'AStarNode') -> bool:
        """
        Comparison for priority queue.
        优先队列比较函数。
        
        Args:
            other: Other node to compare with
                   其他要比较的节点
            
        Returns:
            True if this node has lower f_cost, False otherwise
            如果此节点的f_cost更低则返回True，否则返回False
        """
        if self.f_cost == other.f_cost:
            return self.h_cost < other.h_cost
        return self.f_cost < other.f_cost
    
    def __eq__(self, other: object) -> bool:
        """
        Equality check for nodes based on position.
        基于位置的节点相等性检查。
        
        Args:
            other: Other node to compare with
                   其他要比较的节点
            
        Returns:
            True if positions are equal, False otherwise
            如果位置相等则返回True，否则返回False
        """
        if not isinstance(other, AStarNode):
            return False
        return self.position == other.position
    
    def __hash__(self) -> int:
        """
        Hash function for nodes based on position.
        基于位置的节点哈希函数。
        
        Returns:
            Hash of the node's position
            节点位置的哈希值
        """
        return hash(self.position)

class AStarAlgorithm(PathPlanningAlgorithm):
    """
    A* algorithm for UAV path planning.
    用于无人机路径规划的A*算法。
    """
    
    def __init__(
        self,
        grid_size: float = 20.0,  # Grid size for discretization
        diagonal_movement: bool = True,  # Allow diagonal movement
        weight: float = 1.0  # Weight for heuristic (1.0 = standard A*)
    ):
        """
        Initialize the A* algorithm.
        初始化A*算法。
        
        Args:
            grid_size: Grid size for discretization
                      离散化的网格大小
            diagonal_movement: Allow diagonal movement
                              是否允许对角线移动
            weight: Weight for heuristic (1.0 = standard A*)
                   启发式函数的权重（1.0 = 标准A*）
        """
        super().__init__("A*")
        self.grid_size = grid_size
        self.diagonal_movement = diagonal_movement
        self.weight = weight
        
        # Additional attributes
        self.path = []
        self.current_path_index = 0
        self.current_goal_index = 0
        self.goals = []  # List of (position, user_id) pairs
        self.planning_in_progress = False
        self.obstacles = []  # List of obstacle positions and radii
    
    def setup(self, env) -> None:
        """
        Set up the algorithm with the environment.
        
        Args:
            env: Simulation environment
        """
        super().setup(env)
        
        # Reset path planning
        self.path = []
        self.current_path_index = 0
        self.current_goal_index = 0
        self.goals = []
        self.planning_in_progress = False
        
        # Get obstacles from environment
        state = env.get_state()
        self.obstacles = state.get('obstacles', [])
    
    def compute_action(self, state: Dict[str, Any]) -> Tuple[Optional[Tuple[float, float]], Optional[int]]:
        """
        Compute the next action using A*.
        使用A*计算下一步动作。
        
        Args:
            state: Current state of the environment
                  环境的当前状态
            
        Returns:
            Tuple of (target_position, user_id_to_service)
            返回一个元组：(目标位置, 要服务的用户ID)
        """
        # Get current UAV position
        uav_position = state.get('uav_position', (0, 0))
        
        # Update obstacles
        self.obstacles = state.get('obstacles', [])
        
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
            self.path = self.plan_path(uav_position, goal)
            self.current_path_index = 0
            
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
            service_range = state.get('service_range', SERVICE_RANGE)
            distance = math.sqrt((next_position[0] - uav_position[0]) ** 2 + (next_position[1] - uav_position[1]) ** 2)
            if distance < service_range / 2:  # Use half service range as a threshold
                # Move to next waypoint
                self.current_path_index += 1
                
                # If we reached the end of the path
                if self.current_path_index >= len(self.path):
                    # Move to next goal
                    self.current_goal_index += 1
                    
                    # If we have more goals with positions, plan a path to the next goal
                    if self.current_goal_index < len(self.goals) and self.goals[self.current_goal_index][0] is not None:
                        self.path = self.plan_path(uav_position, self.goals[self.current_goal_index][0])
                        self.current_path_index = 0
                    else:
                        # Reset planning if we have no more position goals
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
        Plan a path using A*.
        使用A*规划路径。
        
        Args:
            start: Start position
                  起始位置
            goal: Goal position
                 目标位置
            
        Returns:
            List of waypoints
            航点列表
        """
        # Discretize start and goal positions to grid
        start_grid = self._to_grid(start)
        goal_grid = self._to_grid(goal)
        
        # Create start and goal nodes
        start_node = AStarNode(start_grid, 0.0, self._heuristic(start_grid, goal_grid))
        goal_node = AStarNode(goal_grid)
        
        # Initialize open and closed sets
        open_set: List[AStarNode] = []
        closed_set: Set[Tuple[float, float]] = set()
        
        # Add start node to open set
        heapq.heappush(open_set, start_node)
        
        # Main A* loop
        while open_set:
            # Get node with lowest f_cost
            current = heapq.heappop(open_set)
            
            # If we reached the goal
            if current.position == goal_grid:
                # Extract and return the path
                path = self._extract_path(current)
                
                # Convert grid coordinates back to world coordinates
                return [self._to_world(pos) for pos in path]
            
            # Add current node to closed set
            closed_set.add(current.position)
            
            # Generate neighbors
            neighbors = self._get_neighbors(current)
            
            for neighbor in neighbors:
                # Skip if in closed set
                if neighbor.position in closed_set:
                    continue
                
                # Calculate tentative g_cost
                # Convert position tuples to int for the distance function
                curr_pos = (int(current.position[0]), int(current.position[1]))
                neigh_pos = (int(neighbor.position[0]), int(neighbor.position[1]))
                tentative_g_cost = current.g_cost + self._distance(curr_pos, neigh_pos)
                
                # Check if neighbor is in open set
                in_open_set = False
                for node in open_set:
                    if node.position == neighbor.position:
                        in_open_set = True
                        
                        # If we found a better path
                        if tentative_g_cost < node.g_cost:
                            # Update g_cost and parent
                            node.g_cost = tentative_g_cost
                            node.f_cost = node.g_cost + node.h_cost
                            node.parent = current
                        break
                
                # If not in open set, add it
                if not in_open_set:
                    # Set g_cost and parent
                    neighbor.g_cost = tentative_g_cost
                    # Convert position tuples to int if needed for the heuristic function
                    grid_pos = (int(neighbor.position[0]), int(neighbor.position[1]))
                    grid_goal = (int(goal_grid[0]), int(goal_grid[1]))
                    neighbor.h_cost = self._heuristic(grid_pos, grid_goal)
                    neighbor.f_cost = neighbor.g_cost + neighbor.h_cost * self.weight
                    neighbor.parent = current
                    
                    # Add to open set
                    heapq.heappush(open_set, neighbor)
        
        # If no path found, return straight line path (will try to avoid obstacles during execution)
        return [start, goal]
    
    def _to_grid(self, position: Tuple[float, float]) -> Tuple[int, int]:
        """
        Convert world coordinates to grid coordinates.
        将世界坐标转换为网格坐标。
        
        Args:
            position: Position in world coordinates
                     世界坐标中的位置
            
        Returns:
            Position in grid coordinates
            网格坐标中的位置
        """
        x = int(position[0] / self.grid_size)
        y = int(position[1] / self.grid_size)
        return (x, y)
    
    def _to_world(self, grid_position: Tuple[int, int]) -> Tuple[float, float]:
        """
        Convert grid coordinates to world coordinates.
        将网格坐标转换为世界坐标。
        
        Args:
            grid_position: Position in grid coordinates
                          网格坐标中的位置
            
        Returns:
            Position in world coordinates
            世界坐标中的位置
        """
        x = (grid_position[0] + 0.5) * self.grid_size  # Center of the grid cell
        y = (grid_position[1] + 0.5) * self.grid_size  # Center of the grid cell
        return (x, y)
    
    def _heuristic(self, position: Tuple[int, int], goal: Tuple[int, int]) -> float:
        """
        Heuristic function for A*.
        A*的启发式函数。
        
        Args:
            position: Current position
                     当前位置
            goal: Goal position
                 目标位置
            
        Returns:
            Heuristic cost
            启发式成本
        """
        # Octile distance (allows diagonal movement)
        dx = abs(position[0] - goal[0])
        dy = abs(position[1] - goal[1])
        
        if self.diagonal_movement:
            # Octile distance
            return max(dx, dy) + (math.sqrt(2) - 1) * min(dx, dy)
        else:
            # Manhattan distance
            return dx + dy
    
    def _distance(self, a: Tuple[int, int], b: Tuple[int, int]) -> float:
        """
        Calculate distance between two grid positions.
        计算两个网格位置之间的距离。
        
        Args:
            a: First position
               第一个位置
            b: Second position
               第二个位置
            
        Returns:
            Distance between positions
            位置之间的距离
        """
        dx = abs(a[0] - b[0])
        dy = abs(a[1] - b[1])
        
        if self.diagonal_movement:
            # Octile distance
            return max(dx, dy) + (math.sqrt(2) - 1) * min(dx, dy)
        else:
            # Manhattan distance
            return dx + dy
    
    def _get_neighbors(self, node: AStarNode) -> List[AStarNode]:
        """
        Get neighboring nodes.
        获取相邻节点。
        
        Args:
            node: Current node
                 当前节点
            
        Returns:
            List of neighbor nodes
            相邻节点列表
        """
        neighbors = []
        x, y = node.position
        
        # Get world size
        world_size = WORLD_SIZE
        if self.env:
            state = self.env.get_state()
            if state:
                world_size = state.get('world_size', WORLD_SIZE)
        
        # Convert to grid size
        max_x = int(world_size[0] / self.grid_size)
        max_y = int(world_size[1] / self.grid_size)
        
        # Directions: N, NE, E, SE, S, SW, W, NW
        directions = [
            (0, 1), (1, 1), (1, 0), (1, -1),
            (0, -1), (-1, -1), (-1, 0), (-1, 1)
        ]
        
        # If diagonal movement is not allowed, only use cardinal directions
        if not self.diagonal_movement:
            directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            
            # Check bounds
            if nx < 0 or nx >= max_x or ny < 0 or ny >= max_y:
                continue
            
            # Check for obstacles
            neighbor_position = (nx, ny)
            world_position = self._to_world(neighbor_position)
            
            if self._check_collision(world_position):
                continue
            
            # Create neighbor node
            neighbor = AStarNode(neighbor_position)
            neighbors.append(neighbor)
        
        return neighbors
    
    def _check_collision(self, position: Tuple[float, float]) -> bool:
        """
        Check if position collides with any obstacle.
        检查位置是否与任何障碍物碰撞。
        
        Args:
            position: Position to check
                     要检查的位置
            
        Returns:
            True if collision, False otherwise
            如果碰撞则返回True，否则返回False
        """
        for obstacle in self.obstacles:
            obstacle_pos = obstacle.get('position', (0, 0))
            obstacle_radius = obstacle.get('radius', 0)
            
            # Calculate distance to obstacle
            distance = math.sqrt((position[0] - obstacle_pos[0]) ** 2 + (position[1] - obstacle_pos[1]) ** 2)
            
            # Check if collision (add some buffer)
            if distance < obstacle_radius + self.grid_size / 2:
                return True
        
        return False
    
    def _extract_path(self, end_node: AStarNode) -> List[Tuple[int, int]]:
        """
        Extract path from the tree.
        从树中提取路径。
        
        Args:
            end_node: End node of the path
                     路径的终点节点
            
        Returns:
            List of waypoints in grid coordinates
            网格坐标中的航点列表
        """
        path = []
        current = end_node
        
        # Traverse from end node to start node
        while current:
            path.append(current.position)
            current = current.parent
        
        # Reverse to get path from start to end
        path.reverse()
        
        # Path smoothing
        return self._smooth_path(path)
    
    def _smooth_path(self, path: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
        """
        Smooth the path to remove unnecessary waypoints.
        平滑路径以移除不必要的航点。
        
        Args:
            path: Original path
                 原始路径
            
        Returns:
            Smoothed path
            平滑后的路径
        """
        if len(path) <= 2:
            return path
        
        # Initialize smoothed path with first point
        smoothed_path = [path[0]]
        
        # For each point, check if we can go directly to a later point
        i = 0
        while i < len(path) - 1:
            # Find the furthest point we can go to directly
            furthest = i + 1
            for j in range(i + 2, len(path)):
                # Check if direct path from i to j is collision-free
                if self._is_path_clear(path[i], path[j]):
                    furthest = j
                else:
                    break
            
            # Add the furthest point to the smoothed path
            smoothed_path.append(path[furthest])
            
            # Continue from that point
            i = furthest
        
        return smoothed_path
    
    def _is_path_clear(self, a: Tuple[int, int], b: Tuple[int, int]) -> bool:
        """
        Check if the direct path between two grid positions is clear.
        检查两个网格位置之间的直接路径是否畅通。
        
        Args:
            a: First position
               第一个位置
            b: Second position
               第二个位置
            
        Returns:
            True if path is clear, False otherwise
            如果路径畅通则返回True，否则返回False
        """
        # Convert to world coordinates
        a_world = self._to_world(a)
        b_world = self._to_world(b)
        
        # Number of points to check along the line
        num_points = max(abs(b[0] - a[0]), abs(b[1] - a[1])) * 2
        
        # Check points along the line
        for i in range(1, num_points):
            t = i / num_points
            x = a_world[0] + t * (b_world[0] - a_world[0])
            y = a_world[1] + t * (b_world[1] - a_world[1])
            
            if self._check_collision((x, y)):
                return False
        
        return True