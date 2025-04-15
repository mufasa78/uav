# UAV Path Planning Simulation Framework

## Latest Updates

Based on experimental results, the framework has been significantly enhanced with the following improvements:

- **Optimized User Distribution**: Users are now arranged in two neat rows with equal spacing for more consistent testing
- **Removed Obstacle Components**: Simplified the environment to focus on algorithm comparison without obstacle interference
- **Enhanced Performance Metrics**: Added energy efficiency, task completion rate, and service latency measurements
- **Advanced Visualization**: Implemented comprehensive comparative analysis tools with detailed metrics
- **Algorithm Comparison**: Updated to compare MCTS, QL, and DQN algorithms with the new environment configuration

## Project Overview

This framework is a comprehensive Python-based simulation environment for advanced Unmanned Aerial Vehicle (UAV) path planning research. It implements and compares multiple advanced algorithmic approaches including Monte Carlo Tree Search (MCTS), Q-Learning (QL), and Deep Q-Network (DQN) algorithms.

The system provides a platform for developing, testing, and comparing path planning strategies in controlled environments with stationary users in organized distributions, energy constraints, and service-oriented tasks. This makes it well-suited for research in various UAV application domains including telecommunications, data collection, surveillance, and service provisioning.

![UAV Path Planning Simulation](images/uav_path_planning.png)

## Key Features

- **Multiple Path Planning Algorithms**: Implementation of three state-of-the-art algorithms:
  - **Monte Carlo Tree Search (MCTS)**: A sampling-based planning algorithm using tree search with exploration-exploitation balance.
  - **Rapidly-exploring Random Tree (RRT*)**: An optimized sampling-based algorithm with asymptotic optimality guarantees.
  - **A* Search**: A classic heuristic-based shortest path algorithm with grid-based world representation.

- **Dynamic Environment Simulation**: Realistic modeling of:
  - Stationary ground users arranged in two neat rows with equal spacing
  - Task generation based on probabilistic models
  - Energy consumption tracking with different power modes (hover, movement, communication)
  - Optimized for direct path planning without obstacles

- **Enhanced Performance Metrics**:
  - Number of serviced tasks
  - Energy efficiency (data processed per unit of energy)
  - Task completion rate
  - Average service latency
  - Composite performance score
  - Total distance traveled
  - Data processed

- **Enhanced Interactive Web Interface**:
  - Real-time visualization of UAV trajectories with two-row user distribution
  - Advanced algorithm performance comparisons with comprehensive metrics
  - Detailed visualization of energy consumption patterns
  - Bilingual support (English and Chinese)

- **Modular Architecture**:
  - Extensible algorithm framework
  - Configurable simulation parameters
  - Separation of concerns between algorithms, environment, and visualization

## System Architecture

The framework is organized into several modules:

1. **Algorithms**: Contains implementations of path planning algorithms:
   - `base.py`: Base class for all algorithms
   - `mcts.py`: Monte Carlo Tree Search implementation
   - `rrt.py`: Rapidly-exploring Random Tree (RRT*) implementation
   - `astar.py`: A* Search implementation

2. **Simulation**: Environment modeling and dynamics:
   - `environment.py`: Main simulation environment
   - `uav.py`: UAV model with energy and movement characteristics

3. **Visualization**: Plotting and result analysis:
   - `plotter.py`: Functions for visualizing trajectories and metrics

4. **Utils**: Supporting functionality:
   - `config.py`: Configuration parameters
   - `metrics.py`: Performance evaluation metrics

5. **Web**: Web application interface:
   - `app.py`: Flask web application
   - `templates/`: HTML templates for the web interface

## Theoretical Background

### Monte Carlo Tree Search (MCTS)

MCTS is a heuristic search algorithm that combines tree search with random sampling. It constructs a search tree through four main steps:

1. **Selection**: Starting from the root, select successive child nodes down to a leaf node using a selection strategy (UCT - Upper Confidence Bound for Trees).
2. **Expansion**: Create one or more child nodes and select one from them.
3. **Simulation**: Conduct a random simulation from the selected node.
4. **Backpropagation**: Use the results to update information in the nodes on the path from the selected node to the root.

Our implementation uses a discounted reward system that balances service completion, energy consumption, and distance optimization. The exploration parameter is dynamically adjusted based on the simulation stage.

### Rapidly-exploring Random Tree (RRT*)

RRT* is an asymptotically optimal variant of RRT that builds a space-filling tree to search a state space efficiently. Key components include:

1. **Random Sampling**: Generate random points in the state space.
2. **Nearest Neighbor**: Find the nearest existing node in the tree.
3. **Steering**: Move from nearest node toward the sampled point by a limited step size.
4. **Rewiring**: Optimize the tree by reconnecting nodes if a shorter path is found.

Our implementation includes enhanced features such as:
- Adaptive sampling with goal biasing
- Path smoothing post-processing
- Collision checking with safety buffers
- Dynamic neighborhood radius calculation

### A* Algorithm

A* is a best-first search algorithm that uses a heuristic to guide path search. It maintains two sets of nodes:

1. **Open Set**: Nodes that have been discovered but not fully explored.
2. **Closed Set**: Nodes that have been fully explored.

At each iteration, A* selects the node with the lowest f-cost (f = g + h), where:
- g: Cost from start to the current node
- h: Heuristic estimate of cost from current node to goal

Our implementation uses an octile distance heuristic for diagonal movement and includes grid-based discretization with configurable resolution.

## Performance Comparison

The framework enables quantitative comparison between algorithms based on several metrics:

| Algorithm | Serviced Tasks | Energy Efficiency | Task Completion Rate | Performance Score |
|-----------|---------------|-------------------|---------------------|-------------------|
| MCTS      | High          | Medium            | High                | High              |
| QL        | Medium        | High              | Medium              | Medium            |
| DQN       | High          | High              | High                | High              |

Qualitative observations based on experimental results:
- MCTS excels in environments with stationary users in organized distributions
- QL (Q-Learning) provides efficient energy usage with consistent performance
- DQN combines the strengths of both approaches with high overall performance scores

## Implementation Details

### MCTS Implementation Highlights

```python
def _rollout(self, node: Node) -> float:
    """Perform a rollout from the node."""
    # Create a copy of the environment
    env_copy = self._create_env_copy(node.state)

    # Track rewards at each step
    accumulated_reward = 0.0
    discount_factor = 0.95  # Discount factor for future rewards

    # Improved rollout with heuristic-based actions
    for step in range(self.rollout_depth):
        if env_copy.is_done():
            break

        # Get possible actions
        possible_actions = self._get_possible_actions(env_copy.get_state())

        if not possible_actions:
            break

        # Choose action according to improved rollout policy
        action = self._improved_rollout_policy(env_copy.get_state(), possible_actions)

        # Apply action and accumulate rewards
        env_copy.step(action)
        immediate_reward = self._calculate_immediate_reward(
            current_state, env_copy.get_state()
        )
        accumulated_reward += (discount_factor ** step) * immediate_reward

    return accumulated_reward
```

### RRT* Implementation Highlights

```python
def _choose_parent(self, node: RRTNode, near_indices: List[int]) -> None:
    """Choose the parent that results in the lowest cost path."""
    if not near_indices:
        return

    # Get obstacles
    obstacles = self._get_obstacles()

    # Find minimum cost parent
    min_cost = node.cost
    min_node = node.parent

    for idx in near_indices:
        near_node = self.nodes[idx]

        # Calculate potential cost
        edge_cost = self._calculate_distance(node.position, near_node.position)
        potential_cost = near_node.cost + edge_cost

        # If lower cost and collision-free
        if (potential_cost < min_cost and
            self._check_collision_free(near_node.position, node.position, obstacles)):
            min_cost = potential_cost
            min_node = near_node

    # Set new parent and cost
    if min_node != node.parent:
        node.parent = min_node
        node.cost = min_cost
```

### A* Implementation Highlights

```python
def plan_path(self, start: Tuple[float, float], goal: Tuple[float, float]) -> List[Tuple[float, float]]:
    """Plan a path using A*."""
    # Discretize start and goal positions
    start_grid = self._to_grid(start)
    goal_grid = self._to_grid(goal)

    # Create start and goal nodes
    start_node = AStarNode(start_grid, 0.0, self._heuristic(start_grid, goal_grid))
    goal_node = AStarNode(goal_grid)

    # Initialize open and closed sets
    open_set = [start_node]  # Priority queue
    closed_set = set()

    # Main A* loop
    while open_set:
        # Get node with lowest f_cost
        current = heapq.heappop(open_set)

        # Check if goal reached
        if current.position == goal_grid:
            return self._extract_path(current)

        # Add to closed set
        closed_set.add(current.position)

        # Generate and process neighbors
        for neighbor in self._get_neighbors(current):
            if neighbor.position in closed_set:
                continue

            # Calculate tentative g_cost
            tentative_g_cost = current.g_cost + self._distance(
                current.position, neighbor.position
            )

            # Update if better path found
            if self._update_neighbor(neighbor, current, tentative_g_cost, goal_grid):
                heapq.heappush(open_set, neighbor)

    # If no path found, return straight line
    return [start, goal]
```

## Usage Instructions

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/username/uav-path-planning.git
   cd uav-path-planning
   ```

2. Install required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Running the Web Interface

Start the web server:
```bash
python main.py
```

Access the web interface at: http://localhost:5000

### Command Line Usage

Run a single algorithm:
```bash
python main.py --cli --algorithm mcts --sim_time 300
```

Compare all algorithms:
```bash
python main.py --cli --compare --sim_time 300
```

### Configuration

Modify parameters in `utils/config.py` to adjust:
- World size and environment properties
- UAV characteristics (speed, energy consumption)
- User parameters (task generation probability, data size)
- Performance metrics weights (energy efficiency, task completion, latency)
- Algorithm-specific parameters (exploration weights, step sizes)
- Simulation duration and service parameters

## Results and Analysis

The framework provides enhanced visualization tools for comprehensive result analysis:

1. **Advanced Trajectory Visualization**: Shows the path taken by the UAV with users arranged in two neat rows.
2. **Energy Consumption Analysis**: Detailed plots of energy usage over time with comparative views.
3. **Comprehensive Metrics Dashboard**: Interactive charts comparing all performance metrics across algorithms.
4. **Service Statistics**: Detailed data on task completion, latency, and service efficiency.
5. **Composite Performance Visualization**: Combined view of all metrics weighted by importance.

Example comparative analysis based on experimental results shows:
- MCTS achieves higher service rates in the new user distribution pattern
- QL provides the most energy-efficient paths with good task completion
- DQN balances energy efficiency and service quality for optimal overall performance

## Recent Improvements

The framework has been recently enhanced with the following improvements:

1. **User Distribution Optimization**:
   - Implemented two-row user distribution with equal spacing
   - Removed obstacle components for more focused algorithm comparison
   - Simplified collision detection for improved performance

2. **Enhanced Performance Metrics**:
   - Added energy efficiency calculation (data processed per unit of energy)
   - Implemented task completion rate tracking
   - Added service latency measurement
   - Created composite performance score with configurable weights

3. **Visualization Enhancements**:
   - Advanced comparative visualization with GridSpec layout
   - Detailed energy consumption analysis
   - Comprehensive algorithm comparison dashboards

## Future Directions

Planned future enhancements include:

1. **Algorithm Extensions**:
   - Further optimization of QL and DQN implementations
   - Multi-agent path planning
   - Hybrid algorithms combining strengths of existing methods

2. **Environment Enhancements**:
   - Dynamic user task priority modeling
   - More realistic communication models
   - Battery degradation modeling

3. **User Interface**:
   - Real-time parameter tuning
   - Interactive scenario creation
   - Exportable analysis reports

## Code Documentation

Each module is thoroughly documented with docstrings in both English and Chinese to facilitate international research collaboration. Key classes and methods include:

- `PathPlanningAlgorithm`: Base class for all algorithms
- `Environment`: Main simulation environment
- `UAV`: UAV model with energy and movement
- Visualization functions for results analysis

## References

1. Browne, C. B., et al. (2012). *A Survey of Monte Carlo Tree Search Methods*. IEEE Transactions on Computational Intelligence and AI in Games, 4(1), 1-43.
2. Karaman, S., & Frazzoli, E. (2011). *Sampling-based algorithms for optimal motion planning*. The International Journal of Robotics Research, 30(7), 846-894.
3. Hart, P. E., Nilsson, N. J., & Raphael, B. (1968). *A Formal Basis for the Heuristic Determination of Minimum Cost Paths*. IEEE Transactions on Systems Science and Cybernetics, 4(2), 100-107.
4. Cabreira, T. M., et al. (2019). *An Energy-Aware Path Planning for UAV in Large-Scale Farmlands*. Sensors, 19(18), 3974.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contributors

- Main Developer: [Your Name]
- Academic Advisors: [Advisor Names]

## Acknowledgments

This research was supported by [University/Institution Name] and made possible through contributions from the robotics and AI research community.#   u a v 
 
 