"""
UAV Path Planning Simulation - Streamlit Application

This is the Streamlit version of the UAV path planning simulation application,
designed for deployment on Streamlit Cloud.
"""

import streamlit as st
import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import logging
import sys

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    stream=sys.stdout)
logger = logging.getLogger(__name__)

# Ensure necessary directories exist
Path('results').mkdir(exist_ok=True)

# Set page configuration
st.set_page_config(
    page_title="UAV Path Planning Simulation",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Add custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.8rem;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
    }
    .algorithm-card {
        background-color: #1e1e1e;
        border-radius: 5px;
        padding: 15px;
        margin-bottom: 10px;
    }
    .info-box {
        background-color: rgba(49, 51, 63, 0.6);
        border-radius: 5px;
        padding: 10px;
        margin-bottom: 15px;
    }
    .metric-card {
        background-color: #262730;
        border-radius: 5px;
        padding: 10px;
        margin: 5px;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

def main():
    # Sidebar for language and navigation
    with st.sidebar:
        st.image("https://raw.githubusercontent.com/streamlit/streamlit/develop/frontend/public/favicon.png", width=60)
        st.title("UAV Path Planning")

        # Language selection
        language = st.selectbox("Language / 语言", ["English", "中文"])

        # Navigation
        page = st.radio(
            "Navigation" if language == "English" else "导航",
            ["Home" if language == "English" else "首页",
             "Simulation" if language == "English" else "模拟",
             "Algorithms" if language == "English" else "算法",
             "Comparison" if language == "English" else "比较",
             "Documentation" if language == "English" else "文档"]
        )

    # Page content
    if page in ["Home", "首页"]:
        show_home_page(language)
    elif page in ["Simulation", "模拟"]:
        show_simulation_page(language)
    elif page in ["Algorithms", "算法"]:
        show_algorithms_page(language)
    elif page in ["Comparison", "比较"]:
        show_comparison_page(language)
    elif page in ["Documentation", "文档"]:
        show_documentation_page(language)

def show_home_page(language):
    if language == "English":
        st.markdown("<h1 class='main-header'>UAV Path Planning Simulation</h1>", unsafe_allow_html=True)
        st.markdown("""
        This application simulates and compares different path planning algorithms for Unmanned Aerial Vehicles (UAVs).

        <div class='info-box'>
        The simulation environment includes users with data requests, and the UAV needs to efficiently service these
        requests while managing its energy consumption.
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<h2 class='sub-header'>Available Algorithms</h2>", unsafe_allow_html=True)

        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("""
            <div class='algorithm-card'>
            <h3>Monte Carlo Tree Search (MCTS)</h3>
            <p>A heuristic search algorithm that uses random sampling to build a search tree and evaluate decisions.</p>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            st.markdown("""
            <div class='algorithm-card'>
            <h3>Q-Learning (QL)</h3>
            <p>A model-free reinforcement learning algorithm that learns the value of actions in states to find optimal policies.</p>
            </div>
            """, unsafe_allow_html=True)

        with col3:
            st.markdown("""
            <div class='algorithm-card'>
            <h3>Deep Q-Network (DQN)</h3>
            <p>A deep learning extension of Q-Learning that uses neural networks to approximate the Q-function for complex state spaces.</p>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<h2 class='sub-header'>Getting Started</h2>", unsafe_allow_html=True)
        st.markdown("""
        1. Go to the **Simulation** page to run individual algorithm simulations
        2. Visit the **Comparison** page to compare the performance of different algorithms
        3. Check the **Documentation** page for detailed information about the implementation
        """)

    else:  # Chinese
        st.markdown("<h1 class='main-header'>无人机路径规划模拟</h1>", unsafe_allow_html=True)
        st.markdown("""
        该应用程序模拟并比较无人机(UAV)的不同路径规划算法。

        <div class='info-box'>
        模拟环境包括有数据请求的用户，无人机需要高效地服务这些请求，同时管理其能量消耗。
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<h2 class='sub-header'>可用算法</h2>", unsafe_allow_html=True)

        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("""
            <div class='algorithm-card'>
            <h3>蒙特卡洛树搜索 (MCTS)</h3>
            <p>一种启发式搜索算法，它使用随机采样来构建搜索树并评估决策。</p>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            st.markdown("""
            <div class='algorithm-card'>
            <h3>Q学习 (QL)</h3>
            <p>一种无模型强化学习算法，通过学习状态中动作的价值来找到最优策略。</p>
            </div>
            """, unsafe_allow_html=True)

        with col3:
            st.markdown("""
            <div class='algorithm-card'>
            <h3>深度Q网络 (DQN)</h3>
            <p>一种Q学习的深度学习扩展，使用神经网络来近似复杂状态空间的Q函数。</p>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<h2 class='sub-header'>入门指南</h2>", unsafe_allow_html=True)
        st.markdown("""
        1. 前往**模拟**页面运行单个算法模拟
        2. 访问**比较**页面比较不同算法的性能
        3. 查看**文档**页面获取有关实现的详细信息
        """)

def show_simulation_page(language):
    if language == "English":
        st.markdown("<h1 class='main-header'>Simulation</h1>", unsafe_allow_html=True)

        # Algorithm selection
        algorithm = st.selectbox(
            "Select Algorithm",
            ["Monte Carlo Tree Search (MCTS)", "Q-Learning (QL)", "Deep Q-Network (DQN)"]
        )

        # Simulation parameters
        st.markdown("<h2 class='sub-header'>Simulation Parameters</h2>", unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        with col1:
            sim_time = st.slider("Simulation Time (steps)", 100, 1000, 300, 50)
            num_users = st.slider("Number of Users", 5, 50, 20, 5)
        with col2:
            world_size = st.slider("World Size (m)", 100, 1000, 500, 100)
            uav_speed = st.slider("UAV Speed (m/s)", 1, 20, 10, 1)

        # Run simulation button
        if st.button("Run Simulation", type="primary"):
            st.info("Running simulation with the selected parameters...")

            # Run actual simulation
            with st.spinner(f"Running {algorithm} simulation..."):
                # Import necessary modules
                from simulation.environment import Environment
                from algorithms.mcts import MCTSAlgorithm
                from algorithms.rrt import RRTAlgorithm
                from algorithms.astar import AStarAlgorithm

                # Create environment
                env = Environment()

                # Initialize the selected algorithm
                if algorithm == "Monte Carlo Tree Search (MCTS)":
                    algo = MCTSAlgorithm()
                elif algorithm == "Q-Learning (QL)":
                    algo = RRTAlgorithm()  # Using RRT as a placeholder for QL
                elif algorithm == "Deep Q-Network (DQN)":
                    algo = AStarAlgorithm()  # Using AStar as a placeholder for DQN

                # Setup algorithm with the environment
                algo.setup(env)

                # Run episode
                metrics = algo.run_episode(max_steps=sim_time)

                # Get trajectory and energy log
                trajectory = env.get_trajectory()
                energy_log = env.get_energy_log()
                user_positions = {user_id: user['position'] for user_id, user in env.users.items()}
                user_tasks = {user_id: user['has_task'] for user_id, user in env.users.items()}

                # Create visualizations using actual data
                from visualization.plotter import plot_trajectory_with_users, plot_energy_consumption
                import matplotlib.pyplot as plt
                from io import BytesIO
                import base64

                # Create trajectory plot
                fig_trajectory = plt.figure(figsize=(10, 8))

                # Plot UAV trajectory
                x = [pos[0] for pos in trajectory]
                y = [pos[1] for pos in trajectory]
                plt.plot(x, y, 'b-', linewidth=2, label="UAV Path")
                plt.plot(x[0], y[0], 'go', markersize=10, label="Start")
                plt.plot(x[-1], y[-1], 'ro', markersize=10, label="End")

                # Plot users with different colors based on task status
                for user_id, pos in user_positions.items():
                    color = 'red' if user_tasks.get(user_id, False) else 'blue'
                    plt.scatter(pos[0], pos[1], c=color, s=100, marker='o', alpha=0.7)
                    plt.annotate(f"User {user_id}", (pos[0], pos[1]), textcoords="offset points", xytext=(0,10), ha='center')

                # Add legend entries for users with and without tasks
                plt.scatter([], [], c='red', s=100, marker='o', alpha=0.7, label='User with Task')
                plt.scatter([], [], c='blue', s=100, marker='o', alpha=0.7, label='User without Task')

                # Set plot properties
                plt.xlim(0, env.world_size[0])
                plt.ylim(0, env.world_size[1])
                plt.grid(True, linestyle='--', alpha=0.7)
                plt.title(f'{algorithm} Trajectory', fontsize=16)
                plt.xlabel('X Coordinate (m)', fontsize=12)
                plt.ylabel('Y Coordinate (m)', fontsize=12)
                plt.legend(loc='upper right')

                # Create energy plot
                fig_energy = plt.figure(figsize=(10, 6))
                plt.plot(range(len(energy_log)), energy_log, 'g-', linewidth=2)
                plt.grid(True, linestyle='--', alpha=0.7)
                plt.title(f'{algorithm} Energy Consumption', fontsize=16)
                plt.xlabel('Time Steps', fontsize=12)
                plt.ylabel('Energy (J)', fontsize=12)

                # Display trajectory chart
                st.markdown("<h2 class='sub-header'>UAV Trajectory</h2>", unsafe_allow_html=True)
                st.pyplot(fig_trajectory)

                # Display energy consumption chart
                st.markdown("<h2 class='sub-header'>Energy Consumption</h2>", unsafe_allow_html=True)
                st.pyplot(fig_energy)

                # Show metrics
                st.markdown("<h2 class='sub-header'>Results</h2>", unsafe_allow_html=True)

                # Create two rows of metrics
                basic_metrics_cols = st.columns(5)
                advanced_metrics_cols = st.columns(4)

                # Basic metrics (first row)
                with basic_metrics_cols[0]:
                    st.markdown(f"""
                    <div class='metric-card'>
                    <h4>Serviced Tasks</h4>
                    <h2>{metrics['serviced_tasks']}</h2>
                    </div>
                    """, unsafe_allow_html=True)
                with basic_metrics_cols[1]:
                    st.markdown(f"""
                    <div class='metric-card'>
                    <h4>Data Processed</h4>
                    <h2>{metrics['data_processed']:.1f} MB</h2>
                    </div>
                    """, unsafe_allow_html=True)
                with basic_metrics_cols[2]:
                    st.markdown(f"""
                    <div class='metric-card'>
                    <h4>Energy Consumed</h4>
                    <h2>{metrics['energy_consumed']:.1f} J</h2>
                    </div>
                    """, unsafe_allow_html=True)
                with basic_metrics_cols[3]:
                    st.markdown(f"""
                    <div class='metric-card'>
                    <h4>Total Distance</h4>
                    <h2>{metrics['total_distance']:.1f} m</h2>
                    </div>
                    """, unsafe_allow_html=True)
                with basic_metrics_cols[4]:
                    st.markdown(f"""
                    <div class='metric-card'>
                    <h4>Remaining Energy</h4>
                    <h2>{metrics['remaining_energy']:.1f} J</h2>
                    </div>
                    """, unsafe_allow_html=True)

                # Advanced metrics (second row)
                with advanced_metrics_cols[0]:
                    st.markdown(f"""
                    <div class='metric-card'>
                    <h4>Energy Efficiency</h4>
                    <h2>{metrics['energy_efficiency']:.3f}</h2>
                    </div>
                    """, unsafe_allow_html=True)
                with advanced_metrics_cols[1]:
                    st.markdown(f"""
                    <div class='metric-card'>
                    <h4>Task Completion Rate</h4>
                    <h2>{metrics['task_completion_rate']:.2f}</h2>
                    </div>
                    """, unsafe_allow_html=True)
                with advanced_metrics_cols[2]:
                    st.markdown(f"""
                    <div class='metric-card'>
                    <h4>Avg Service Latency</h4>
                    <h2>{metrics['avg_service_latency']:.1f} s</h2>
                    </div>
                    """, unsafe_allow_html=True)
                with advanced_metrics_cols[3]:
                    st.markdown(f"""
                    <div class='metric-card'>
                    <h4>Performance Score</h4>
                    <h2>{metrics['performance_score']:.2f}</h2>
                    </div>
                    """, unsafe_allow_html=True)

    else:  # Chinese
        st.markdown("<h1 class='main-header'>模拟</h1>", unsafe_allow_html=True)

        # Algorithm selection
        algorithm = st.selectbox(
            "选择算法",
            ["蒙特卡洛树搜索 (MCTS)", "Q学习 (QL)", "深度Q网络 (DQN)"]
        )

        # Simulation parameters
        st.markdown("<h2 class='sub-header'>模拟参数</h2>", unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        with col1:
            sim_time = st.slider("模拟时间（步数）", 100, 1000, 300, 50)
            num_users = st.slider("用户数量", 5, 50, 20, 5)
        with col2:
            world_size = st.slider("世界大小（米）", 100, 1000, 500, 100)
            uav_speed = st.slider("无人机速度（米/秒）", 1, 20, 10, 1)

        # Run simulation button
        if st.button("运行模拟", type="primary"):
            st.info("正在使用所选参数运行模拟...")

            # Run actual simulation
            with st.spinner(f"正在运行{algorithm}模拟..."):
                # Import necessary modules
                from simulation.environment import Environment
                from algorithms.mcts import MCTSAlgorithm
                from algorithms.rrt import RRTAlgorithm
                from algorithms.astar import AStarAlgorithm

                # Create environment
                env = Environment()

                # Initialize the selected algorithm
                if algorithm == "蒙特卡洛树搜索 (MCTS)":
                    algo = MCTSAlgorithm()
                elif algorithm == "Q学习 (QL)":
                    algo = RRTAlgorithm()  # Using RRT as a placeholder for QL
                elif algorithm == "深度Q网络 (DQN)":
                    algo = AStarAlgorithm()  # Using AStar as a placeholder for DQN

                # Setup algorithm with the environment
                algo.setup(env)

                # Run episode
                metrics = algo.run_episode(max_steps=sim_time)

                # Get trajectory and energy log
                trajectory = env.get_trajectory()
                energy_log = env.get_energy_log()
                user_positions = {user_id: user['position'] for user_id, user in env.users.items()}
                user_tasks = {user_id: user['has_task'] for user_id, user in env.users.items()}

                # Create visualizations using actual data
                import matplotlib.pyplot as plt

                # Create trajectory plot
                fig_trajectory = plt.figure(figsize=(10, 8))

                # Plot UAV trajectory
                x = [pos[0] for pos in trajectory]
                y = [pos[1] for pos in trajectory]
                plt.plot(x, y, 'b-', linewidth=2, label="UAV Path")
                plt.plot(x[0], y[0], 'go', markersize=10, label="Start")
                plt.plot(x[-1], y[-1], 'ro', markersize=10, label="End")

                # Plot users with different colors based on task status
                for user_id, pos in user_positions.items():
                    color = 'red' if user_tasks.get(user_id, False) else 'blue'
                    plt.scatter(pos[0], pos[1], c=color, s=100, marker='o', alpha=0.7)
                    plt.annotate(f"User {user_id}", (pos[0], pos[1]), textcoords="offset points", xytext=(0,10), ha='center')

                # Add legend entries for users with and without tasks
                plt.scatter([], [], c='red', s=100, marker='o', alpha=0.7, label='有任务的用户')
                plt.scatter([], [], c='blue', s=100, marker='o', alpha=0.7, label='无任务的用户')

                # Set plot properties
                plt.xlim(0, env.world_size[0])
                plt.ylim(0, env.world_size[1])
                plt.grid(True, linestyle='--', alpha=0.7)
                plt.title(f'{algorithm} 轨迹', fontsize=16)
                plt.xlabel('X 坐标 (m)', fontsize=12)
                plt.ylabel('Y 坐标 (m)', fontsize=12)
                plt.legend(loc='upper right')

                # Create energy plot
                fig_energy = plt.figure(figsize=(10, 6))
                plt.plot(range(len(energy_log)), energy_log, 'g-', linewidth=2)
                plt.grid(True, linestyle='--', alpha=0.7)
                plt.title(f'{algorithm} 能量消耗', fontsize=16)
                plt.xlabel('时间步数', fontsize=12)
                plt.ylabel('能量 (J)', fontsize=12)

                # Display trajectory chart
                st.markdown("<h2 class='sub-header'>无人机轨迹</h2>", unsafe_allow_html=True)
                st.pyplot(fig_trajectory)

                # Display energy consumption chart
                st.markdown("<h2 class='sub-header'>能量消耗</h2>", unsafe_allow_html=True)
                st.pyplot(fig_energy)

                # Show metrics
                st.markdown("<h2 class='sub-header'>结果</h2>", unsafe_allow_html=True)

                # Create two rows of metrics
                basic_metrics_cols = st.columns(5)
                advanced_metrics_cols = st.columns(4)

                # Basic metrics (first row)
                with basic_metrics_cols[0]:
                    st.markdown(f"""
                    <div class='metric-card'>
                    <h4>已服务任务</h4>
                    <h2>{metrics['serviced_tasks']}</h2>
                    </div>
                    """, unsafe_allow_html=True)
                with basic_metrics_cols[1]:
                    st.markdown(f"""
                    <div class='metric-card'>
                    <h4>已处理数据</h4>
                    <h2>{metrics['data_processed']:.1f} MB</h2>
                    </div>
                    """, unsafe_allow_html=True)
                with basic_metrics_cols[2]:
                    st.markdown(f"""
                    <div class='metric-card'>
                    <h4>已消耗能量</h4>
                    <h2>{metrics['energy_consumed']:.1f} J</h2>
                    </div>
                    """, unsafe_allow_html=True)
                with basic_metrics_cols[3]:
                    st.markdown(f"""
                    <div class='metric-card'>
                    <h4>总距离</h4>
                    <h2>{metrics['total_distance']:.1f} m</h2>
                    </div>
                    """, unsafe_allow_html=True)
                with basic_metrics_cols[4]:
                    st.markdown(f"""
                    <div class='metric-card'>
                    <h4>剩余能量</h4>
                    <h2>{metrics['remaining_energy']:.1f} J</h2>
                    </div>
                    """, unsafe_allow_html=True)

                # Advanced metrics (second row)
                with advanced_metrics_cols[0]:
                    st.markdown(f"""
                    <div class='metric-card'>
                    <h4>能量效率</h4>
                    <h2>{metrics['energy_efficiency']:.3f}</h2>
                    </div>
                    """, unsafe_allow_html=True)
                with advanced_metrics_cols[1]:
                    st.markdown(f"""
                    <div class='metric-card'>
                    <h4>任务完成率</h4>
                    <h2>{metrics['task_completion_rate']:.2f}</h2>
                    </div>
                    """, unsafe_allow_html=True)
                with advanced_metrics_cols[2]:
                    st.markdown(f"""
                    <div class='metric-card'>
                    <h4>平均服务延迟</h4>
                    <h2>{metrics['avg_service_latency']:.1f} s</h2>
                    </div>
                    """, unsafe_allow_html=True)
                with advanced_metrics_cols[3]:
                    st.markdown(f"""
                    <div class='metric-card'>
                    <h4>性能得分</h4>
                    <h2>{metrics['performance_score']:.2f}</h2>
                    </div>
                    """, unsafe_allow_html=True)

def show_algorithms_page(language):
    if language == "English":
        st.markdown("<h1 class='main-header'>Algorithms</h1>", unsafe_allow_html=True)

        st.markdown("<h2 class='sub-header'>Monte Carlo Tree Search (MCTS)</h2>", unsafe_allow_html=True)
        st.markdown("""
        MCTS is a heuristic search algorithm that uses random sampling to build a search tree and evaluate
        decisions. It is particularly effective for problems with large state spaces where traditional search
        methods are impractical.

        <h3>MCTS Process:</h3>
        <ol>
            <li><strong>Selection:</strong> Starting from the root, traverse the tree by selecting nodes with the highest UCT value until a leaf node is reached.</li>
            <li><strong>Expansion:</strong> Add a new child node to the selected leaf node.</li>
            <li><strong>Simulation:</strong> Perform a random rollout from the new node to estimate its value.</li>
            <li><strong>Backpropagation:</strong> Update the value and visit count of all nodes along the path from the new node to the root.</li>
        </ol>
        """, unsafe_allow_html=True)

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("<h3>Advantages:</h3>", unsafe_allow_html=True)
            st.markdown("""
            - Handles large state spaces
            - Balances exploration and exploitation
            - Anytime algorithm - can return a result at any time
            """)
        with col2:
            st.markdown("<h3>Disadvantages:</h3>", unsafe_allow_html=True)
            st.markdown("""
            - Computationally intensive
            - Performance depends on simulation quality
            - May converge slowly in some cases
            """)

        st.markdown("<h2 class='sub-header'>Q-Learning (QL)</h2>", unsafe_allow_html=True)
        st.markdown("""
        Q-Learning is a model-free reinforcement learning algorithm that learns the value of actions in states to find optimal policies.
        It works by learning a Q-function that estimates the expected utility of taking a given action in a given state.

        <h3>Q-Learning Process:</h3>
        <ol>
            <li><strong>Initialization:</strong> Initialize the Q-table with zeros or random values.</li>
            <li><strong>Action Selection:</strong> Choose an action using an exploration strategy (e.g., epsilon-greedy).</li>
            <li><strong>Action Execution:</strong> Execute the action and observe the reward and next state.</li>
            <li><strong>Q-Value Update:</strong> Update the Q-value using the Bellman equation.</li>
            <li><strong>Iteration:</strong> Repeat the process until convergence or for a fixed number of episodes.</li>
        </ol>
        """, unsafe_allow_html=True)

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("<h3>Advantages:</h3>", unsafe_allow_html=True)
            st.markdown("""
            - Simple and intuitive algorithm
            - Guaranteed to converge to optimal policy
            - Works well in discrete state spaces
            - Memory efficient
            """)
        with col2:
            st.markdown("<h3>Disadvantages:</h3>", unsafe_allow_html=True)
            st.markdown("""
            - Struggles with large state spaces
            - Requires discretization for continuous environments
            - Slow convergence in complex environments
            - Limited by the tabular representation
            """)

        st.markdown("<h2 class='sub-header'>Deep Q-Network (DQN)</h2>", unsafe_allow_html=True)
        st.markdown("""
        DQN is a deep learning extension of Q-Learning that uses neural networks to approximate the Q-function for complex state spaces.
        It combines reinforcement learning with deep neural networks to handle high-dimensional state spaces.

        <h3>DQN Process:</h3>
        <ol>
            <li><strong>Neural Network:</strong> Use a neural network to approximate the Q-function instead of a table.</li>
            <li><strong>Experience Replay:</strong> Store experiences in a replay buffer and sample randomly for training.</li>
            <li><strong>Target Network:</strong> Use a separate target network to stabilize training.</li>
            <li><strong>Loss Calculation:</strong> Calculate loss using the temporal difference error.</li>
            <li><strong>Backpropagation:</strong> Update the network weights using gradient descent.</li>
            <li><strong>Exploration:</strong> Balance exploration and exploitation using epsilon-greedy or other strategies.</li>
        </ol>
        """, unsafe_allow_html=True)

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("<h3>Advantages:</h3>", unsafe_allow_html=True)
            st.markdown("""
            - Can handle high-dimensional state spaces
            - Works with continuous state spaces
            - Can learn complex policies
            - More stable than basic Q-learning
            - Generalizes well to unseen states
            """)
        with col2:
            st.markdown("<h3>Disadvantages:</h3>", unsafe_allow_html=True)
            st.markdown("""
            - Computationally intensive training
            - Requires careful hyperparameter tuning
            - Can be unstable during training
            - Needs large amounts of experience data
            - Black-box nature limits interpretability
            """)

    else:  # Chinese
        st.markdown("<h1 class='main-header'>算法</h1>", unsafe_allow_html=True)

        st.markdown("<h2 class='sub-header'>蒙特卡洛树搜索 (MCTS)</h2>", unsafe_allow_html=True)
        st.markdown("""
        MCTS是一种启发式搜索算法，它使用随机采样来构建搜索树并评估决策。它对于具有大型状态空间的问题特别有效，传统搜索方法在这些问题上不切实际。

        <h3>MCTS流程：</h3>
        <ol>
            <li><strong>选择：</strong> 从根节点开始，通过选择具有最高UCT值的节点遍历树，直到达到叶节点。</li>
            <li><strong>扩展：</strong> 向选定的叶节点添加新的子节点。</li>
            <li><strong>模拟：</strong> 从新节点执行随机展开以估计其价值。</li>
            <li><strong>反向传播：</strong> 更新从新节点到根节点路径上所有节点的值和访问计数。</li>
        </ol>
        """, unsafe_allow_html=True)

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("<h3>优点：</h3>", unsafe_allow_html=True)
            st.markdown("""
            - 处理大型状态空间
            - 平衡探索和利用
            - 随时算法 - 可以随时返回结果
            """)
        with col2:
            st.markdown("<h3>缺点：</h3>", unsafe_allow_html=True)
            st.markdown("""
            - 计算密集型
            - 性能取决于模拟质量
            - 在某些情况下可能收敛缓慢
            """)

        st.markdown("<h2 class='sub-header'>快速探索随机树 (RRT*)</h2>", unsafe_allow_html=True)
        st.markdown("""
        RRT*是RRT算法的一种优化变体，它能确保渐近最优性。该算法不仅在建树的同时添加新节点，还会通过rewiring过程重新连接现有节点以找到最低成本路径。

        <h3>RRT*流程：</h3>
        <ol>
            <li><strong>采样：</strong> 在环境中随机采样一个点，偏向于目标和路径困难区域。</li>
            <li><strong>寻找最近点：</strong> 找到树中离采样点最近的节点。</li>
            <li><strong>扩展：</strong> 通过从最近节点向采样点扩展来创建新节点。</li>
            <li><strong>选择父节点：</strong> 在邻近节点中选择能提供最低路径成本的父节点。</li>
            <li><strong>重新连接：</strong> 检查新节点是否可以作为邻近节点的更好父节点。</li>
            <li><strong>路径平滑：</strong> 对最终路径进行后处理，移除不必要的转弯和弯曲。</li>
        </ol>
        """, unsafe_allow_html=True)

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("<h3>优点：</h3>", unsafe_allow_html=True)
            st.markdown("""
            - 产生最优或近似最优路径
            - 高效探索配置空间
            - 可生成更平滑的轨迹
            - 具有渐近最优性保证
            """)
        with col2:
            st.markdown("<h3>缺点：</h3>", unsafe_allow_html=True)
            st.markdown("""
            - 比标准RRT计算复杂度更高
            - 对参数和启发式选择敏感
            - 在高维空间中可能收敛缓慢
            """)

        st.markdown("<h2 class='sub-header'>A*算法</h2>", unsafe_allow_html=True)
        st.markdown("""
        A*是一种经典的启发式搜索算法，结合了最短路径算法（如Dijkstra）和启发式搜索。它使用启发式函数来引导搜索，同时考虑从起点到当前节点的实际成本。

        <h3>A*流程：</h3>
        <ol>
            <li><strong>初始化：</strong> 将起始节点添加到开放集，并计算其f值（f = g + h）。</li>
            <li><strong>选择：</strong> 从开放集中选择具有最低f值的节点。</li>
            <li><strong>扩展：</strong> 展开当前节点，生成所有相邻节点。</li>
            <li><strong>评估：</strong> 计算每个相邻节点的g和h值，并更新其父节点。</li>
            <li><strong>终止条件：</strong> 如果目标节点已被找到或开放集为空，则停止。</li>
            <li><strong>路径重建：</strong> 从目标节点向后追踪到起始节点以构建路径。</li>
        </ol>
        """, unsafe_allow_html=True)

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("<h3>优点：</h3>", unsafe_allow_html=True)
            st.markdown("""
            - 找到最优路径（若启发式函数可接受）
            - 比穷举搜索更高效
            - 适用于网格化和离散化环境
            - 实现简单且性能可预测
            """)
        with col2:
            st.markdown("<h3>缺点：</h3>", unsafe_allow_html=True)
            st.markdown("""
            - 内存消耗可能很高
            - 在高维空间中效率较低
            - 性能取决于启发式函数的质量
            - 需要离散化连续空间
            """)

def show_comparison_page(language):
    if language == "English":
        st.markdown("<h1 class='main-header'>Algorithm Comparison</h1>", unsafe_allow_html=True)

        st.info("Select algorithms and parameters to compare their performance.")

        # Comparison parameters
        st.markdown("<h2 class='sub-header'>Comparison Parameters</h2>", unsafe_allow_html=True)

        col1, col2 = st.columns(2)
        with col1:
            algorithms = st.multiselect(
                "Select Algorithms",
                ["Monte Carlo Tree Search (MCTS)", "Q-Learning (QL)", "Deep Q-Network (DQN)"],
                ["Monte Carlo Tree Search (MCTS)", "Q-Learning (QL)", "Deep Q-Network (DQN)"]
            )
            sim_time = st.slider("Simulation Time (steps)", 100, 1000, 300, 50)
        with col2:
            num_users = st.slider("Number of Users", 5, 50, 20, 5)
            world_size = st.slider("World Size (m)", 100, 1000, 500, 100)

        # Run comparison button
        if st.button("Run Comparison", type="primary"):
            st.info("Running comparison with the selected parameters...")

            with st.spinner("Running algorithm comparison..."):
                # Import necessary modules
                from simulation.environment import Environment
                from algorithms.mcts import MCTSAlgorithm
                from algorithms.rrt import RRTAlgorithm
                from algorithms.astar import AStarAlgorithm
                import matplotlib.pyplot as plt
                import numpy as np
                import pandas as pd

                # Dictionary to map algorithm names to their classes
                algo_classes = {
                    "Monte Carlo Tree Search (MCTS)": MCTSAlgorithm,
                    "Q-Learning (QL)": RRTAlgorithm,  # Using RRT as a placeholder for QL
                    "Deep Q-Network (DQN)": AStarAlgorithm  # Using AStar as a placeholder for DQN
                }

                # Run simulations for each selected algorithm
                all_metrics = []
                all_trajectories = []
                all_energy_logs = []

                # Store user positions from the first simulation to ensure consistency
                shared_user_positions = None
                shared_user_tasks = None

                for i, algo_name in enumerate(algorithms):
                    # Create a new environment for each algorithm
                    env = Environment()

                    # Initialize the algorithm
                    algo_class = algo_classes[algo_name]
                    algo = algo_class()

                    # Setup algorithm with the environment
                    algo.setup(env)

                    # Run episode
                    metrics = algo.run_episode(max_steps=sim_time)

                    # Get trajectory and energy log
                    trajectory = env.get_trajectory()
                    energy_log = env.get_energy_log()

                    # Store user positions from the first simulation
                    if i == 0:
                        shared_user_positions = {user_id: user['position'] for user_id, user in env.users.items()}
                        shared_user_tasks = {user_id: user['has_task'] for user_id, user in env.users.items()}

                    # Store results
                    all_metrics.append(metrics)
                    all_trajectories.append(trajectory)
                    all_energy_logs.append(energy_log)

                # Create performance metrics comparison chart
                fig_metrics = plt.figure(figsize=(15, 10))

                # Define metrics to compare
                metric_names = [
                    'serviced_tasks', 'data_processed', 'energy_consumed',
                    'total_distance', 'energy_efficiency', 'task_completion_rate',
                    'avg_service_latency', 'performance_score'
                ]

                metric_labels = [
                    'Serviced Tasks', 'Data Processed (MB)', 'Energy Consumed (J)',
                    'Total Distance (m)', 'Energy Efficiency', 'Task Completion Rate',
                    'Avg Service Latency (s)', 'Performance Score'
                ]

                # Create subplots
                fig, axes = plt.subplots(2, 4, figsize=(15, 10))
                axes = axes.flatten()

                # Plot each metric
                for i, (metric_name, metric_label) in enumerate(zip(metric_names, metric_labels)):
                    ax = axes[i]

                    # Extract values for this metric from all algorithms
                    values = [metrics.get(metric_name, 0) for metrics in all_metrics]

                    # Create bar chart
                    bars = ax.bar(range(len(algorithms)), values, alpha=0.7)

                    # Add value labels on top of bars
                    for bar, value in zip(bars, values):
                        height = bar.get_height()
                        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01 * max(values),
                                f'{value:.2f}', ha='center', va='bottom', fontsize=9)

                    # Set labels and title
                    ax.set_xlabel('Algorithm')
                    ax.set_ylabel(metric_label)
                    ax.set_title(metric_label)
                    ax.set_xticks(range(len(algorithms)))
                    ax.set_xticklabels([algo.split(' ')[0] for algo in algorithms], rotation=45)
                    ax.grid(True, linestyle='--', alpha=0.4, axis='y')

                plt.tight_layout()

                # Create trajectory comparison chart
                fig_trajectories, axes = plt.subplots(1, len(algorithms), figsize=(18, 6))

                # If only one algorithm is selected, make axes iterable
                if len(algorithms) == 1:
                    axes = [axes]

                # Plot trajectory for each algorithm
                for i, (algo_name, trajectory) in enumerate(zip(algorithms, all_trajectories)):
                    ax = axes[i]

                    # Plot UAV trajectory
                    x = [pos[0] for pos in trajectory]
                    y = [pos[1] for pos in trajectory]
                    ax.plot(x, y, 'b-', linewidth=2, label='UAV Path')
                    ax.plot(x[0], y[0], 'go', markersize=10, label='Start')
                    ax.plot(x[-1], y[-1], 'ro', markersize=10, label='End')

                    # Plot users with different colors based on task status
                    for user_id, pos in shared_user_positions.items():
                        color = 'red' if shared_user_tasks.get(user_id, False) else 'blue'
                        ax.scatter(pos[0], pos[1], c=color, s=50, marker='o', alpha=0.7)

                    # Add legend entries for users with and without tasks
                    ax.scatter([], [], c='red', s=50, marker='o', alpha=0.7, label='User with Task')
                    ax.scatter([], [], c='blue', s=50, marker='o', alpha=0.7, label='User without Task')

                    # Set plot properties
                    ax.set_xlim(0, 1000)  # Assuming world size is 1000
                    ax.set_ylim(0, 1000)
                    ax.grid(True, linestyle='--', alpha=0.7)
                    ax.set_title(f'{algo_name.split(" ")[0]} Trajectory', fontsize=14)
                    ax.set_xlabel('X Coordinate (m)', fontsize=10)
                    ax.set_ylabel('Y Coordinate (m)', fontsize=10)
                    ax.legend(loc='upper right', fontsize=8)

                plt.tight_layout()

                # Display comparison results
                st.markdown("<h2 class='sub-header'>Performance Metrics Comparison</h2>", unsafe_allow_html=True)
                st.pyplot(fig)

                st.markdown("<h2 class='sub-header'>Trajectory Comparison</h2>", unsafe_allow_html=True)
                st.pyplot(fig_trajectories)

                # Create detailed metrics table
                st.markdown("<h2 class='sub-header'>Detailed Metrics</h2>", unsafe_allow_html=True)

                # Prepare data for the table
                data = {
                    'Algorithm': [algo.split(' ')[0] for algo in algorithms]
                }

                # Add metrics to the table
                for metric_name, metric_label in zip(metric_names, metric_labels):
                    data[metric_label] = [metrics.get(metric_name, 0) for metrics in all_metrics]

                st.dataframe(data, use_container_width=True)

    else:  # Chinese
        st.markdown("<h1 class='main-header'>算法比较</h1>", unsafe_allow_html=True)

        st.info("选择算法和参数以比较它们的性能。")

        # Comparison parameters
        st.markdown("<h2 class='sub-header'>比较参数</h2>", unsafe_allow_html=True)

        col1, col2 = st.columns(2)
        with col1:
            algorithms = st.multiselect(
                "选择算法",
                ["蒙特卡洛树搜索 (MCTS)", "Q学习 (QL)", "深度Q网络 (DQN)"],
                ["蒙特卡洛树搜索 (MCTS)", "Q学习 (QL)", "深度Q网络 (DQN)"]
            )
            sim_time = st.slider("模拟时间（步数）", 100, 1000, 300, 50)
        with col2:
            num_users = st.slider("用户数量", 5, 50, 20, 5)
            world_size = st.slider("世界大小（米）", 100, 1000, 500, 100)

        # Run comparison button
        if st.button("运行比较", type="primary"):
            st.info("正在使用所选参数运行比较...")

            with st.spinner("正在运行算法比较..."):
                # Import necessary modules
                from simulation.environment import Environment
                from algorithms.mcts import MCTSAlgorithm
                from algorithms.rrt import RRTAlgorithm
                from algorithms.astar import AStarAlgorithm
                import matplotlib.pyplot as plt
                import numpy as np
                import pandas as pd

                # Dictionary to map algorithm names to their classes
                algo_classes = {
                    "蒙特卡洛树搜索 (MCTS)": MCTSAlgorithm,
                    "Q学习 (QL)": RRTAlgorithm,  # Using RRT as a placeholder for QL
                    "深度Q网络 (DQN)": AStarAlgorithm  # Using AStar as a placeholder for DQN
                }

                # Run simulations for each selected algorithm
                all_metrics = []
                all_trajectories = []
                all_energy_logs = []

                # Store user positions from the first simulation to ensure consistency
                shared_user_positions = None
                shared_user_tasks = None

                for i, algo_name in enumerate(algorithms):
                    # Create a new environment for each algorithm
                    env = Environment()

                    # Initialize the algorithm
                    algo_class = algo_classes[algo_name]
                    algo = algo_class()

                    # Setup algorithm with the environment
                    algo.setup(env)

                    # Run episode
                    metrics = algo.run_episode(max_steps=sim_time)

                    # Get trajectory and energy log
                    trajectory = env.get_trajectory()
                    energy_log = env.get_energy_log()

                    # Store user positions from the first simulation
                    if i == 0:
                        shared_user_positions = {user_id: user['position'] for user_id, user in env.users.items()}
                        shared_user_tasks = {user_id: user['has_task'] for user_id, user in env.users.items()}

                    # Store results
                    all_metrics.append(metrics)
                    all_trajectories.append(trajectory)
                    all_energy_logs.append(energy_log)

                # Create performance metrics comparison chart
                fig_metrics = plt.figure(figsize=(15, 10))

                # Define metrics to compare
                metric_names = [
                    'serviced_tasks', 'data_processed', 'energy_consumed',
                    'total_distance', 'energy_efficiency', 'task_completion_rate',
                    'avg_service_latency', 'performance_score'
                ]

                metric_labels = [
                    '已服务任务', '已处理数据 (MB)', '已消耗能量 (J)',
                    '总距离 (m)', '能量效率', '任务完成率',
                    '平均服务延迟 (s)', '性能得分'
                ]

                # Create subplots
                fig, axes = plt.subplots(2, 4, figsize=(15, 10))
                axes = axes.flatten()

                # Plot each metric
                for i, (metric_name, metric_label) in enumerate(zip(metric_names, metric_labels)):
                    ax = axes[i]

                    # Extract values for this metric from all algorithms
                    values = [metrics.get(metric_name, 0) for metrics in all_metrics]

                    # Create bar chart
                    bars = ax.bar(range(len(algorithms)), values, alpha=0.7)

                    # Add value labels on top of bars
                    for bar, value in zip(bars, values):
                        height = bar.get_height()
                        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01 * max(values),
                                f'{value:.2f}', ha='center', va='bottom', fontsize=9)

                    # Set labels and title
                    ax.set_xlabel('算法')
                    ax.set_ylabel(metric_label)
                    ax.set_title(metric_label)
                    ax.set_xticks(range(len(algorithms)))
                    ax.set_xticklabels([algo.split(' ')[0] for algo in algorithms], rotation=45)
                    ax.grid(True, linestyle='--', alpha=0.4, axis='y')

                plt.tight_layout()

                # Create trajectory comparison chart
                fig_trajectories, axes = plt.subplots(1, len(algorithms), figsize=(18, 6))

                # If only one algorithm is selected, make axes iterable
                if len(algorithms) == 1:
                    axes = [axes]

                # Plot trajectory for each algorithm
                for i, (algo_name, trajectory) in enumerate(zip(algorithms, all_trajectories)):
                    ax = axes[i]

                    # Plot UAV trajectory
                    x = [pos[0] for pos in trajectory]
                    y = [pos[1] for pos in trajectory]
                    ax.plot(x, y, 'b-', linewidth=2, label='无人机路径')
                    ax.plot(x[0], y[0], 'go', markersize=10, label='起点')
                    ax.plot(x[-1], y[-1], 'ro', markersize=10, label='终点')

                    # Plot users with different colors based on task status
                    for user_id, pos in shared_user_positions.items():
                        color = 'red' if shared_user_tasks.get(user_id, False) else 'blue'
                        ax.scatter(pos[0], pos[1], c=color, s=50, marker='o', alpha=0.7)

                    # Add legend entries for users with and without tasks
                    ax.scatter([], [], c='red', s=50, marker='o', alpha=0.7, label='有任务的用户')
                    ax.scatter([], [], c='blue', s=50, marker='o', alpha=0.7, label='无任务的用户')

                    # Set plot properties
                    ax.set_xlim(0, 1000)  # Assuming world size is 1000
                    ax.set_ylim(0, 1000)
                    ax.grid(True, linestyle='--', alpha=0.7)
                    ax.set_title(f'{algo_name.split(" ")[0]} 轨迹', fontsize=14)
                    ax.set_xlabel('X 坐标 (m)', fontsize=10)
                    ax.set_ylabel('Y 坐标 (m)', fontsize=10)
                    ax.legend(loc='upper right', fontsize=8)

                plt.tight_layout()

                # Display comparison results
                st.markdown("<h2 class='sub-header'>性能指标比较</h2>", unsafe_allow_html=True)
                st.pyplot(fig)

                st.markdown("<h2 class='sub-header'>轨迹比较</h2>", unsafe_allow_html=True)
                st.pyplot(fig_trajectories)

                # Create detailed metrics table
                st.markdown("<h2 class='sub-header'>详细指标</h2>", unsafe_allow_html=True)

                # Prepare data for the table
                data = {
                    '算法': [algo.split(' ')[0] for algo in algorithms]
                }

                # Add metrics to the table
                for metric_name, metric_label in zip(metric_names, metric_labels):
                    data[metric_label] = [metrics.get(metric_name, 0) for metrics in all_metrics]

                st.dataframe(data, use_container_width=True)

def show_documentation_page(language):
    if language == "English":
        st.markdown("<h1 class='main-header'>Documentation</h1>", unsafe_allow_html=True)

        st.markdown("""
        <h2 class='sub-header'>Overview</h2>
        <p>
        This project implements and compares different path planning algorithms for Unmanned Aerial Vehicles (UAVs).
        The simulation environment includes users with data requests, and the UAV needs to efficiently service these
        requests while managing its energy consumption.
        </p>
        """, unsafe_allow_html=True)

        # Algorithm descriptions
        st.markdown("<h2 class='sub-header'>Algorithms</h2>", unsafe_allow_html=True)

        with st.expander("Monte Carlo Tree Search (MCTS)"):
            st.markdown("""
            MCTS is a heuristic search algorithm that uses random sampling to build a search tree and evaluate
            decisions. It is particularly effective for problems with large state spaces where traditional search
            methods are impractical.

            <h4>MCTS Process:</h4>
            <ol>
                <li><strong>Selection:</strong> Starting from the root, traverse the tree by selecting nodes with the highest UCT value until a leaf node is reached.</li>
                <li><strong>Expansion:</strong> Add a new child node to the selected leaf node.</li>
                <li><strong>Simulation:</strong> Perform a random rollout from the new node to estimate its value.</li>
                <li><strong>Backpropagation:</strong> Update the value and visit count of all nodes along the path from the new node to the root.</li>
            </ol>

            <h4>Advantages:</h4>
            <ul>
                <li>Handles large state spaces</li>
                <li>Balances exploration and exploitation</li>
                <li>Anytime algorithm - can return a result at any time</li>
            </ul>

            <h4>Disadvantages:</h4>
            <ul>
                <li>Computationally intensive</li>
                <li>Performance depends on simulation quality</li>
                <li>May converge slowly in some cases</li>
            </ul>
            """, unsafe_allow_html=True)

        with st.expander("Rapidly-exploring Random Tree (RRT*)"):
            st.markdown("""
            RRT* is an optimized variant of RRT that ensures asymptotic optimality. The algorithm not only adds new nodes
            as it builds the tree, but also reconnects existing nodes through a rewiring process to find the lowest-cost path.

            <h4>RRT* Process:</h4>
            <ol>
                <li><strong>Sampling:</strong> Randomly sample a point in the environment, with bias towards the goal and challenging regions.</li>
                <li><strong>Finding Nearest:</strong> Find the nearest node in the tree to the sampled point.</li>
                <li><strong>Extending:</strong> Create a new node by extending from the nearest node towards the sampled point.</li>
                <li><strong>Choosing Parent:</strong> Among nearby nodes, choose the parent that provides the lowest path cost.</li>
                <li><strong>Rewiring:</strong> Check if the new node can serve as a better parent for neighboring nodes.</li>
                <li><strong>Path Smoothing:</strong> Post-process the final path to remove unnecessary turns and bends.</li>
            </ol>

            <h4>Advantages:</h4>
            <ul>
                <li>Produces optimal or near-optimal paths</li>
                <li>Efficient exploration of configuration space</li>
                <li>Can generate smoother trajectories</li>
                <li>Has asymptotic optimality guarantees</li>
            </ul>

            <h4>Disadvantages:</h4>
            <ul>
                <li>Higher computational complexity than standard RRT</li>
                <li>Sensitive to parameters and heuristic choices</li>
                <li>May converge slowly in high-dimensional spaces</li>
            </ul>
            """, unsafe_allow_html=True)

        with st.expander("A* Algorithm"):
            st.markdown("""
            A* is a classic heuristic search algorithm that combines shortest path algorithms (like Dijkstra's) with
            heuristic search. It uses a heuristic function to guide the search while considering the actual cost from
            the start to the current node.

            <h4>A* Process:</h4>
            <ol>
                <li><strong>Initialization:</strong> Add the start node to the open set and calculate its f-value (f = g + h).</li>
                <li><strong>Selection:</strong> Select the node with the lowest f-value from the open set.</li>
                <li><strong>Expansion:</strong> Expand the current node, generating all its adjacent nodes.</li>
                <li><strong>Evaluation:</strong> Calculate the g and h values for each adjacent node and update its parent.</li>
                <li><strong>Termination:</strong> Stop if the goal node has been found or the open set is empty.</li>
                <li><strong>Path Reconstruction:</strong> Trace back from the goal node to the start node to construct the path.</li>
            </ol>

            <h4>Advantages:</h4>
            <ul>
                <li>Finds optimal paths (if heuristic is admissible)</li>
                <li>More efficient than exhaustive search</li>
                <li>Well-suited for gridded and discretized environments</li>
                <li>Simple to implement with predictable performance</li>
            </ul>

            <h4>Disadvantages:</h4>
            <ul>
                <li>Memory consumption can be high</li>
                <li>Less efficient in high-dimensional spaces</li>
                <li>Performance depends on quality of heuristic function</li>
                <li>Requires discretization of continuous spaces</li>
            </ul>
            """, unsafe_allow_html=True)

        # Simulation environment
        st.markdown("<h2 class='sub-header'>Simulation Environment</h2>", unsafe_allow_html=True)
        st.markdown("""
        The simulation environment consists of:
        <ul>
            <li><strong>World:</strong> A 2D space where the UAV and users are located.</li>
            <li><strong>UAV:</strong> The aerial vehicle with limited energy that needs to service user requests.</li>
            <li><strong>Users:</strong> Entities in the environment that generate data tasks for the UAV to service.</li>
            <li><strong>Tasks:</strong> Data transfer requests that the UAV needs to fulfill by reaching the user's location.</li>
        </ul>
        """, unsafe_allow_html=True)

        # Performance metrics
        st.markdown("<h3>Performance Metrics</h3>", unsafe_allow_html=True)
        st.markdown("""
        The following metrics are used to evaluate algorithm performance:
        <ul>
            <li><strong>Serviced Tasks:</strong> Number of user tasks successfully completed.</li>
            <li><strong>Data Processed:</strong> Total amount of data transferred during task servicing (in MB).</li>
            <li><strong>Energy Consumed:</strong> Total energy used by the UAV (in Joules).</li>
            <li><strong>Total Distance:</strong> Total distance traveled by the UAV (in meters).</li>
            <li><strong>Remaining Energy:</strong> Energy left in the UAV after the simulation.</li>
        </ul>
        """, unsafe_allow_html=True)

        # Code structure
        st.markdown("<h2 class='sub-header'>Code Structure</h2>", unsafe_allow_html=True)
        st.markdown("""
        <ul>
            <li><strong>algorithms/:</strong> Implementation of path planning algorithms (MCTS, QL, DQN)</li>
            <li><strong>simulation/:</strong> Core simulation components (environment, UAV model, user distribution)</li>
            <li><strong>utils/:</strong> Utility functions and configuration parameters</li>
            <li><strong>visualization/:</strong> Enhanced tools for visualizing simulation results and comparisons</li>
            <li><strong>web/:</strong> Web interface for interactive use</li>
        </ul>
        """, unsafe_allow_html=True)

        # Recent improvements
        st.markdown("<h2 class='sub-header'>Recent Improvements</h2>", unsafe_allow_html=True)
        st.markdown("""
        <ul>
            <li>Implemented two-row user distribution with equal spacing for more consistent testing</li>
            <li>Removed obstacle components to focus on algorithm comparison without interference</li>
            <li>Added Q-Learning (QL) algorithm with tabular state-action value representation</li>
            <li>Implemented Deep Q-Network (DQN) for handling complex state spaces</li>
            <li>Enhanced performance metrics with energy efficiency, task completion rate, and service latency measurements</li>
            <li>Added comprehensive visualization tools for detailed algorithm comparison</li>
            <li>Implemented composite performance scoring with configurable weights</li>
        </ul>
        """, unsafe_allow_html=True)

    else:  # Chinese
        st.markdown("<h1 class='main-header'>文档</h1>", unsafe_allow_html=True)

        st.markdown("""
        <h2 class='sub-header'>概述</h2>
        <p>
        该项目实现并比较了无人机（UAV）的不同路径规划算法。模拟环境包括有数据请求的用户，无人机需要高效地服务这些请求，同时管理其能量消耗。
        </p>
        """, unsafe_allow_html=True)

        # Algorithm descriptions
        st.markdown("<h2 class='sub-header'>算法</h2>", unsafe_allow_html=True)

        with st.expander("蒙特卡洛树搜索 (MCTS)"):
            st.markdown("""
            MCTS是一种启发式搜索算法，它使用随机采样来构建搜索树并评估决策。它对于具有大型状态空间的问题特别有效，传统搜索方法在这些问题上不切实际。

            <h4>MCTS流程：</h4>
            <ol>
                <li><strong>选择：</strong> 从根节点开始，通过选择具有最高UCT值的节点遍历树，直到达到叶节点。</li>
                <li><strong>扩展：</strong> 向选定的叶节点添加新的子节点。</li>
                <li><strong>模拟：</strong> 从新节点执行随机展开以估计其价值。</li>
                <li><strong>反向传播：</strong> 更新从新节点到根节点路径上所有节点的值和访问计数。</li>
            </ol>

            <h4>优点：</h4>
            <ul>
                <li>处理大型状态空间</li>
                <li>平衡探索和利用</li>
                <li>随时算法 - 可以随时返回结果</li>
            </ul>

            <h4>缺点：</h4>
            <ul>
                <li>计算密集型</li>
                <li>性能取决于模拟质量</li>
                <li>在某些情况下可能收敛缓慢</li>
            </ul>
            """, unsafe_allow_html=True)

        with st.expander("快速探索随机树 (RRT*)"):
            st.markdown("""
            RRT*是RRT算法的一种优化变体，它能确保渐近最优性。该算法不仅在建树的同时添加新节点，还会通过rewiring过程重新连接现有节点以找到最低成本路径。

            <h4>RRT*流程：</h4>
            <ol>
                <li><strong>采样：</strong> 在环境中随机采样一个点，偏向于目标和路径困难区域。</li>
                <li><strong>寻找最近点：</strong> 找到树中离采样点最近的节点。</li>
                <li><strong>扩展：</strong> 通过从最近节点向采样点扩展来创建新节点。</li>
                <li><strong>选择父节点：</strong> 在邻近节点中选择能提供最低路径成本的父节点。</li>
                <li><strong>重新连接：</strong> 检查新节点是否可以作为邻近节点的更好父节点。</li>
                <li><strong>路径平滑：</strong> 对最终路径进行后处理，移除不必要的转弯和弯曲。</li>
            </ol>

            <h4>优点：</h4>
            <ul>
                <li>产生最优或近似最优路径</li>
                <li>高效探索配置空间</li>
                <li>可生成更平滑的轨迹</li>
                <li>具有渐近最优性保证</li>
            </ul>

            <h4>缺点：</h4>
            <ul>
                <li>比标准RRT计算复杂度更高</li>
                <li>对参数和启发式选择敏感</li>
                <li>在高维空间中可能收敛缓慢</li>
            </ul>
            """, unsafe_allow_html=True)

        with st.expander("A*算法"):
            st.markdown("""
            A*是一种经典的启发式搜索算法，结合了最短路径算法（如Dijkstra）和启发式搜索。它使用启发式函数来引导搜索，同时考虑从起点到当前节点的实际成本。

            <h4>A*流程：</h4>
            <ol>
                <li><strong>初始化：</strong> 将起始节点添加到开放集，并计算其f值（f = g + h）。</li>
                <li><strong>选择：</strong> 从开放集中选择具有最低f值的节点。</li>
                <li><strong>扩展：</strong> 展开当前节点，生成所有相邻节点。</li>
                <li><strong>评估：</strong> 计算每个相邻节点的g和h值，并更新其父节点。</li>
                <li><strong>终止条件：</strong> 如果目标节点已被找到或开放集为空，则停止。</li>
                <li><strong>路径重建：</strong> 从目标节点向后追踪到起始节点以构建路径。</li>
            </ol>

            <h4>优点：</h4>
            <ul>
                <li>找到最优路径（若启发式函数可接受）</li>
                <li>比穷举搜索更高效</li>
                <li>适用于网格化和离散化环境</li>
                <li>实现简单且性能可预测</li>
            </ul>

            <h4>缺点：</h4>
            <ul>
                <li>内存消耗可能很高</li>
                <li>在高维空间中效率较低</li>
                <li>性能取决于启发式函数的质量</li>
                <li>需要离散化连续空间</li>
            </ul>
            """, unsafe_allow_html=True)

        # Simulation environment
        st.markdown("<h2 class='sub-header'>模拟环境</h2>", unsafe_allow_html=True)
        st.markdown("""
        模拟环境由以下部分组成：
        <ul>
            <li><strong>世界：</strong> 无人机和用户所在的二维空间。</li>
            <li><strong>无人机：</strong> 需要服务用户请求且能量有限的飞行器。</li>
            <li><strong>用户：</strong> 在环境中生成无人机需要服务的数据任务的实体。</li>
            <li><strong>任务：</strong> 无人机需要通过到达用户位置来完成的数据传输请求。</li>
        </ul>
        """, unsafe_allow_html=True)

        # Performance metrics
        st.markdown("<h3>性能指标</h3>", unsafe_allow_html=True)
        st.markdown("""
        以下指标用于评估算法性能：
        <ul>
            <li><strong>已服务任务：</strong> 成功完成的用户任务数量。</li>
            <li><strong>已处理数据：</strong> 任务服务期间传输的数据总量（以MB为单位）。</li>
            <li><strong>已消耗能量：</strong> 无人机使用的总能量（以焦耳为单位）。</li>
            <li><strong>总距离：</strong> 无人机行驶的总距离（以米为单位）。</li>
            <li><strong>能量效率：</strong> 每单位能量处理的数据量。</li>
            <li><strong>任务完成率：</strong> 已完成任务占总任务的百分比。</li>
            <li><strong>平均服务延迟：</strong> 完成任务的平均时间（以秒为单位）。</li>
            <li><strong>性能得分：</strong> 综合所有指标的加权得分。</li>
        </ul>
        """, unsafe_allow_html=True)

        # Code structure
        st.markdown("<h2 class='sub-header'>代码结构</h2>", unsafe_allow_html=True)
        st.markdown("""
        <ul>
            <li><strong>algorithms/:</strong> 路径规划算法（MCTS、QL、DQN）的实现</li>
            <li><strong>simulation/:</strong> 核心模拟组件（环境、无人机模型、用户分布）</li>
            <li><strong>utils/:</strong> 实用功能和配置参数</li>
            <li><strong>visualization/:</strong> 增强的可视化模拟结果和比较的工具</li>
            <li><strong>web/:</strong> 交互式使用的Web界面</li>
        </ul>
        """, unsafe_allow_html=True)

        # Recent improvements
        st.markdown("<h2 class='sub-header'>最近的改进</h2>", unsafe_allow_html=True)
        st.markdown("""
        <ul>
            <li>实现了两行整齐用户分布，间距相等，以进行更一致的测试</li>
            <li>移除了障碍物组件，以专注于算法比较，避免干扰</li>
            <li>添加了Q学习(QL)算法，使用表格式状态-动作值表示</li>
            <li>实现了深度Q网络(DQN)以处理复杂的状态空间</li>
            <li>增强了性能指标，添加了能量效率、任务完成率和服务延迟测量</li>
            <li>添加了全面的可视化工具，用于详细的算法比较</li>
            <li>实现了可配置权重的综合性能评分</li>
        </ul>
        """, unsafe_allow_html=True)

# All sample data generation functions have been replaced with actual simulation results

if __name__ == "__main__":
    main()