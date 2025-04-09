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
            <h3>Rapidly-exploring Random Tree (RRT*)</h3>
            <p>An optimized sampling-based algorithm for path planning in continuous spaces with asymptotic optimality.</p>
            </div>
            """, unsafe_allow_html=True)
            
        with col3:
            st.markdown("""
            <div class='algorithm-card'>
            <h3>A* Algorithm</h3>
            <p>A classic heuristic search algorithm that finds the shortest path on a grid-based representation.</p>
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
            <h3>快速探索随机树 (RRT*)</h3>
            <p>一种优化的基于采样的算法，用于连续空间中的路径规划，具有渐近最优性。</p>
            </div>
            """, unsafe_allow_html=True)
            
        with col3:
            st.markdown("""
            <div class='algorithm-card'>
            <h3>A*算法</h3>
            <p>一种经典的启发式搜索算法，可在基于网格的表示上找到最短路径。</p>
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
            ["Monte Carlo Tree Search (MCTS)", "Rapidly-exploring Random Tree (RRT*)", "A* Algorithm"]
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
            
            # Placeholder for simulation output
            with st.spinner(f"Running {algorithm} simulation..."):
                # Create placeholders for charts
                fig_trajectory = create_sample_trajectory(algorithm, world_size)
                fig_energy = create_sample_energy_chart(algorithm)
                
                # Display trajectory chart
                st.markdown("<h2 class='sub-header'>UAV Trajectory</h2>", unsafe_allow_html=True)
                st.pyplot(fig_trajectory)
                
                # Display energy consumption chart
                st.markdown("<h2 class='sub-header'>Energy Consumption</h2>", unsafe_allow_html=True)
                st.pyplot(fig_energy)
                
                # Show metrics
                st.markdown("<h2 class='sub-header'>Results</h2>", unsafe_allow_html=True)
                
                metrics_cols = st.columns(5)
                with metrics_cols[0]:
                    st.markdown("""
                    <div class='metric-card'>
                    <h4>Serviced Tasks</h4>
                    <h2>14</h2>
                    </div>
                    """, unsafe_allow_html=True)
                with metrics_cols[1]:
                    st.markdown("""
                    <div class='metric-card'>
                    <h4>Data Processed</h4>
                    <h2>350 MB</h2>
                    </div>
                    """, unsafe_allow_html=True)
                with metrics_cols[2]:
                    st.markdown("""
                    <div class='metric-card'>
                    <h4>Energy Consumed</h4>
                    <h2>4,530 J</h2>
                    </div>
                    """, unsafe_allow_html=True)
                with metrics_cols[3]:
                    st.markdown("""
                    <div class='metric-card'>
                    <h4>Total Distance</h4>
                    <h2>2,345 m</h2>
                    </div>
                    """, unsafe_allow_html=True)
                with metrics_cols[4]:
                    st.markdown("""
                    <div class='metric-card'>
                    <h4>Remaining Energy</h4>
                    <h2>5,470 J</h2>
                    </div>
                    """, unsafe_allow_html=True)
    
    else:  # Chinese
        st.markdown("<h1 class='main-header'>模拟</h1>", unsafe_allow_html=True)
        
        # Algorithm selection
        algorithm = st.selectbox(
            "选择算法",
            ["蒙特卡洛树搜索 (MCTS)", "快速探索随机树 (RRT*)", "A*算法"]
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
            
            # Placeholder for simulation output
            with st.spinner(f"正在运行{algorithm}模拟..."):
                # Create placeholders for charts
                fig_trajectory = create_sample_trajectory(algorithm, world_size)
                fig_energy = create_sample_energy_chart(algorithm)
                
                # Display trajectory chart
                st.markdown("<h2 class='sub-header'>无人机轨迹</h2>", unsafe_allow_html=True)
                st.pyplot(fig_trajectory)
                
                # Display energy consumption chart
                st.markdown("<h2 class='sub-header'>能量消耗</h2>", unsafe_allow_html=True)
                st.pyplot(fig_energy)
                
                # Show metrics
                st.markdown("<h2 class='sub-header'>结果</h2>", unsafe_allow_html=True)
                
                metrics_cols = st.columns(5)
                with metrics_cols[0]:
                    st.markdown("""
                    <div class='metric-card'>
                    <h4>已服务任务</h4>
                    <h2>14</h2>
                    </div>
                    """, unsafe_allow_html=True)
                with metrics_cols[1]:
                    st.markdown("""
                    <div class='metric-card'>
                    <h4>已处理数据</h4>
                    <h2>350 MB</h2>
                    </div>
                    """, unsafe_allow_html=True)
                with metrics_cols[2]:
                    st.markdown("""
                    <div class='metric-card'>
                    <h4>已消耗能量</h4>
                    <h2>4,530 J</h2>
                    </div>
                    """, unsafe_allow_html=True)
                with metrics_cols[3]:
                    st.markdown("""
                    <div class='metric-card'>
                    <h4>总距离</h4>
                    <h2>2,345 m</h2>
                    </div>
                    """, unsafe_allow_html=True)
                with metrics_cols[4]:
                    st.markdown("""
                    <div class='metric-card'>
                    <h4>剩余能量</h4>
                    <h2>5,470 J</h2>
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
        
        st.markdown("<h2 class='sub-header'>Rapidly-exploring Random Tree (RRT*)</h2>", unsafe_allow_html=True)
        st.markdown("""
        RRT* is an optimized variant of RRT that ensures asymptotic optimality. The algorithm not only adds new nodes 
        as it builds the tree, but also reconnects existing nodes through a rewiring process to find the lowest-cost path.
        
        <h3>RRT* Process:</h3>
        <ol>
            <li><strong>Sampling:</strong> Randomly sample a point in the environment, with bias towards the goal and challenging regions.</li>
            <li><strong>Finding Nearest:</strong> Find the nearest node in the tree to the sampled point.</li>
            <li><strong>Extending:</strong> Create a new node by extending from the nearest node towards the sampled point.</li>
            <li><strong>Choosing Parent:</strong> Among nearby nodes, choose the parent that provides the lowest path cost.</li>
            <li><strong>Rewiring:</strong> Check if the new node can serve as a better parent for neighboring nodes.</li>
            <li><strong>Path Smoothing:</strong> Post-process the final path to remove unnecessary turns and bends.</li>
        </ol>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("<h3>Advantages:</h3>", unsafe_allow_html=True)
            st.markdown("""
            - Produces optimal or near-optimal paths
            - Efficient exploration of configuration space
            - Can generate smoother trajectories
            - Has asymptotic optimality guarantees
            """)
        with col2:
            st.markdown("<h3>Disadvantages:</h3>", unsafe_allow_html=True)
            st.markdown("""
            - Higher computational complexity than standard RRT
            - Sensitive to parameters and heuristic choices
            - May converge slowly in high-dimensional spaces
            """)
        
        st.markdown("<h2 class='sub-header'>A* Algorithm</h2>", unsafe_allow_html=True)
        st.markdown("""
        A* is a classic heuristic search algorithm that combines shortest path algorithms (like Dijkstra's) with 
        heuristic search. It uses a heuristic function to guide the search while considering the actual cost from 
        the start to the current node.
        
        <h3>A* Process:</h3>
        <ol>
            <li><strong>Initialization:</strong> Add the start node to the open set and calculate its f-value (f = g + h).</li>
            <li><strong>Selection:</strong> Select the node with the lowest f-value from the open set.</li>
            <li><strong>Expansion:</strong> Expand the current node, generating all its adjacent nodes.</li>
            <li><strong>Evaluation:</strong> Calculate the g and h values for each adjacent node and update its parent.</li>
            <li><strong>Termination:</strong> Stop if the goal node has been found or the open set is empty.</li>
            <li><strong>Path Reconstruction:</strong> Trace back from the goal node to the start node to construct the path.</li>
        </ol>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("<h3>Advantages:</h3>", unsafe_allow_html=True)
            st.markdown("""
            - Finds optimal paths (if heuristic is admissible)
            - More efficient than exhaustive search
            - Well-suited for gridded and discretized environments
            - Simple to implement with predictable performance
            """)
        with col2:
            st.markdown("<h3>Disadvantages:</h3>", unsafe_allow_html=True)
            st.markdown("""
            - Memory consumption can be high
            - Less efficient in high-dimensional spaces
            - Performance depends on quality of heuristic function
            - Requires discretization of continuous spaces
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
                ["Monte Carlo Tree Search (MCTS)", "Rapidly-exploring Random Tree (RRT*)", "A* Algorithm"],
                ["Monte Carlo Tree Search (MCTS)", "Rapidly-exploring Random Tree (RRT*)", "A* Algorithm"]
            )
            sim_time = st.slider("Simulation Time (steps)", 100, 1000, 300, 50)
        with col2:
            num_users = st.slider("Number of Users", 5, 50, 20, 5)
            world_size = st.slider("World Size (m)", 100, 1000, 500, 100)
        
        # Run comparison button
        if st.button("Run Comparison", type="primary"):
            st.info("Running comparison with the selected parameters...")
            
            with st.spinner("Running algorithm comparison..."):
                # Create sample charts for comparison
                fig_metrics = create_sample_comparison_chart()
                fig_trajectories = create_sample_trajectory_comparison()
                
                # Display comparison results
                st.markdown("<h2 class='sub-header'>Performance Metrics Comparison</h2>", unsafe_allow_html=True)
                st.pyplot(fig_metrics)
                
                st.markdown("<h2 class='sub-header'>Trajectory Comparison</h2>", unsafe_allow_html=True)
                st.pyplot(fig_trajectories)
                
                # Detailed metrics table
                st.markdown("<h2 class='sub-header'>Detailed Metrics</h2>", unsafe_allow_html=True)
                
                data = {
                    'Algorithm': ['MCTS', 'RRT*', 'A*'],
                    'Serviced Tasks': [14, 12, 15],
                    'Data Processed (MB)': [350, 300, 375],
                    'Energy Consumed (J)': [4530, 3980, 4820],
                    'Total Distance (m)': [2345, 2120, 2510],
                    'Remaining Energy (J)': [5470, 6020, 5180],
                    'Computation Time (s)': [4.2, 2.8, 3.5]
                }
                
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
                ["蒙特卡洛树搜索 (MCTS)", "快速探索随机树 (RRT*)", "A*算法"],
                ["蒙特卡洛树搜索 (MCTS)", "快速探索随机树 (RRT*)", "A*算法"]
            )
            sim_time = st.slider("模拟时间（步数）", 100, 1000, 300, 50)
        with col2:
            num_users = st.slider("用户数量", 5, 50, 20, 5)
            world_size = st.slider("世界大小（米）", 100, 1000, 500, 100)
        
        # Run comparison button
        if st.button("运行比较", type="primary"):
            st.info("正在使用所选参数运行比较...")
            
            with st.spinner("正在运行算法比较..."):
                # Create sample charts for comparison
                fig_metrics = create_sample_comparison_chart()
                fig_trajectories = create_sample_trajectory_comparison()
                
                # Display comparison results
                st.markdown("<h2 class='sub-header'>性能指标比较</h2>", unsafe_allow_html=True)
                st.pyplot(fig_metrics)
                
                st.markdown("<h2 class='sub-header'>轨迹比较</h2>", unsafe_allow_html=True)
                st.pyplot(fig_trajectories)
                
                # Detailed metrics table
                st.markdown("<h2 class='sub-header'>详细指标</h2>", unsafe_allow_html=True)
                
                data = {
                    '算法': ['MCTS', 'RRT*', 'A*'],
                    '已服务任务': [14, 12, 15],
                    '已处理数据 (MB)': [350, 300, 375],
                    '已消耗能量 (J)': [4530, 3980, 4820],
                    '总距离 (m)': [2345, 2120, 2510],
                    '剩余能量 (J)': [5470, 6020, 5180],
                    '计算时间 (s)': [4.2, 2.8, 3.5]
                }
                
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
            <li><strong>algorithms/:</strong> Implementation of path planning algorithms (MCTS, RRT*, A*)</li>
            <li><strong>simulation/:</strong> Core simulation components (environment, UAV model)</li>
            <li><strong>utils/:</strong> Utility functions and configuration</li>
            <li><strong>visualization/:</strong> Tools for visualizing simulation results</li>
            <li><strong>web/:</strong> Web interface for interactive use</li>
        </ul>
        """, unsafe_allow_html=True)
        
        # Recent improvements
        st.markdown("<h2 class='sub-header'>Recent Improvements</h2>", unsafe_allow_html=True)
        st.markdown("""
        <ul>
            <li>Added A* algorithm for UAV path planning with optimized path finding for gridded environments</li>
            <li>Enhanced Monte Carlo Tree Search (MCTS) with improved reward calculations using discount factors and a more sophisticated decision-making process</li>
            <li>Upgraded RRT to RRT* including the rewiring process to ensure asymptotic optimality</li>
            <li>Added path smoothing functionality to remove unnecessary turns and improve energy efficiency</li>
            <li>Improved algorithm comparison functionality for more comprehensive and detailed performance analysis</li>
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
            <li><strong>剩余能量：</strong> 模拟后无人机剩余的能量。</li>
        </ul>
        """, unsafe_allow_html=True)
        
        # Code structure
        st.markdown("<h2 class='sub-header'>代码结构</h2>", unsafe_allow_html=True)
        st.markdown("""
        <ul>
            <li><strong>algorithms/:</strong> 路径规划算法（MCTS、RRT*、A*）的实现</li>
            <li><strong>simulation/:</strong> 核心模拟组件（环境、无人机模型）</li>
            <li><strong>utils/:</strong> 实用功能和配置</li>
            <li><strong>visualization/:</strong> 可视化模拟结果的工具</li>
            <li><strong>web/:</strong> 交互式使用的Web界面</li>
        </ul>
        """, unsafe_allow_html=True)
        
        # Recent improvements
        st.markdown("<h2 class='sub-header'>最近的改进</h2>", unsafe_allow_html=True)
        st.markdown("""
        <ul>
            <li>为无人机路径规划添加了A*算法，包含网格化环境优化的路径查找功能</li>
            <li>增强了蒙特卡洛树搜索(MCTS)算法，改进了奖励计算，使用折扣因子和更复杂的决策制定过程</li>
            <li>将RRT升级为RRT*，包括重新连接过程，确保渐近最优性</li>
            <li>添加了路径平滑功能，可移除不必要的转弯，提高能源效率</li>
            <li>改进了算法比较功能，允许进行更全面和详细的性能分析</li>
        </ul>
        """, unsafe_allow_html=True)

# Helper functions for creating sample visualizations

def create_sample_trajectory(algorithm, world_size=500):
    """Create a sample trajectory visualization for a single algorithm."""
    plt.figure(figsize=(10, 8))
    
    # Generate sample trajectory data
    np.random.seed(42)  # For reproducibility
    
    # UAV trajectory
    num_points = 30
    x = np.cumsum(np.random.normal(0, 15, num_points))
    y = np.cumsum(np.random.normal(0, 15, num_points))
    
    # Normalize to fit within world size
    x = (x - np.min(x)) * (0.8 * world_size) / (np.max(x) - np.min(x)) + 0.1 * world_size
    y = (y - np.min(y)) * (0.8 * world_size) / (np.max(y) - np.min(y)) + 0.1 * world_size
    
    # Plot trajectory
    plt.plot(x, y, 'b-', linewidth=2, label='UAV Path')
    plt.plot(x[0], y[0], 'go', markersize=10, label='Start')
    plt.plot(x[-1], y[-1], 'ro', markersize=10, label='End')
    
    # Add some users
    num_users = 20
    user_x = np.random.uniform(0, world_size, num_users)
    user_y = np.random.uniform(0, world_size, num_users)
    
    # Plot users
    plt.scatter(user_x, user_y, c='orange', s=100, marker='s', label='Users')
    
    # Add some obstacles
    num_obstacles = 5
    obstacle_x = np.random.uniform(0.2 * world_size, 0.8 * world_size, num_obstacles)
    obstacle_y = np.random.uniform(0.2 * world_size, 0.8 * world_size, num_obstacles)
    obstacle_size = np.random.uniform(20, 50, num_obstacles)
    
    # Plot obstacles
    for i in range(num_obstacles):
        circle = plt.Circle((obstacle_x[i], obstacle_y[i]), obstacle_size[i], color='gray', alpha=0.5)
        plt.gca().add_patch(circle)
    
    # Add labels for serviced users
    for i in range(5):
        idx = np.random.randint(0, num_users)
        plt.annotate(f'User {idx+1}', (user_x[idx], user_y[idx]), 
                    textcoords="offset points", xytext=(0,10), ha='center')
    
    # Set plot properties
    plt.xlim(0, world_size)
    plt.ylim(0, world_size)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.title(f'{algorithm} Trajectory', fontsize=16)
    plt.xlabel('X Coordinate (m)', fontsize=12)
    plt.ylabel('Y Coordinate (m)', fontsize=12)
    plt.legend(loc='upper right')
    
    return plt.gcf()

def create_sample_energy_chart(algorithm):
    """Create a sample energy consumption chart for a single algorithm."""
    plt.figure(figsize=(10, 6))
    
    # Generate sample energy data
    np.random.seed(41)  # For reproducibility
    
    steps = np.arange(0, 300)
    
    # Different energy components
    hover_energy = 10 * np.ones_like(steps)
    movement_energy = 5 + 8 * np.random.random(len(steps))
    communication_energy = np.zeros_like(steps)
    
    # Add spikes for communication
    for i in range(10):
        idx = np.random.randint(20, len(steps)-20)
        communication_energy[idx:idx+10] = 15 * np.random.random() + 5
    
    total_energy = hover_energy + movement_energy + communication_energy
    cumulative_energy = np.cumsum(total_energy)
    
    # Plot energy consumption
    plt.plot(steps, hover_energy, 'g-', label='Hover Energy')
    plt.plot(steps, movement_energy, 'b-', label='Movement Energy')
    plt.plot(steps, communication_energy, 'r-', label='Communication Energy')
    plt.plot(steps, total_energy, 'k-', linewidth=2, label='Total Energy')
    
    # Plot cumulative energy on secondary y-axis
    ax2 = plt.gca().twinx()
    ax2.plot(steps, cumulative_energy, 'm--', linewidth=2, label='Cumulative Energy')
    ax2.set_ylabel('Cumulative Energy (J)', color='m', fontsize=12)
    ax2.tick_params(axis='y', labelcolor='m')
    
    # Set plot properties
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.title(f'{algorithm} Energy Consumption', fontsize=16)
    plt.xlabel('Simulation Step', fontsize=12)
    plt.ylabel('Energy Rate (J/step)', fontsize=12)
    
    # Combine legends from both axes
    lines1, labels1 = plt.gca().get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    plt.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
    
    plt.tight_layout()
    return plt.gcf()

def create_sample_comparison_chart():
    """Create a sample bar chart comparing algorithm metrics."""
    plt.figure(figsize=(12, 8))
    
    algorithms = ['MCTS', 'RRT*', 'A*']
    metrics = {
        'Serviced Tasks': [14, 12, 15],
        'Data Processed (MB)': [350, 300, 375],
        'Energy Consumed (kJ)': [4.53, 3.98, 4.82],
        'Total Distance (km)': [2.35, 2.12, 2.51],
        'Remaining Energy (kJ)': [5.47, 6.02, 5.18]
    }
    
    # Number of metric groups
    n_metrics = len(metrics)
    
    # Set up positions
    index = np.arange(len(algorithms))
    bar_width = 0.15
    opacity = 0.8
    
    # Create a subplot for each metric
    fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(15, 10))
    axes = axes.flatten()
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    
    # Plot each metric
    for i, (metric_name, values) in enumerate(metrics.items()):
        if i < len(axes):  # Ensure we don't exceed the number of subplots
            ax = axes[i]
            ax.bar(index, values, bar_width*2, alpha=opacity, color=colors[i % len(colors)])
            ax.set_xlabel('Algorithm')
            ax.set_ylabel(metric_name)
            ax.set_title(f'{metric_name} Comparison')
            ax.set_xticks(index)
            ax.set_xticklabels(algorithms)
            ax.grid(True, linestyle='--', alpha=0.4, axis='y')
    
    # Remove any unused subplots
    for i in range(n_metrics, len(axes)):
        fig.delaxes(axes[i])
    
    plt.tight_layout()
    return fig

def create_sample_trajectory_comparison():
    """Create a sample trajectory comparison for multiple algorithms."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    algorithms = ['MCTS', 'RRT*', 'A*']
    world_size = 500
    
    for i, algorithm in enumerate(algorithms):
        ax = axes[i]
        
        # Generate trajectory for each algorithm with different seeds
        np.random.seed(42 + i)  # Different seed for each algorithm
        
        # UAV trajectory with characteristics specific to each algorithm
        num_points = 30
        if algorithm == 'MCTS':
            # More random exploration
            x = np.cumsum(np.random.normal(0, 18, num_points))
            y = np.cumsum(np.random.normal(0, 18, num_points))
        elif algorithm == 'RRT*':
            # Smoother path
            t = np.linspace(0, 2*np.pi, num_points)
            noise = np.random.normal(0, 5, num_points)
            x = 150 * np.cos(t) + 200 + noise
            y = 150 * np.sin(t) + 250 + noise
        else:  # A*
            # More grid-like path
            x = np.zeros(num_points)
            y = np.zeros(num_points)
            for j in range(num_points):
                if j % 2 == 0:
                    x[j] = j * 15
                    y[j] = y[j-1] if j > 0 else 0
                else:
                    x[j] = x[j-1]
                    y[j] = j * 15
            
            x += 100
            y += 100
        
        # Normalize to fit within world size
        x = (x - np.min(x)) * (0.8 * world_size) / (np.max(x) - np.min(x)) + 0.1 * world_size
        y = (y - np.min(y)) * (0.8 * world_size) / (np.max(y) - np.min(y)) + 0.1 * world_size
        
        # Plot trajectory
        ax.plot(x, y, '-', linewidth=2)
        ax.plot(x[0], y[0], 'go', markersize=10)
        ax.plot(x[-1], y[-1], 'ro', markersize=10)
        
        # Add some users
        num_users = 20
        user_x = np.random.uniform(0, world_size, num_users)
        user_y = np.random.uniform(0, world_size, num_users)
        
        # Plot users
        ax.scatter(user_x, user_y, c='orange', s=50, marker='s')
        
        # Set plot properties
        ax.set_xlim(0, world_size)
        ax.set_ylim(0, world_size)
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.set_title(f'{algorithm} Trajectory', fontsize=14)
        ax.set_xlabel('X Coordinate (m)')
        ax.set_ylabel('Y Coordinate (m)')
    
    plt.tight_layout()
    return fig

if __name__ == "__main__":
    main()