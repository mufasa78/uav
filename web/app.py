"""
Flask web application for the UAV path planning simulation.
用于无人机路径规划模拟的Flask Web应用程序。
"""

import os
import logging
import json
import random
import math
from io import BytesIO
import base64
from typing import Dict, List, Any, Tuple, Optional

from flask import Flask, render_template, request, jsonify, redirect, url_for, abort, session
from flask_session import Session

from simulation.environment import Environment
from algorithms.base import PathPlanningAlgorithm
from algorithms.mcts import MCTSAlgorithm
from algorithms.rrt import RRTAlgorithm
from utils.config import WORLD_SIZE, NUM_USERS

# Configure logging
# 配置日志
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Create Flask app
# 创建Flask应用
app = Flask(__name__)
app.config["SECRET_KEY"] = os.environ.get("SECRET_KEY", "default-secret-key")
app.config["SESSION_TYPE"] = "filesystem"
app.config["SESSION_PERMANENT"] = True
app.config["SESSION_USE_SIGNER"] = True
Session(app)

# Global variables
# 全局变量
environment = Environment()
algorithms = {
    "mcts": MCTSAlgorithm(),
    "rrt": RRTAlgorithm()
}

# Set up algorithms
# 设置算法
for alg_name, alg in algorithms.items():
    alg.setup(environment)

@app.route("/")
def index():
    """
    Home page - redirect to simulation.
    主页 - 重定向到模拟页面。
    """
    return redirect(url_for("simulation"))

@app.route("/simulation")
def simulation():
    """
    Simulation page.
    模拟页面。
    """
    return render_template(
        "simulation.html",
        world_size=WORLD_SIZE,
        num_users=NUM_USERS,
        algorithms=list(algorithms.keys())
    )

@app.route("/api/reset", methods=["POST"])
def reset_simulation():
    """
    Reset the simulation.
    重置模拟。
    """
    environment.reset()
    return jsonify({"status": "success"})

@app.route("/api/run", methods=["POST"])
def run_simulation():
    """
    Run the simulation with the selected algorithm.
    使用选定的算法运行模拟。
    """
    data = request.json
    algorithm_name = data.get("algorithm", "mcts")
    max_steps = data.get("max_steps", 1000)
    
    # Get the algorithm
    # 获取算法
    if algorithm_name not in algorithms:
        return jsonify({"status": "error", "message": f"Unknown algorithm: {algorithm_name}"})
    
    algorithm = algorithms[algorithm_name]
    
    # Run the simulation
    # 运行模拟
    metrics = algorithm.run_episode(max_steps=max_steps)
    
    # Save results to file
    # 将结果保存到文件
    results_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results")
    os.makedirs(results_dir, exist_ok=True)
    
    filename = f"{algorithm_name}_{random.randint(0, 10000)}.json"
    filepath = os.path.join(results_dir, filename)
    
    with open(filepath, "w") as f:
        json.dump(metrics, f)
    
    return jsonify({
        "status": "success",
        "metrics": metrics,
        "filename": filename
    })

@app.route("/comparison")
def comparison():
    """
    Comparison page.
    比较页面。
    """
    # Get all result files
    # 获取所有结果文件
    results = []
    results_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results")
    
    if os.path.exists(results_dir):
        for filename in os.listdir(results_dir):
            if filename.endswith(".json"):
                filepath = os.path.join(results_dir, filename)
                try:
                    with open(filepath, "r") as f:
                        data = json.load(f)
                        results.append({
                            "filename": filename,
                            "algorithm": data.get("algorithm", "unknown"),
                            "metrics": {
                                "serviced_tasks": data.get("serviced_tasks", 0),
                                "data_processed": data.get("data_processed", 0),
                                "energy_consumed": data.get("energy_consumed", 0),
                                "total_distance": data.get("total_distance", 0),
                                "remaining_energy": data.get("remaining_energy", 0)
                            }
                        })
                except (json.JSONDecodeError, OSError) as e:
                    logger.error(f"Error reading {filepath}: {e}")
    
    return render_template("comparison.html", results=results)

@app.route("/documentation")
def documentation():
    """
    Documentation page.
    文档页面。
    """
    logger.debug(f"Current session language: {session.get('language', 'not set')}")
    return render_template("documentation.html")

@app.route("/debug_session")
def debug_session():
    """
    Debug route to check session data.
    调试路由，用于检查会话数据。
    """
    return jsonify({
        "session_data": {
            "language": session.get("language", "not set"),
            "all_data": dict(session)
        },
        "request_path": request.path,
        "cookies": dict(request.cookies)
    })

# Add the language switcher route
# 添加语言切换路由
@app.route("/switch_language", methods=["POST"])
def switch_language():
    """
    Switch between English and Chinese language.
    在英文和中文之间切换语言。
    """
    lang = request.form.get("language", "en")
    session["language"] = lang
    logger.debug(f"Language switched to: {lang}")
    session.modified = True
    return redirect(request.referrer or url_for("index"))

@app.errorhandler(404)
def page_not_found(e):
    """
    Return a custom 404 error.
    返回自定义404错误页面。
    """
    return render_template('404.html'), 404

@app.errorhandler(500)
def server_error(e):
    """
    Return a custom 500 error.
    返回自定义500错误页面。
    """
    return render_template('500.html'), 500

# Before each request, check the language setting
# 在每个请求之前，检查语言设置
@app.before_request
def before_request():
    """
    Set language preference before each request.
    在每个请求之前设置语言首选项。
    """
    if "language" not in session:
        session["language"] = "en"  # Default to English

# Pass the language to all templates
# 将语言传递给所有模板
@app.context_processor
def inject_language():
    """
    Inject language variable into all templates.
    将语言变量注入所有模板。
    """
    lang = session.get("language", "en")
    logger.debug(f"Injecting language in template: {lang}")
    return {"language": lang}

if __name__ == "__main__":
    # Start the Flask development server
    # 启动Flask开发服务器
    app.run(host="0.0.0.0", port=5000, debug=True)