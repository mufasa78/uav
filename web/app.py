"""
Flask web application for the UAV path planning simulation.
"""

import os
import logging
import json
import random
import math
from io import BytesIO
import base64
from typing import Dict, List, Any, Tuple, Optional

from flask import Flask, render_template, request, jsonify, redirect, url_for, abort

from simulation.environment import Environment
from algorithms.base import PathPlanningAlgorithm
from algorithms.mcts import MCTSAlgorithm
from algorithms.rrt import RRTAlgorithm
from utils.config import WORLD_SIZE, NUM_USERS

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create Flask app
app = Flask(__name__)
app.config["SECRET_KEY"] = os.environ.get("SECRET_KEY", "default-secret-key")

# Global variables
environment = Environment()
algorithms = {
    "mcts": MCTSAlgorithm(),
    "rrt": RRTAlgorithm()
}

# Set up algorithms
for alg_name, alg in algorithms.items():
    alg.setup(environment)

@app.route("/")
def index():
    """Home page - redirect to simulation."""
    return redirect(url_for("simulation"))

@app.route("/simulation")
def simulation():
    """Simulation page."""
    return render_template(
        "simulation.html",
        world_size=WORLD_SIZE,
        num_users=NUM_USERS,
        algorithms=list(algorithms.keys())
    )

@app.route("/api/reset", methods=["POST"])
def reset_simulation():
    """Reset the simulation."""
    environment.reset()
    return jsonify({"status": "success"})

@app.route("/api/run", methods=["POST"])
def run_simulation():
    """Run the simulation with the selected algorithm."""
    data = request.json
    algorithm_name = data.get("algorithm", "mcts")
    max_steps = data.get("max_steps", 1000)
    
    # Get the algorithm
    if algorithm_name not in algorithms:
        return jsonify({"status": "error", "message": f"Unknown algorithm: {algorithm_name}"})
    
    algorithm = algorithms[algorithm_name]
    
    # Run the simulation
    metrics = algorithm.run_episode(max_steps=max_steps)
    
    # Save results to file
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
    """Comparison page."""
    # Get all result files
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
    """Documentation page."""
    return render_template("documentation.html")

@app.errorhandler(404)
def page_not_found(e):
    """Return a custom 404 error."""
    return render_template('404.html'), 404

@app.errorhandler(500)
def server_error(e):
    """Return a custom 500 error."""
    return render_template('500.html'), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)