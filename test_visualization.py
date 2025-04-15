"""
Test script for the visualization module.
"""

import matplotlib.pyplot as plt
import base64
from io import BytesIO
from PIL import Image
import numpy as np

from visualization.plotter import plot_trajectory_with_users

def main():
    # Create sample data
    trajectory = [(100, 100), (150, 150), (200, 200), (250, 250), (300, 300), 
                 (350, 350), (400, 400), (450, 450), (500, 500)]
    
    user_positions = {
        0: (150, 100),
        1: (250, 150),
        2: (350, 200),
        3: (450, 250),
        4: (550, 300),
        5: (150, 400),
        6: (250, 450),
        7: (350, 500),
        8: (450, 550),
    }
    
    user_tasks = {
        0: True,
        1: False,
        2: True,
        3: False,
        4: True,
        5: False,
        6: True,
        7: False,
        8: True,
    }
    
    # Generate fixed points
    fixed_points = [(100, 100), (200, 200), (300, 300), (400, 400), (500, 500)]
    
    # Generate service positions
    service_positions = [(150, 150), (350, 350)]
    
    # Generate the plot
    plot_base64 = plot_trajectory_with_users(
        trajectory=trajectory,
        user_positions=user_positions,
        user_tasks=user_tasks,
        world_size=(600, 600),
        fixed_points=fixed_points,
        service_positions=service_positions,
        title="UAV Path Planning Simulation Test",
        language="English"
    )
    
    # Convert base64 to image
    img_data = base64.b64decode(plot_base64)
    img = Image.open(BytesIO(img_data))
    
    # Save the image
    img.save("test_visualization.png")
    print("Visualization saved to test_visualization.png")
    
    # Also display the image
    plt.figure(figsize=(12, 9))
    plt.imshow(np.array(img))
    plt.axis('off')
    plt.show()

if __name__ == "__main__":
    main()
