import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
import os 
import random

def generate_random_spline_with_tangents(start_point, end_point, num_waypoints=5, seed=None):
    """
    Generates a random spline from a start point to an endpoint with random positive waypoints
    and computes tangents at each point.

    Parameters:
        start_point (tuple): The (x, y) coordinates of the start point.
        end_point (tuple): The (x, y) coordinates of the end point.
        num_waypoints (int): Number of random waypoints to generate between the start and end points.
        seed (int, optional): Seed for reproducibility.

    Returns:
        tuple: x and y coordinates of the spline, and the tangents (slopes) at each point.
    """
    if seed is not None:
        np.random.seed(seed)

    # Generate random waypoints
    x_coords = np.linspace(start_point[0], end_point[0], num_waypoints + 2)
    y_coords = np.random.uniform(low=0, high=random.randint(10,50), size=num_waypoints + 2)  # Positive values
    
    y_coords[0], y_coords[-1] = start_point[1], end_point[1]  # Set start and end points

    # Create the cubic spline
    cs = CubicSpline(x_coords, y_coords)

    # Generate points for the spline
    x_spline = np.linspace(start_point[0], end_point[0], 100)
    y_spline = cs(x_spline)

    # Compute the tangents (slopes) at each point
    tangents = cs.derivative()(x_spline)
    
    return x_spline, y_spline, tangents


def plot_spline_with_arrows(x, y,tangents,arrow_length=0.5):
    """
    Plots the spline and adds arrows indicating tangents (headings).

    Parameters:
        x (array): x-coordinates of the spline points.
        y (array): y-coordinates of the spline points.
        tangents (array): Tangents (slopes) at each point.
        start (tuple): Start point coordinates.
        end (tuple): End point coordinates.
        arrow_length (float): Desired length of the arrows.
    """
    # Normalize the tangents for arrow lengths
    dx = np.ones_like(tangents)  # x-component of the tangent is 1
    dy = tangents  # y-component of the tangent is the slope
    arrow_lengths = np.sqrt(dx**2 + dy**2)
    dx = (dx / arrow_lengths) * arrow_length  # Scale to desired length
    dy = (dy / arrow_lengths) * arrow_length  # Scale to desired length

    # Plot the spline
    plt.figure(figsize=(10, 8))
    plt.plot(x, y, label='Random Spline', color='blue')
    plt.scatter([x[0], x[-1]], [y[0], y[-1]], color='red', label='Start/End Points')

    # Add arrows at regular intervals
    for i in range(0, len(x), 10):  # Plot arrows every 10th point
        plt.arrow(x[i], y[i], dx[i], dy[i], head_width=0.2, head_length=0.3, color='green')

    # Add plot labels and legend
    plt.title('Random Spline with Controlled Arrow Lengths')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend()
    plt.grid(True)
    plt.show()
    cor_figure_path = os.path.join('reporting', "spline.png")
    plt.savefig(cor_figure_path)

if __name__ == "__main__":
    # Define start and end points
    start = (0, 5)
    end = (10, 5)
    # Generate the spline and tangents
    x, y, tangents = generate_random_spline_with_tangents(start, end, num_waypoints=random.randint(1,10), seed=42)
    # Plot the spline with arrows
    plot_spline_with_arrows(x, y, tangents,arrow_length=0.5)
