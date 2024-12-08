import sys
import os

sys.path.append(os.getcwd())

from utils.generate_spline import generate_random_spline_with_tangents
import random
import math
import matplotlib.pyplot as plt


start = (20, 20)
end = (200, 200)
# Generate the spline and tangents
x, y, tangents = generate_random_spline_with_tangents(start, end, num_waypoints=random.randint(1,10), seed=42)



def simulate_movement(starting_point, headings, speed, time_step=1, map_size=300):
    """
    Simulates a moving object on a 300x300 meter map.

    Args:
        starting_point (tuple): The starting coordinates (x, y) on the map in meters.
        headings (list): A list of heading angles (in degrees) defining the object's direction at each step.
        speed (float): The constant speed of the object in meters per second.
        time_step (float): The time interval between each step in seconds.
        map_size (int): The size of the map in meters (default 300x300).

    Returns:
        list: A list of (x, y) positions of the object at each time step.
    """
    positions = [starting_point]
    x, y = starting_point

    for heading in headings:
        # Convert heading angle to radians
        
        
        # Calculate the change in position
        dx = speed * math.cos(heading) * time_step
        dy = speed * math.sin(heading) * time_step
        
        # Update the current position
        x += dx
        y += dy

        # Keep the object within the map boundaries
        x = max(0, min(x, map_size))
        y = max(0, min(y, map_size))

        positions.append((x, y))

    return positions

# Example usage
starting_point = (x[0], y[0])  # Start at the center of the map
headings = tangents  # Directions in degrees
speed = 2  # Speed in meters per second
time_step = 1  # Time interval in seconds

positions = simulate_movement(starting_point, headings, speed, time_step)

# Plot the trajectory
x_coords, y_coords = zip(*positions)
plt.figure(figsize=(6, 6))
plt.plot(x_coords, y_coords, marker='o')
plt.xlim(0, 300)
plt.ylim(0, 300)
plt.title("Object Trajectory on 300x300 Map")
plt.xlabel("X Coordinate (meters)")
plt.ylabel("Y Coordinate (meters)")
plt.grid(True)
plt.show()
cor_figure_path = os.path.join('reporting', "traj_2D_test.png")
plt.savefig(cor_figure_path)
