import threading
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline

def generate_random_spline(num_points, spline_length):
    # Generate random points with total length constraint
    points = [(0, 0)]  # Start point at origin
    total_length = 0

    # Generate random points with distances that sum up to the desired total length
    for _ in range(num_points - 1):
        # Generate a random angle (direction)
        angle = np.random.uniform(0, 2 * np.pi)
        # Generate a random step length
        step_length = np.random.uniform(0, spline_length / num_points)
        # Calculate new point position
        last_point = points[-1]
        dx = step_length * np.cos(angle)
        dy = step_length * np.sin(angle)
        new_point = (last_point[0] + dx, last_point[1] + dy)
        points.append(new_point)
        total_length += step_length

    # Scale the points so the total length matches spline_length
    scaling_factor = spline_length / total_length
    scaled_points = np.array(points) * scaling_factor

    # Ensure x coordinates are strictly increasing by sorting them
    x = scaled_points[:, 0]
    y = scaled_points[:, 1]

    # Sort points by x values (necessary for cubic spline interpolation)
    sorted_indices = np.argsort(x)
    x_sorted = x[sorted_indices]
    y_sorted = y[sorted_indices]

    # Fit a cubic spline to the sorted points
    spline = CubicSpline(x_sorted, y_sorted)
    
    # Generate a smooth set of points along the spline
    x_smooth = np.linspace(min(x_sorted), max(x_sorted), 1000)
    y_smooth = spline(x_smooth)
    
    # Compute the headings (angle between consecutive points)
    headings = []
    for i in range(1, len(x_smooth)):
        dx = x_smooth[i] - x_smooth[i - 1]
        dy = y_smooth[i] - y_smooth[i - 1]
        angle = np.arctan2(dy, dx)  # Calculate angle in radians
        headings.append(angle)
    
    # Plot the spline
    plt.figure(figsize=(8, 6))
    plt.plot(x_smooth, y_smooth, label='Spline')
    plt.scatter(x_sorted, y_sorted, color='red', label='Random Points')
    plt.title(f'Random 2D Spline (Total Length = {spline_length})')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend()
    plt.grid(True)
    plt.savefig('random_spline.png')
    plt.show()

    return headings




if __name__ == "__main__":
    # Example usage
    num_points = 10
    spline_length = 50  # Total length of the spline
    headings = generate_random_spline(num_points, spline_length)
    print(f"Generated headings (in radians): {headings}")