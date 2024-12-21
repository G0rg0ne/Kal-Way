import numpy as np
from scipy.interpolate import CubicSpline
from scipy.integrate import quad
import matplotlib.pyplot as plt
import random

def generate_random_spline_with_tangents(start_point, end_point, num_waypoints=5, seed=None):
    """
    Generates a random spline from a start point to an endpoint with random positive waypoints
    and computes tangents at each point.
    """
    if seed is not None:
        np.random.seed(seed)

    # Generate random waypoints
    x_coords = np.linspace(start_point[0], end_point[0], num_waypoints + 2)
    y_coords = np.random.uniform(low=0, high=random.randint(10, 50), size=num_waypoints + 2)  # Positive values
    y_coords[0], y_coords[-1] = start_point[1], end_point[1]  # Set start and end points

    # Create the cubic spline
    cs = CubicSpline(x_coords, y_coords)

    # Generate points for the spline
    x_spline = np.linspace(start_point[0], end_point[0], 100)
    y_spline = cs(x_spline)

    # Compute the tangents (slopes) at each point
    tangents = cs.derivative()(x_spline)
    
    return x_spline, y_spline, tangents, cs
def sample_points_uniformly(cs, start_x, end_x, speed, num_samples=50):
    """
    Samples points uniformly along the spline based on a constant speed.

    Parameters:
        cs (CubicSpline): The cubic spline object.
        start_x (float): Starting x-coordinate of the spline.
        end_x (float): Ending x-coordinate of the spline.
        speed (float): Speed of the object.
        num_samples (int): Number of points to sample.

    Returns:
        list: A list of dictionaries containing uniform positions, headings, and speed.
    """
    # Define arc length function
    def arc_length_func(x):
        dx = 1.0  # Derivative of x with respect to itself is 1
        dy = cs.derivative()(x)  # Derivative of the spline
        return np.sqrt(dx**2 + dy**2) #

    # Compute total arc length of the spline , this the formula to Compute a definite integral.
    total_arc_length, _ = quad(arc_length_func, start_x, end_x)

    # Define uniform distances along the spline
    distances = np.linspace(0, total_arc_length, num_samples)

    # Find x-coordinates corresponding to these uniform distances
    uniform_x = [start_x]
    current_length = 0

    for d in distances[1:]:
        while current_length < d:
            step = 0.01  # Increment x by small steps
            current_length += arc_length_func(uniform_x[-1]) * step
            uniform_x[-1] += step
        uniform_x.append(uniform_x[-1])

    uniform_x = np.array(uniform_x)

    # Compute y-coordinates, headings, and speeds for these x-coordinates
    uniform_y = cs(uniform_x)
    tangents = cs.derivative()(uniform_x)
    headings = np.arctan2(tangents, 1)  # Compute headings in radians

    samples = []
    for x, y, heading in zip(uniform_x, uniform_y, headings):
        samples.append({"x": x, "y": y, "heading": heading, "speed": speed})
    return samples

def plot_samples(samples, plot_name):
    """
    Plots the sampled points with arrows representing the heading and adds red dots at the sample positions.
    """
    x = [sample["x"] for sample in samples]
    y = [sample["y"] for sample in samples]
    headings = [sample["heading"] for sample in samples]

    plt.figure(figsize=(10, 8))
    
    # Plot the sampled path as a line
    plt.plot(x, y, label='Sampled Path', color='blue')
    
    # Add red dots at the sample positions
    plt.scatter(x, y, color='red', label='Sample Positions', zorder=5)
    
    # Add arrows for the headings
    for i in range(0, len(samples), 10):  # Plot every 10th point
        dx = 0.5 * np.cos(headings[i])
        dy = 0.5 * np.sin(headings[i])
        plt.arrow(x[i], y[i], dx, dy, head_width=0.2, head_length=0.3, color='green')

    plt.title('Sampled Path with Uniform Movement')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'reporting/{plot_name}.png')
    plt.show()

def add_noise_to_samples(samples, noise_std=0.5):
    """
    Adds Gaussian noise to the original sampled points.

    Parameters:
        samples (list): List of original sampled points.
        noise_std (float): Standard deviation of the Gaussian noise.

    Returns:
        list: A list of noisy samples with positions, headings, and speed.
    """
    noisy_samples = []
    for sample in samples:
        noisy_x = sample["x"] + np.random.normal(0, noise_std)  # Add noise to x
        noisy_y = sample["y"] + np.random.normal(0, noise_std)  # Add noise to y
        noisy_samples.append({
            "x": noisy_x,
            "y": noisy_y,
            "heading": sample["heading"],  # Retain the original heading
            "speed": sample["speed"],  # Retain the original speed
        })
    return noisy_samples

if __name__ == "__main__":
    # Define start and end points
    start = (0, 30)
    end = (30, 5)
    
    # Generate the spline and tangents
    x_spline, y_spline, tangents, cs = generate_random_spline_with_tangents(
        start, end, num_waypoints=random.randint(1, 10), seed=42
    )
    
    # Simulate movement by sampling points uniformly along the spline
    speed = 1.0  # Speed of the object
    uniform_samples = sample_points_uniformly(cs, start[0], end[0], speed, num_samples=50)
    noisy_samples = add_noise_to_samples(uniform_samples, noise_std=1)
    
    # Plot the uniform samples
    plot_samples(uniform_samples,"moving_object_2D")
    plot_samples(noisy_samples,"moving_object_2D_with_noise")
