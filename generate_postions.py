import numpy as np
import matplotlib.pyplot as plt
import threading
import os
from filterpy.kalman import KalmanFilter

# Parameters
start_point = np.array([0, 0])  # Starting point (x, y)
end_point = np.array([100, 100])  # Ending point (x, y)
velocity = 1  # Constant velocity (distance per time step)
vx = velocity * np.cos(np.radians(45))
vy = velocity * np.sin(np.radians(45))
time_steps = 200  # Number of time steps
noise_std_dev = 5  # Standard deviation of Gaussian noise
output_dir = "reporting"  # Directory to save the figures

# Create the output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)


# Generate the trajectory
def generate_trajectory(start, end, velocity, time_steps):
    # Calculate total distance between start and end
    distance = np.linalg.norm(end - start)

    # Time step duration (constant velocity)
    duration = distance / velocity

    # Time intervals
    time_intervals = np.linspace(0, duration, time_steps)

    # Calculate the trajectory points along the straight line
    trajectory = np.array(
        [start + (end - start) * (t / duration) for t in time_intervals]
    )

    return trajectory,duration


# Add Gaussian noise to the trajectory
def add_gaussian_noise(trajectory, noise_std_dev):
    noise = np.random.normal(0, noise_std_dev, trajectory.shape)  # Gaussian noise
    noisy_trajectory = trajectory + noise
    return noisy_trajectory


def kalmanfilter(dt,noisy_trajectory):
    kalman = KalmanFilter(dim_x=4, dim_z=2)
    kalman.x = np.array([noisy_trajectory[0][0],noisy_trajectory[0][1],vx,vy])
    kalman.F = np.array(
        [
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ]
    )
    kalman.H = np.array([[1., 0., 0., 0.],  # x
                     [0., 1., 0., 0.]])
    kalman.P *= 5**2
     # Process noise (representing process uncertainty)
    # Adjusted Process noise (Q) - assume low uncertainty in position, moderate in velocity
    kalman.Q =0.1**2 * np.array([[dt**4/4, 0, dt**3/2, 0],  # x (position)
                         [0, dt**4/4, 0, dt**3/2],  # y (position)
                         [dt**3/2, 0, dt**2, 0],    # vx (velocity)
                         [0, dt**3/2, 0, dt**2]])   # vy (velocity)

    # Adjusted Measurement noise covariance (R) - assume moderate noise in position measurements
    kalman.R = np.array([[noise_std_dev**2, 0],    # x measurement noise
                         [0, noise_std_dev**2]])   # y measurement noise
    
    # Store the corrected trajectory
    corrected_trajectory = []
    for i in range(len(noisy_trajectory)):
        # Prediction step
        kalman.predict()

        # Get the current noisy measurement (x, y)
        z = np.array([noisy_trajectory[i][0], noisy_trajectory[i][1]])

        # Update step with the noisy measurement
        kalman.update(z)

        # Store the corrected position (x, y) in the trajectory
        corrected_trajectory.append(kalman.x[:2])  # We store only the position (x, y)
    
    return np.array(corrected_trajectory)


def compute_rmse(true_trajectory, corrected_trajectory):
    """
    Compute the Root Mean Squared Error (RMSE) between the true trajectory and the corrected trajectory.
    
    :param true_trajectory: True positions (Nx2 array)
    :param corrected_trajectory: Corrected positions (Nx2 array)
    :return: RMSE value
    """
    # Compute squared differences
    squared_diff = (true_trajectory[:, 0] - corrected_trajectory[:, 0])**2 + \
                   (true_trajectory[:, 1] - corrected_trajectory[:, 1])**2
    
    # Mean of squared differences
    mean_squared_diff = np.mean(squared_diff)
    
    # RMSE (Root Mean Squared Error)
    rmse = np.sqrt(mean_squared_diff)
    
    return rmse

# Thread function to generate and store trajectories
def generate_and_store_trajectories():
    # Generate the noiseless trajectory
    trajectory,duration = generate_trajectory(start_point, end_point, velocity, time_steps)

    # Add Gaussian noise
    noisy_trajectory = add_gaussian_noise(trajectory, noise_std_dev)
    corrected_trajectory = kalmanfilter(duration/time_steps,noisy_trajectory)
    rmse = compute_rmse(corrected_trajectory, trajectory)
    rmse_noise = compute_rmse(noisy_trajectory, trajectory)
    # Store or visualize the trajectories (for now, print the final points)
    print("Noiseless trajectory final point:", trajectory[-1])
    print("Noisy trajectory final point:", noisy_trajectory[-1])

    # Visualize and save the result
    plt.figure(figsize=(10, 6))
    plt.plot(trajectory[:, 0], trajectory[:, 1], label="Noiseless Trajectory")
    plt.scatter(
        noisy_trajectory[:, 0],
        noisy_trajectory[:, 1],
        color="r",
        label="Noisy Trajectory",
        alpha=0.5,
    )
    plt.legend()
    plt.title("Trajectory with Gaussian Noise")

    # Save the figure in the 'figures' folder
    figure_path = os.path.join(output_dir, "trajectory_with_noise.png")
    plt.savefig(figure_path)
    plt.close()
    # Plot the noisy trajectory and the corrected trajectory
    plt.figure(figsize=(10, 6))
    plt.plot(noisy_trajectory[:, 0], noisy_trajectory[:, 1], 'r.', label='Noisy Trajectory')
    plt.plot(corrected_trajectory[:, 0], corrected_trajectory[:, 1], 'b-', label='Corrected Trajectory')
    plt.text(70, 4.5, f"RMSE_kalman = {rmse:.4f}", fontsize=12, color='blue', bbox=dict(facecolor='white', alpha=0.7))
    plt.text(70, 12.5, f"RMSE_noisy = {rmse_noise:.4f}", fontsize=12, color='green', bbox=dict(facecolor='white', alpha=0.7))

    # Adding labels and title
    plt.xlabel('X Position')
    plt.ylabel('Y Position')
    plt.title('Noisy vs Corrected Trajectory using Kalman Filter')
    plt.legend()
    cor_figure_path = os.path.join(output_dir, "corrected_trajectory_with_noise.png")
    plt.savefig(cor_figure_path)

    # Show plot
    plt.grid(True)
    plt.show()




# Create and start the thread
trajectory_thread = threading.Thread(target=generate_and_store_trajectories)
trajectory_thread.start()
trajectory_thread.join()
