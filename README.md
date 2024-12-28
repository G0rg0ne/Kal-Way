docker run -it --rm -v $(pwd):/app kayway /bin/bash


# Bayesian Filters for Denoising 2D Robot Trajectories

## Overview

This project aims to explore and implement Bayesian filtering techniques for denoising and correcting 2D robot trajectories. The goal is to benchmark the performance of these filters on randomly generated trajectories with added noise. The findings will help evaluate the effectiveness of Bayesian filters in improving trajectory accuracy in noisy environments.

## Features

- **Trajectory Generation**: Generate random 2D robot trajectories using splines.
- **Noise Addition**: Add controlled levels of noise to simulate real-world sensor inaccuracies.
- **Bayesian Filters**: Implement and test filtering techniques such as:
  - Kalman Filter
  - Extended Kalman Filter (EKF)
  - Unscented Kalman Filter (UKF)
  - Particle Filter
- **Benchmarking**: Compare the performance of these filters using predefined metrics.

## Installation

### Prerequisites

- Python 3.8+
- Recommended libraries: `numpy`, `scipy`, `matplotlib`, `pandas`
- For Bayesian filters: `filterpy` (optional for additional filter utilities)

### Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/G0rg0ne/Kal-Way.git
   cd bayesian-filters-robot-trajectories
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### 1. Generate Trajectories

Run the script to generate random 2D trajectories:

```bash
python generate_trajectories.py --num-trajectories 10 --noise-level 0.5
```

This will save noisy and ground-truth trajectories in the `data/` directory.

### 2. Apply Bayesian Filters

Run the filtering script to apply a selected filter on noisy trajectories:

```bash
python apply_filters.py --filter kalman --input data/noisy_trajectories.csv
```

Options for `--filter` include:

- `kalman`
- `ekf`
- `ukf`
- `particle`

### 3. Benchmarking

Evaluate the performance of the filters:

```bash
python benchmark.py --input data/filtered_trajectories.csv
```

This will output metrics such as Mean Squared Error (MSE) and trajectory deviation.

## Project Structure

```
.
├── data/                 # Generated trajectories and filter outputs
├── modules/                  # Source code
│   ├── filters/          # Implementations of Bayesian filters
│   ├── utils/            # Utility functions for trajectory generation and noise
│   ├── generate_trajectories.py
│   ├── apply_filters.py
│   └── benchmark.py
├── requirements.txt      # Python dependencies
├── README.md             # Project documentation
└── reporting/                # results
```

## Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository.
2. Create a feature branch (`git checkout -b feature-name`).
3. Commit your changes (`git commit -m 'Add new feature'`).
4. Push to your fork (`git push origin feature-name`).
5. Open a Pull Request.

## License

This project is licensed under the MIT License. See `LICENSE` for details.

## Acknowledgments

- Inspired by applications in robotics and control systems.
- Built using Python and popular scientific libraries.

## Contact

For questions or feedback, feel free to open an issue or contact me at [[your\_email@example.com](mailto\:your_email@example.com)].

