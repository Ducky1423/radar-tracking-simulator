import numpy as np
import matplotlib.pyplot as plt
from ground_truth import generate_ct_trajectory
from sensor import generate_radar_measurements, polar_to_cartesian

# Step 1: true trajectory
true_states = generate_ct_trajectory(x0=[0, 0, 5, 0], omega=0.0, dt=1.0, n_steps=100)

# Step 2: noisy radar measurements
rng = np.random.default_rng(seed=1)
measurements = generate_radar_measurements(true_states, range_std=1.0, bearing_std=0.01, rng=rng)

# Convert noisy measurements back to (px, py) just so we can plot them
meas_px, meas_py = polar_to_cartesian(measurements[:, 0], measurements[:, 1])

# Plot true trajectory vs noisy measurements
plt.figure(figsize=(8, 6))
plt.plot(true_states[:, 0], true_states[:, 1], label='True trajectory', linewidth=2)
plt.scatter(meas_px, meas_py, s=10, color='red', alpha=0.5, label='Noisy radar measurements')
plt.xlabel('X position')
plt.ylabel('Y position')
plt.title('True Trajectory vs Noisy Radar Measurements')
plt.legend()
plt.axis('equal')
plt.grid(True)
plt.show()