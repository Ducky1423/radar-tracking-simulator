import numpy as np
import matplotlib.pyplot as plt
from ground_truth import generate_ct_trajectory
from sensor import generate_radar_measurements
from track_manager import TrackManager
from data_association import associate_measurements

# Target 1: present the whole time. Target 2: only appears from timestep 30 onward.
true1 = generate_ct_trajectory(x0=[0, 50, 5, 0], omega=0.05, dt=1.0, n_steps=100)
true2 = generate_ct_trajectory(x0=[0, -50, 5, 0], omega=-0.03, dt=1.0, n_steps=100)

rng = np.random.default_rng(seed=4)
meas1 = generate_radar_measurements(true1, range_std=1.0, bearing_std=0.01, rng=rng)
meas2 = generate_radar_measurements(true2, range_std=1.0, bearing_std=0.01, rng=rng)

manager = TrackManager(max_misses=5)
track_counts = []

for k in range(100):
    if k < 30:
        pool = np.array([meas1[k]])  # only target 1 visible
    else:
        pool = np.array([meas1[k], meas2[k]])  # both visible

    manager.step(pool, associate_measurements)
    track_counts.append(len(manager.tracks))

print("Track count over time (sampled every 10 steps):", track_counts[::10])
