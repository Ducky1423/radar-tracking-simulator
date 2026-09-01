# test_ground_truth.py
import numpy as np
import matplotlib.pyplot as plt
from ground_truth import generate_ct_trajectory

dt = 1.0
n_steps = 100

# Three test objects: straight line, gentle left turn, sharper right turn
traj1 = generate_ct_trajectory(x0=[0, 0, 5, 0],     omega=0.0,  dt=dt, n_steps=n_steps)
traj2 = generate_ct_trajectory(x0=[0, 50, 5, 0],    omega=0.05, dt=dt, n_steps=n_steps)
traj3 = generate_ct_trajectory(x0=[0, -50, 5, 0],   omega=-0.1, dt=dt, n_steps=n_steps)

plt.figure(figsize=(8, 6))
plt.plot(traj1[:, 0], traj1[:, 1], label='omega=0 (straight)')
plt.plot(traj2[:, 0], traj2[:, 1], label='omega=0.05 (left turn)')
plt.plot(traj3[:, 0], traj3[:, 1], label='omega=-0.1 (right turn)')
plt.xlabel('X position')
plt.ylabel('Y position')
plt.title('Ground Truth Trajectories (Constant Turn-Rate Model)')
plt.legend()
plt.axis('equal')
plt.grid(True)
plt.show()