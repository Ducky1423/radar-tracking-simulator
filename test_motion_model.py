import numpy as np
from motion_model import ct_motion_model

# Start at the same initial state as traj2 from Step 1's test: [px, py, vx, vy], plus omega=0.05
x = np.array([0, 50, 5, 0, 0.05])
dt = 1.0

# Manually step forward 5 times using ct_motion_model
for step in range(5):
    x = ct_motion_model(x, dt)
    print(x)