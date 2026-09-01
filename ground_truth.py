import numpy as np

def generate_ct_trajectory(x0, omega, dt, n_steps):
    """
    Generate a ground-truth trajectory under a constant turn-rate model.

    x0: initial state [px, py, vx, vy]
    omega: turn rate in rad/s (0 = straight line)
    dt: timestep in seconds
    n_steps: number of steps to simulate

    Returns: array of shape (n_steps, 4), the true state at each timestep
    """
    states = np.zeros((n_steps, 4))
    states[0] = x0

    for k in range(1, n_steps):
        px, py, vx, vy = states[k-1]

        if abs(omega) < 1e-6:
            # straight-line special case
            px_new = px + vx * dt
            py_new = py + vy * dt
            vx_new = vx
            vy_new = vy
        else:
            sin_wt = np.sin(omega * dt)
            cos_wt = np.cos(omega * dt)
            px_new = px + (vx * sin_wt - vy * (1 - cos_wt)) / omega
            py_new = py + (vy * sin_wt + vx * (1 - cos_wt)) / omega
            vx_new = vx * cos_wt - vy * sin_wt
            vy_new = vx * sin_wt + vy * cos_wt

        states[k] = [px_new, py_new, vx_new, vy_new]

    return states