import numpy as np

def ct_motion_model(x, dt):
    px = x[0]
    py = x[1]
    vx = x[2]
    vy = x[3]
    omega = x[4]

    if abs(omega) < 1e-6:
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

    omega_new = omega

    x_new = np.array([px_new, py_new, vx_new, vy_new, omega_new])
    return x_new