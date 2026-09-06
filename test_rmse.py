import numpy as np
from ground_truth import generate_ct_trajectory
from sensor import generate_radar_measurements, polar_to_cartesian
from ukf import compute_weights, predict, predict_measurement, update

true_states = generate_ct_trajectory(x0=[0, 50, 5, 0], omega=0.05, dt=1.0, n_steps=100)
rng = np.random.default_rng(seed=1)
measurements = generate_radar_measurements(true_states, range_std=1.0, bearing_std=0.01, rng=rng)

n = 5
alpha, beta, kappa = 0.1, 2, 0
lambda_ = alpha**2 * (n + kappa) - n
Wm, Wc = compute_weights(n, alpha, beta, kappa)
Q = np.diag([0.1, 0.1, 0.1, 0.1, 0.001])
R = np.diag([1.0**2, 0.01**2])

x = np.array([0.0, 50.0, 5.0, 0.0, 0.0])
P = np.diag([10.0, 10.0, 1.0, 1.0, 0.01])

estimates = np.zeros((100, 5))
for k in range(100):
    x_pred, P_pred, pred_sigma = predict(x, P, dt=1.0, Wm=Wm, Wc=Wc, lambda_=lambda_, Q=Q)
    z_pred, S, z_sigma = predict_measurement(pred_sigma, Wm, Wc, R)
    x, P = update(x_pred, P_pred, pred_sigma, measurements[k], Wc, z_pred, S, z_sigma)
    estimates[k] = x

meas_px, meas_py = polar_to_cartesian(measurements[:, 0], measurements[:, 1])

rmse_measurements = np.sqrt(np.mean((meas_px - true_states[:,0])**2 + (meas_py - true_states[:,1])**2))
rmse_ukf = np.sqrt(np.mean((estimates[:,0] - true_states[:,0])**2 + (estimates[:,1] - true_states[:,1])**2))

print(f"RMSE (raw noisy measurements): {rmse_measurements:.3f}")
print(f"RMSE (UKF estimate): {rmse_ukf:.3f}")