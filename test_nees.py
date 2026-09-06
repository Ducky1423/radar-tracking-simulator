import numpy as np
from ground_truth import generate_ct_trajectory
from sensor import generate_radar_measurements
from ukf import compute_weights, predict, predict_measurement, update

true_states = generate_ct_trajectory(x0=[0, 50, 5, 0], omega=0.05, dt=1.0, n_steps=100)
rng = np.random.default_rng(seed=1)
measurements = generate_radar_measurements(true_states, range_std=1.0, bearing_std=0.01, rng=rng)

n = 5
alpha, beta, kappa = 0.1, 2, 0
lambda_ = alpha**2 * (n + kappa) - n
Wm, Wc = compute_weights(n, alpha, beta, kappa)
Q = np.diag([0.01, 0.01, 0.01, 0.01, 0.0001])
R = np.diag([1.0**2, 0.01**2])

x = np.array([0.0, 50.0, 5.0, 0.0, 0.0])
P = np.diag([10.0, 10.0, 1.0, 1.0, 0.01])

nees_values = []

for k in range(100):
    x_pred, P_pred, pred_sigma = predict(x, P, dt=1.0, Wm=Wm, Wc=Wc, lambda_=lambda_, Q=Q)
    z_pred, S, z_sigma = predict_measurement(pred_sigma, Wm, Wc, R)
    x, P = update(x_pred, P_pred, pred_sigma, measurements[k], Wc, z_pred, S, z_sigma)

    # true state includes true omega, which we don't directly have -> compare only px,py,vx,vy (first 4)
    true_x4 = true_states[k]
    est_x4 = x[:4]
    P4 = P[:4, :4]

    diff = true_x4 - est_x4
    nees = diff.T @ np.linalg.inv(P4) @ diff
    nees_values.append(nees)

avg_nees = np.mean(nees_values)
print(f"Average NEES: {avg_nees:.2f} (ideal for n=4 comparable states: ~4.0)")