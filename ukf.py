import numpy as np
from motion_model import ct_motion_model
from sensor import cartesian_to_polar

def compute_weights(n, alpha, beta, kappa):
    lambda_ = alpha**2 * (n + kappa) - n

    Wm = np.zeros(2*n + 1)
    Wc = np.zeros(2*n + 1)

    Wm[0] = lambda_ / (n + lambda_)
    Wc[0] = lambda_ / (n + lambda_) + (1 - alpha**2 + beta)

    for i in range(1, 2*n + 1):
        Wm[i] = 1 / (2 * (n + lambda_))
        Wc[i] = 1 / (2 * (n + lambda_))

    return Wm, Wc

def generate_sigma_points(x, P, lambda_):
    n = len(x)
    sigma_points = np.zeros((2*n + 1, n))
    sigma_points[0] = x

    sqrt_matrix = np.linalg.cholesky((n + lambda_) * P)

    for i in range(n):
        sigma_points[i + 1] = x + sqrt_matrix[:, i]
        sigma_points[i + 1 + n] = x - sqrt_matrix[:, i]

    return sigma_points


def predict(x, P, dt, Wm, Wc, lambda_, Q):
    n = len(x)
    sigma_points = generate_sigma_points(x, P, lambda_)

    # push each sigma point through the motion model
    predicted_sigma_points = np.array([ct_motion_model(sp, dt) for sp in sigma_points])

    # recombine into predicted mean
    x_pred = np.sum(Wm[:, None] * predicted_sigma_points, axis=0)

    # recombine into predicted covariance
    P_pred = np.zeros((n, n))
    for i in range(2*n + 1):
        diff = predicted_sigma_points[i] - x_pred
        P_pred += Wc[i] * np.outer(diff, diff)
    P_pred += Q  # add process noise (uncertainty from unmodeled effects)

    return x_pred, P_pred, predicted_sigma_points

def predict_measurement(predicted_sigma_points, Wm, Wc, R):
    n_sigma = predicted_sigma_points.shape[0]
    z_sigma_points = np.array([cartesian_to_polar(sp[0], sp[1]) for sp in predicted_sigma_points])

    z_pred = np.sum(Wm[:, None] * z_sigma_points, axis=0)

    S = np.zeros((2, 2))  # measurement space is 2D: range, bearing
    for i in range(n_sigma):
        diff = z_sigma_points[i] - z_pred
        S += Wc[i] * np.outer(diff, diff)
    S += R  # measurement noise covariance

    return z_pred, S, z_sigma_points


def update(x_pred, P_pred, predicted_sigma_points, z_actual, Wc, z_pred, S, z_sigma_points):
    n = len(x_pred)
    n_sigma = predicted_sigma_points.shape[0]

    # cross-covariance between state and measurement
    Pxz = np.zeros((n, 2))
    for i in range(n_sigma):
        dx = predicted_sigma_points[i] - x_pred
        dz = z_sigma_points[i] - z_pred
        Pxz += Wc[i] * np.outer(dx, dz)

    K = Pxz @ np.linalg.inv(S)  # Kalman gain

    x_new = x_pred + K @ (z_actual - z_pred)
    P_new = P_pred - K @ S @ K.T

    return x_new, P_new