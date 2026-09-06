import numpy as np
from ukf import compute_weights, predict, predict_measurement, update

class Track:
    def __init__(self, initial_x, initial_P, dt=1.0):
        self.x = initial_x
        self.P = initial_P
        self.dt = dt

        n = 5
        self.alpha, self.beta, self.kappa = 0.1, 2, 0
        self.lambda_ = self.alpha**2 * (n + self.kappa) - n
        self.Wm, self.Wc = compute_weights(n, self.alpha, self.beta, self.kappa)
        self.Q = np.diag([0.1, 0.1, 0.1, 0.1, 0.001])
        self.R = np.diag([1.0**2, 0.01**2])

    def predict_step(self):
        self.x_pred, self.P_pred, self.pred_sigma = predict(
            self.x, self.P, self.dt, self.Wm, self.Wc, self.lambda_, self.Q
        )
        self.z_pred, self.S, self.z_sigma = predict_measurement(
            self.pred_sigma, self.Wm, self.Wc, self.R
        )
        return self.z_pred  # needed later for data association

    def update_step(self, z_actual):
        self.x, self.P = update(
            self.x_pred, self.P_pred, self.pred_sigma, z_actual,
            self.Wc, self.z_pred, self.S, self.z_sigma
        )