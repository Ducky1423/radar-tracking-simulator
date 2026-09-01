import numpy as np

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

Wm, Wc = compute_weights(5, 0.1, 2, 0)
print(Wm)
print(Wc)
print(np.sum(Wm))  # this should equal 1.0