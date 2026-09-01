import numpy as np

def cartesian_to_polar(px, py):
    range_val = np.sqrt(px**2 + py**2)
    bearing_val = np.arctan2(py, px)
    return range_val, bearing_val

def noisy_measurement(px, py, range_std, bearing_std, rng):
    range_val, bearing_val = cartesian_to_polar(px, py)
    noisy_range = range_val + rng.normal(0, range_std)
    noisy_bearing = bearing_val + rng.normal(0, bearing_std)
    return noisy_range, noisy_bearing

def generate_radar_measurements(true_states, range_std, bearing_std, rng):
    n_steps = true_states.shape[0]
    measurements = np.zeros((n_steps, 2))

    for k in range(n_steps):
        px = true_states[k, 0]
        py = true_states[k, 1]
        noisy_range, noisy_bearing = noisy_measurement(px, py, range_std, bearing_std, rng)
        measurements[k, 0] = noisy_range
        measurements[k, 1] = noisy_bearing

    return measurements

def polar_to_cartesian(range_val, bearing_val):
    px = range_val * np.cos(bearing_val)
    py = range_val * np.sin(bearing_val)
    return px, py