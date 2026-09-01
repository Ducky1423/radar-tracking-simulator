import numpy as np

rng = np.random.default_rng(seed=42)

for i in range(5):
    noise_value = rng.normal(0, 1.0)
    print(noise_value)

def cartesian_to_polar(px, py):
    range_val = np.sqrt(px**2 + py**2)
    bearing_val = np.arctan2(py, px)
    return range_val, bearing_val

def noisy_measurement(px, py, range_std, bearing_std, rng):
    range_val, bearing_val = cartesian_to_polar(px, py)
    noisy_range = range_val + rng.normal(0, range_std)
    noisy_bearing = bearing_val + rng.normal(0, bearing_std)
    return noisy_range, noisy_bearing