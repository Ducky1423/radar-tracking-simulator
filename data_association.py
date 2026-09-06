import numpy as np

def mahalanobis_distance(z_actual, z_pred, S):
    diff = z_actual - z_pred
    return np.sqrt(diff.T @ np.linalg.inv(S) @ diff)

def associate_measurements(tracks, measurements):
    """
    tracks: list of Track objects (already had predict_step() called this timestep)
    measurements: array of shape (n_meas, 2)

    Returns: dict mapping track index -> measurement index (best match)
    """
    assignments = {}
    used_measurements = set()

    for t_idx, track in enumerate(tracks):
        best_meas_idx = None
        best_dist = float('inf')

        for m_idx, z in enumerate(measurements):
            if m_idx in used_measurements:
                continue
            dist = mahalanobis_distance(z, track.z_pred, track.S)
            if dist < best_dist:
                best_dist = dist
                best_meas_idx = m_idx

        if best_meas_idx is not None:
            assignments[t_idx] = best_meas_idx
            used_measurements.add(best_meas_idx)

    return assignments
