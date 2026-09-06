import numpy as np
from ground_truth import generate_ct_trajectory
from sensor import generate_radar_measurements
from track import Track
from data_association import associate_measurements

true1 = generate_ct_trajectory(x0=[0, 50, 5, 0], omega=0.05, dt=1.0, n_steps=100)
true2 = generate_ct_trajectory(x0=[0, -50, 5, 0], omega=-0.03, dt=1.0, n_steps=100)

rng = np.random.default_rng(seed=3)
meas1 = generate_radar_measurements(true1, range_std=1.0, bearing_std=0.01, rng=rng)
meas2 = generate_radar_measurements(true2, range_std=1.0, bearing_std=0.01, rng=rng)

track1 = Track(np.array([0.0, 50.0, 5.0, 0.0, 0.0]), np.diag([10.0,10.0,1.0,1.0,0.01]))
track2 = Track(np.array([0.0, -50.0, 5.0, 0.0, 0.0]), np.diag([10.0,10.0,1.0,1.0,0.01]))
tracks = [track1, track2]

correct_count = 0

for k in range(100):
    # combine both measurements into one pool, in RANDOM order (simulating no known labels)
    pool = np.array([meas1[k], meas2[k]])
    shuffle_idx = np.random.permutation(2)
    pool_shuffled = pool[shuffle_idx]
    # true_origin[i] tells us which track SHOULD match pool_shuffled[i]
    true_origin = np.argsort(shuffle_idx)  # inverse permutation

    for track in tracks:
        track.predict_step()

    assignments = associate_measurements(tracks, pool_shuffled)

    # check correctness: track 0 should be assigned measurement index true_origin[0], etc.
    for t_idx, m_idx in assignments.items():
        if m_idx == true_origin[t_idx]:
            correct_count += 1

    for t_idx, m_idx in assignments.items():
        tracks[t_idx].update_step(pool_shuffled[m_idx])

print(f"Correct associations: {correct_count} / 200")