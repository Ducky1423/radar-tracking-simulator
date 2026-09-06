import numpy as np
import matplotlib.pyplot as plt
from ground_truth import generate_ct_trajectory
from sensor import generate_radar_measurements, polar_to_cartesian
from track import Track

true1 = generate_ct_trajectory(x0=[0, 50, 5, 0], omega=0.05, dt=1.0, n_steps=100)
true2 = generate_ct_trajectory(x0=[0, -50, 5, 0], omega=-0.03, dt=1.0, n_steps=100)

rng = np.random.default_rng(seed=2)
meas1 = generate_radar_measurements(true1, range_std=1.0, bearing_std=0.01, rng=rng)
meas2 = generate_radar_measurements(true2, range_std=1.0, bearing_std=0.01, rng=rng)

track1 = Track(np.array([0.0, 50.0, 5.0, 0.0, 0.0]), np.diag([10.0,10.0,1.0,1.0,0.01]))
track2 = Track(np.array([0.0, -50.0, 5.0, 0.0, 0.0]), np.diag([10.0,10.0,1.0,1.0,0.01]))

est1 = np.zeros((100,5))
est2 = np.zeros((100,5))

for k in range(100):
    track1.predict_step()
    track1.update_step(meas1[k])
    est1[k] = track1.x

    track2.predict_step()
    track2.update_step(meas2[k])
    est2[k] = track2.x

plt.figure(figsize=(8,6))
plt.plot(true1[:,0], true1[:,1], 'b', label='True 1')
plt.plot(est1[:,0], est1[:,1], 'g--', label='UKF 1')
plt.plot(true2[:,0], true2[:,1], 'orange', label='True 2')
plt.plot(est2[:,0], est2[:,1], 'r--', label='UKF 2')
plt.legend()
plt.axis('equal')
plt.grid(True)
plt.title('Independent Multi-Target Tracking (known association)')
plt.savefig('tracking_demo.png')