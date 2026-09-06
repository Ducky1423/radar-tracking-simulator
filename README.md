# Multi-Target Radar Tracking Simulator

A from-scratch implementation of a radar tracking pipeline: simulating maneuvering
targets, noisy sensor measurements, an Unscented Kalman Filter (UKF) for state
estimation, and multi-target data association / track management.

## What it does
- Simulates ground-truth target trajectories under a constant turn-rate motion model
- Simulates a noisy radar sensor (range/bearing measurements)
- Implements an Unscented Kalman Filter from scratch (sigma points, predict/update)
- Extends to multiple simultaneous targets via a `Track` class
- Implements Global Nearest Neighbor data association (Mahalanobis distance)
- Implements track management: spawning new tracks, dropping stale ones
- Validates filter accuracy via RMSE and statistical consistency via NEES

## Results
- UKF reduced tracking RMSE by ~41% vs. raw noisy measurements (1.972 -> 1.166)
- Data association achieved 200/200 correct assignments on well-separated targets
- Track manager correctly spawned/dropped tracks as targets appeared
- NEES analysis identified the filter as statistically underconfident (avg ~1.25 vs.
  ideal ~4.0), likely due to conservative covariance growth in the UKF update step —
  noted as a tuning target for future work

## Known limitations / future work
- Data association uses greedy Global Nearest Neighbor; a Hungarian-algorithm or
  JPDA approach would handle closely-spaced/crossing targets more robustly
- NEES calibration could be improved with more rigorous Q/R tuning
- Motion model assumes constant turn rate; could be extended to a multi-model
  (IMM) approach for more realistic maneuvering targets

## Project structure
- `ground_truth.py` — trajectory simulation
- `sensor.py` — radar measurement simulation
- `motion_model.py` — UKF prediction model
- `ukf.py` — Unscented Kalman Filter (sigma points, predict, update)
- `track.py` — per-target Track class
- `data_association.py` — Global Nearest Neighbor
- `track_manager.py` — multi-target lifecycle management
- `test_*.py` — validation scripts for each component