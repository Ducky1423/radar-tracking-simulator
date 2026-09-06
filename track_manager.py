import numpy as np
from track import Track

class TrackManager:
    def __init__(self, max_misses=5):
        self.tracks = []
        self.miss_counts = []  # parallel list: consecutive misses per track
        self.max_misses = max_misses

    def step(self, measurements, associate_fn):
        # 1. Predict all existing tracks
        for track in self.tracks:
            track.predict_step()

        # 2. Associate measurements to tracks
        assignments = associate_fn(self.tracks, measurements) if self.tracks else {}

        # 3. Update matched tracks, reset their miss count
        matched_meas_idxs = set()
        for t_idx, m_idx in assignments.items():
            self.tracks[t_idx].update_step(measurements[m_idx])
            self.miss_counts[t_idx] = 0
            matched_meas_idxs.add(m_idx)

        # 4. Increment miss count for unmatched tracks
        for t_idx in range(len(self.tracks)):
            if t_idx not in assignments:
                self.miss_counts[t_idx] += 1

        # 5. Spawn new tracks from unmatched measurements
        for m_idx, z in enumerate(measurements):
            if m_idx not in matched_meas_idxs:
                new_x = np.array([z[0]*np.cos(z[1]), z[0]*np.sin(z[1]), 0.0, 0.0, 0.0])
                new_P = np.diag([10.0, 10.0, 5.0, 5.0, 0.1])
                self.tracks.append(Track(new_x, new_P))
                self.miss_counts.append(0)

        # 6. Drop stale tracks (too many consecutive misses)
        keep = [i for i in range(len(self.tracks)) if self.miss_counts[i] < self.max_misses]
        self.tracks = [self.tracks[i] for i in keep]
        self.miss_counts = [self.miss_counts[i] for i in keep]