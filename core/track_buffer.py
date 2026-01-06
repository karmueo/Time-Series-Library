import time
from collections import deque

import numpy as np


class TrackWindowBuffer:
    def __init__(self, seq_len, max_age_s=10.0):
        self.seq_len = seq_len
        self.max_age_s = max_age_s
        self.buffers = {}
        self.last_seen = {}
        self.last_timestamp = {}
        self.since_last_infer = {}
        self.has_inferred = {}

    def update(self, track_id, feature_vec, timestamp_s=None):
        if track_id not in self.buffers:
            self.buffers[track_id] = deque(maxlen=self.seq_len)
            self.since_last_infer[track_id] = 0
            self.has_inferred[track_id] = False
        self.buffers[track_id].append(feature_vec)
        self.last_seen[track_id] = time.time()
        self.since_last_infer[track_id] += 1
        if timestamp_s is not None:
            self.last_timestamp[track_id] = timestamp_s

    def cleanup(self):
        now = time.time()
        expired = [tid for tid, ts in self.last_seen.items() if now - ts > self.max_age_s]
        for tid in expired:
            self.buffers.pop(tid, None)
            self.last_seen.pop(tid, None)
            self.last_timestamp.pop(tid, None)
            self.since_last_infer.pop(tid, None)
            self.has_inferred.pop(tid, None)

    def build_batch(self, min_seq_len=1, window_step=0):
        track_ids = []
        sequences = []
        lengths = []
        for tid, buf in self.buffers.items():
            length = len(buf)
            if length < min_seq_len:
                continue
            if window_step > 0 and self.has_inferred.get(tid, False):
                if self.since_last_infer.get(tid, 0) < window_step:
                    continue
            seq = np.stack(list(buf), axis=0)
            if length > self.seq_len:
                seq = seq[-self.seq_len:, :]
                length = self.seq_len
            if length < self.seq_len:
                pad = np.zeros((self.seq_len - length, seq.shape[1]), dtype=seq.dtype)
                seq = np.concatenate([seq, pad], axis=0)
            track_ids.append(tid)
            sequences.append(seq)
            lengths.append(length)
        if not sequences:
            return [], None, None
        batch = np.stack(sequences, axis=0)
        lengths = np.asarray(lengths, dtype=np.int16)
        return track_ids, batch, lengths

    def mark_inferred(self, track_ids):
        for tid in track_ids:
            if tid in self.since_last_infer:
                self.since_last_infer[tid] = 0
                self.has_inferred[tid] = True

    def get_pending_progress(self, min_seq_len=1, window_step=0):
        progress = {}
        for tid, buf in self.buffers.items():
            length = len(buf)
            if not self.has_inferred.get(tid, False):
                if length < min_seq_len:
                    progress[tid] = (length, min_seq_len)
            elif window_step > 0:
                delta = self.since_last_infer.get(tid, 0)
                if delta < window_step:
                    progress[tid] = (delta, window_step)
        return progress

    def get_last_timestamp(self, track_id):
        return self.last_timestamp.get(track_id)
