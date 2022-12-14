import numpy as np
import random


class IndexPool:
    def __init__(self, capacity=20, priority=True, y_batch=None, idx_batch=None):
        self.capacity = capacity
        self.priority = priority
        if idx_batch is not None and idx_batch.size <= self.capacity:
            self.y_batch, self.idx_batch = y_batch.reshape(-1), idx_batch.reshape(-1)
        else:
            self.y_batch, self.idx_batch = None, None
        if self.y_batch is not None:
            self.n_pos, self.n_neg = np.count_nonzero(self.y_batch > 0), np.count_nonzero(self.y_batch < 0)
        else:
            self.n_neg, self.n_pos = None, None

    @property
    def gini(self):
        assert self.n_pos >= 0 and self.n_neg >= 0
        pos_ratio = self.n_pos / (self.n_pos + self.n_neg)
        neg_ratio = self.n_neg / (self.n_pos + self.n_neg)
        return 1 - pos_ratio ** 2 - neg_ratio ** 2

    @property
    def occupation(self):
        assert self.n_pos >= 0 and self.n_neg >= 0
        return self.n_pos + self.n_neg

    def clear(self):
        self.idx_batch, self.y_batch = None, None
        self.n_pos, self.n_neg = None, None

    def push(self, y, idx):
        if self.idx_batch is None:
            self.idx_batch, self.y_batch = idx.reshape(-1), y.reshape(-1)
            self.n_pos, self.n_neg = 0, 0
        elif self.idx_batch.size < self.capacity:
            self.idx_batch, self.y_batch = np.concatenate((self.idx_batch, idx.reshape(-1)), 0), np.concatenate((self.y_batch, y.reshape(-1)), 0)
        else:
            if self.priority:
                if self.n_pos > self.n_neg:
                    candidates = np.argwhere(self.y_batch > 0).reshape(-1)
                    victim = candidates[random.randint(0, candidates.size - 1)]
                    self.idx_batch[victim], self.y_batch[victim] = idx, y
                    self.n_pos -= 1
                else:
                    candidates = np.argwhere(self.y_batch < 0).reshape(-1)
                    victim = candidates[random.randint(0, candidates.size - 1)]
                    self.idx_batch[victim], self.y_batch[victim] = idx, y
                    self.n_neg -= 1
            else:
                victim = random.randint(0, self.capacity - 1)
                self.idx_batch[victim] = idx
                y_temp = self.y_batch[victim]
                self.y_batch[victim] = y
                if y_temp > 0:
                    self.n_pos -= 1
                else:
                    self.n_neg -= 1
        if y > 0:
            self.n_pos += 1
        else:
            self.n_neg += 1
