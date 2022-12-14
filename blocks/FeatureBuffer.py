import numpy as np
import random


class BufferLine:
    def __init__(self, x=None, y=None, idx=None):
        self.x = x
        self.y = y
        self.idx = idx


class FeatureBuffer:
    def __init__(self, capacity=20):
        self.capacity = capacity
        self.lines = []

    @property
    def occupation(self):
        return len(self.lines)

    def clear(self):
        self.lines = []

    def get_victim(self):
        return random.randint(0, self.capacity - 1)

    def push(self, x, y, idx):
        if self.occupation < self.capacity:
            self.lines.append(BufferLine(x=x, y=y, idx=idx))
        else:
            victim = self.get_victim()
            self.lines[victim] = BufferLine(x=x, y=y, idx=idx)

    def search(self, wanted_indices):
        idx_batch = np.concatenate([line.idx for line in self.lines], 0)
        return [int(np.argwhere(idx_batch == idx)) for idx in wanted_indices if idx in idx_batch]

    def get(self, idx_batch):
        if local_indices := self.search(idx_batch):
            x_target = np.concatenate([self.lines[idx].x for idx in local_indices])
            y_target = np.concatenate([self.lines[idx].y for idx in local_indices])
            idx_target = np.concatenate([self.lines[idx].idx for idx in local_indices])
            return x_target, y_target, idx_target
        else:
            return None, None, None

    def sample(self):
        local_idx = random.randint(0, self.occupation - 1)
        return self.lines[local_idx].x, self.lines[local_idx].y, self.lines[local_idx].idx


class FIFOFeatureBuffer(FeatureBuffer):
    def push(self, x, y, idx):
        if self.occupation < self.capacity:
            self.lines.append(BufferLine(x=x, y=y, idx=idx))
        else:
            self.lines.pop(0)
            self.lines.append(BufferLine(x=x, y=y, idx=idx))


class ClassWiseFeatureBuffer:
    def __init__(self, n_classes=8, pos_class=None, block_capacity=20):
        self.n_classes = n_classes
        self.pos_class = pos_class
        self.block_capacity = block_capacity
        self.block_buffers = [FIFOFeatureBuffer(self.block_capacity) for _ in range(self.n_classes)]

    def __getitem__(self, item):
        return self.block_buffers[item]

    @property
    def classwise_occupation(self):
        return [buffer.occupation for buffer in self.block_buffers]

    @property
    def occupation(self):
        return sum(self.classwise_occupation)

    def binarize(self, y_batch):
        if self.pos_class is None:
            return y_batch
        else:
            y_bin = y_batch.copy()
            y_bin[y_bin != self.pos_class] = -1.0
            y_bin[y_bin == self.pos_class] = 1.0
            return y_bin

    def clear(self):
        for buffer in self.block_buffers:
            buffer.clear()

    def push(self, x, y, idx):
        assert y in range(self.n_classes)
        self.block_buffers[int(y)].push(x, y, idx)

    def get(self, idx_batch):
        triplet_list = [buffer.get(idx_batch) for buffer in self.block_buffers]
        x_target = np.concatenate([triplet[0] for triplet in triplet_list if triplet[0] is not None], 0)
        y_target = self.binarize(np.concatenate([triplet[1] for triplet in triplet_list if triplet[1] is not None], 0))
        idx_target = np.concatenate([triplet[2] for triplet in triplet_list if triplet[2] is not None], 0)
        if np.all(y_target > 0) or np.all(y_target < 0):
            return None, None, None
        else:
            return x_target, y_target, idx_target

    def sample(self):
        assert self.pos_class is not None
        if random.random() <= 0.5:
            x_s, _, idx_s = self.block_buffers[self.pos_class].sample()
            return x_s, np.array([1.0]), idx_s
        else:
            rand_int = random.sample([i for i in range(self.n_classes) if i != self.pos_class], 1)[0]
            x_s, _, idx_s = self.block_buffers[rand_int].sample()
            return x_s, np.array([-1.0]), idx_s
