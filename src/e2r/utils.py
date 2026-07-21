import numpy as np
import torch


class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = self.avg = self.sum = self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class Cacher:
    def __init__(self, n_sample, n_dim, cache_file):
        self.cache_file = cache_file
        self.feature = torch.zeros(n_sample, n_dim)

    def load_batch(self, indices, features):
        self.feature[indices.cpu()] = features.detach().cpu()

    def save(self):
        np.save(self.cache_file, self.feature.numpy())
