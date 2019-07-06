import math
import numpy as np
from torch.optim import lr_scheduler


class CosineAnnealingLR(lr_scheduler._LRScheduler):
    def __init__(self, optimizer, T_max, T_mul=1.0, decay_rate=1.0, T_min=0, last_epoch=-1):
        '''
        :param optimizer:
        :param T_max(float):Maximum number of iterations
        :param T_mul(float): Increase T_max by a factor of T_mul
        :param decay_rate:
        :param T_min:
        :param last_epoch:
        '''
        self.T_max = T_max
        self.T_mul = T_mul
        self.decay_rate = decay_rate
        self.eta_min = T_min
        self.cut_points = [0]
        super().__init__(optimizer, last_epoch)

    def _generate_cut_list(self):
        n = len(self.cut_points) * 2
        val = 0
        seq = [val]
        for i in range(n):
            val += self.T_max * (self.T_mul ** i)
            seq.append(val)
        self.cut_points = seq

    @staticmethod
    def _find_range(cut_list, n):
        for i in range(len(cut_list) - 1):
            left = cut_list[i]
            right = cut_list[i + 1]
            if left <= n < right:
                return left, right, i
        raise Exception('error')

    def get_lr(self):
        last_epoch = self.last_epoch
        if last_epoch >= np.max(self.cut_points):
            self._generate_cut_list()
        left, right, n_peak = self._find_range(self.cut_points, last_epoch)

        last_epoch -= left
        decay = self.decay_rate ** n_peak

        return [self.eta_min + max(base_lr * decay - self.eta_min, 0) *
                (1 + math.cos(math.pi * last_epoch / (right - left))) / 2
                for base_lr in self.base_lrs]
