from pathlib import Path
from random import choice, random

from mmaction.registry import TRANSFORMS
from mmcv.transforms import BaseTransform


@TRANSFORMS.register_module()
class AccessEpoch(BaseTransform):
    def __init__(self, total):
        self.total = total
        self.count = 0

    def transform(self, results):
        if self.count == self.total:
            print("===== EPOCH", self.count / self.total, "=====")
            self.count = 0

        self.count += 1

        return results
