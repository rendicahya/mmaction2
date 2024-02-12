from pathlib import Path
from random import choice, random

from mmaction.registry import TRANSFORMS
from mmcv.transforms import BaseTransform


@TRANSFORMS.register_module()
class AccessEpoch(BaseTransform):
    def __init__(self, total):
        self.total = total

    def transform(self, results):
        if random() > self.prob:
            filepath = Path(results["filename"])
            action = filepath.parent.name
            action_dir = self.acm_root / action
            options = [
                f for f in action_dir.iterdir() if str(f.stem).startswith(filepath.stem)
            ]

            video_pick = choice(options)
            results["filename"] = str(video_pick)

        return results
