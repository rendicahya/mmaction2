from pathlib import Path
from random import random

from mmaction.registry import TRANSFORMS
from mmcv.transforms import BaseTransform


@TRANSFORMS.register_module()
class ActorCutMixCorruptDemo(BaseTransform):
    def __init__(self, corrupted_video_dir, class_index, mix_prob):
        self.corrupted_video_dir = Path(corrupted_video_dir)
        self.mix_prob = mix_prob

    def transform(self, results):
        if random() < self.mix_prob:
            file_path = Path(results["filename"])
            action = file_path.parent.name
            results["filename"] = str(
                self.corrupted_video_dir / action / file_path.with_suffix(".mp4").name
            )

        return results
