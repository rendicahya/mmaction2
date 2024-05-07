import json
import os
from pathlib import Path
from random import choice, random

from mmaction.registry import TRANSFORMS
from mmcv.transforms import BaseTransform


@TRANSFORMS.register_module()
class ActorCutMix(BaseTransform):
    def __init__(self, video_dir, mix_prob=0.5, min_mask_ratio=0.0):
        self.video_dir = Path(video_dir)
        self.mix_prob = mix_prob
        self.min_mask_ratio = min_mask_ratio

        video_list_path = self.video_dir / "list.json"
        mask_ratio_path = self.video_dir.parent / "mask/ratio.json"

        with open(video_list_path) as f:
            self.video_list = json.load(f)

        with open(mask_ratio_path) as f:
            self.mask_ratio = json.load(f)

    def transform(self, results):
        if random() < self.mix_prob:
            file_path = Path(results["filename"])

            if self.mask_ratio[file_path.stem] < self.min_mask_ratio:
                return results

            action = file_path.parent.name
            options = [f for f in self.video_list[action] if file_path.stem in f]
            video_pick = self.video_dir / choice(options)
            results["filename"] = str(video_pick)

        return results
