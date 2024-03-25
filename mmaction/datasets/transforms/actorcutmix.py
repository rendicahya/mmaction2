import json
from pathlib import Path
from random import choice, random
import os
from mmaction.registry import TRANSFORMS
from mmcv.transforms import BaseTransform


@TRANSFORMS.register_module()
class ActorCutMix(BaseTransform):
    def __init__(self, root, file_list, prob=0.5, min_mask_ratio=0.0):
        self.acm_root = Path(root)
        self.prob = prob
        self.min_mask_ratio = min_mask_ratio
        file_list_path = Path(file_list)
        relevancy_thresh = file_list_path.parent
        relevancy_model = relevancy_thresh.parent
        mask_dir = relevancy_model.parent.parent / "mask"
        file_ratio_path = (
            mask_dir / relevancy_model.name / relevancy_thresh.name / "ratio.json"
        )

        with open(file_list_path) as f:
            self.file_list = json.load(f)

        with open(file_ratio_path) as f:
            self.mask_ratio = json.load(f)

    def transform(self, results):
        if random() < self.prob:
            file_path = Path(results["filename"])

            if self.mask_ratio[file_path.stem] < self.min_mask_ratio:
                return results

            action = file_path.parent.name
            options = [f for f in self.file_list[action] if file_path.stem in f]
            video_pick = self.acm_root / choice(options)
            results["filename"] = str(video_pick)

        return results
