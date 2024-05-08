import json
import re
from collections import defaultdict
from pathlib import Path
from random import choice, random

from mmaction.registry import TRANSFORMS
from mmcv.transforms import BaseTransform


@TRANSFORMS.register_module()
class InterCutMix(BaseTransform):
    def __init__(self, video_dir, mix_prob=0.5, min_mask_ratio=0.0):
        self.video_dir = Path(video_dir)
        self.mix_prob = mix_prob
        self.min_mask_ratio = min_mask_ratio
        self.video_list = defaultdict(list)
        path_splitter = re.compile(r"[/-]")

        with open(self.video_dir / "list.txt") as file:
            for line in file:
                path, class_ = line.split()
                action, action_video, scene_class = path_splitter.split(path)

                self.video_list[action_video].append(path)

        relevancy_thresh = self.video_dir
        relevancy_model = self.video_dir.parent
        mask_dir = relevancy_model.parent.parent / "mask"
        file_ratio_path = (
            mask_dir / relevancy_model.name / relevancy_thresh.name / "ratio.json"
        )

        with open(file_ratio_path) as f:
            self.mask_ratio = json.load(f)

    def transform(self, results):
        if random() < self.mix_prob:
            file_path = Path(results["filename"])

            if self.mask_ratio[file_path.stem] < self.min_mask_ratio:
                return results

            options = self.video_list[file_path.stem]
            video_pick = self.video_dir / choice(options)
            results["filename"] = str(video_pick)

        return results
