import re
from collections import defaultdict
from pathlib import Path
from random import choice, random

from mmaction.registry import TRANSFORMS
from mmcv.transforms import BaseTransform


@TRANSFORMS.register_module()
class Mix2Train(BaseTransform):
    def __init__(self, mixed_video_dir, mix_prob=0.5):
        self.mixed_video_dir = Path(mixed_video_dir)
        self.mix_prob = mix_prob
        self.video_list = defaultdict(list)
        path_splitter = re.compile(r"[/-]")

        with open(self.mixed_video_dir / "list.txt") as file:
            for line in file:
                path, class_id = line.split()
                label, action_video, scene_video = path_splitter.split(path)

                self.video_list[action_video].append(path)

    def transform(self, results):
        if random() < self.mix_prob:
            file_path = Path(results["filename"])

            if not file_path.stem in self.video_list:
                return results

            options = self.video_list[file_path.stem]
            video_pick = self.mixed_video_dir / choice(options)
            results["filename"] = str(video_pick)

        return results
