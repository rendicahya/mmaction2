import json
from collections import defaultdict
from pathlib import Path
from random import choice, random

from mmaction.registry import TRANSFORMS
from mmcv.transforms import BaseTransform


@TRANSFORMS.register_module()
class ActorCutMix(BaseTransform):
    def __init__(self, mix_video_dir, class_index, mix_prob, min_mask_ratio):
        self.mix_video_dir = Path(mix_video_dir)
        self.mix_prob = mix_prob
        self.min_mask_ratio = min_mask_ratio
        self.video_list = defaultdict(list)
        self.class_index = {}

        mask_ratio_file = self.mix_video_dir.parent / "mask/ratio.json"

        with open(class_index) as file:
            for line in file:
                id, action = line.split()
                self.class_index[action] = int(id) - 1

        with open(mask_ratio_file) as file:
            self.mask_ratio = json.load(file)

        with open(self.mix_video_dir / "list.txt") as file:
            for line in file:
                path, class_ = line.split()
                action, filename = path.split("/")
                action_video, _, scene_label = filename.rpartition("-")

                self.video_list[action_video].append(path)

    def transform(self, results):
        if random() < self.mix_prob:
            file_path = Path(results["filename"])
            mask_ratio = self.mask_ratio[file_path.stem]
            results["mask_ratio"] = mask_ratio

            if mask_ratio < self.min_mask_ratio:
                return results

            options = self.video_list[file_path.stem]
            video_pick = self.mix_video_dir / choice(options)
            action_video, _, scene_label = video_pick.stem.rpartition("-")
            scene_id = self.class_index[scene_label]

            results["filename"] = str(video_pick)
            results["scene_label"] = scene_id

        return results
