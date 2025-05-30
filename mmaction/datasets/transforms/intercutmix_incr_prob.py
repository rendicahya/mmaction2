import json
from collections import defaultdict
from pathlib import Path
from random import choice, random

from mmaction.registry import TRANSFORMS
from mmcv.transforms import BaseTransform


@TRANSFORMS.register_module()
class InterCutMixIncrProb(BaseTransform):
    def __init__(self, train_list, mix_video_dir, mix_prob, max_epoch, min_mask_ratio):
        self.mix_video_dir = Path(mix_video_dir)
        self.mix_prob = mix_prob
        self.max_epoch = max_epoch
        self.min_mask_ratio = min_mask_ratio
        self.video_list = defaultdict(list)
        self.video_count = 0

        with open(train_list, "rb") as file:
            self.len_train = sum(1 for _ in file)

        with open(self.mix_video_dir / "list.txt") as file:
            for line in file:
                path, class_ = line.split()
                action, filename = path.split("/")
                action_video, _, scene_label = filename.rpartition("-")

                self.video_list[action_video].append(path)

        relevancy_thresh = self.mix_video_dir
        relevancy_model = self.mix_video_dir.parent
        mask_dir = relevancy_model.parent.parent / "mask"
        file_ratio_path = (
            mask_dir / relevancy_model.name / relevancy_thresh.name / "ratio.json"
        )

        with open(file_ratio_path) as f:
            self.mask_ratio = json.load(f)

    def transform(self, results):
        self.video_count += 1
        epoch = self.video_count // self.len_train
        mix_prob = self.mix_prob[0] + (epoch / self.max_epoch) * (
            self.mix_prob[1] - self.mix_prob[0]
        )

        if random() < mix_prob:
            file_path = Path(results["filename"])

            if self.mask_ratio[file_path.stem] < self.min_mask_ratio:
                return results

            options = self.video_list[file_path.stem]
            video_pick = self.mix_video_dir / choice(options)
            results["filename"] = str(video_pick)

        return results
