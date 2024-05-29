import json
import re
from collections import defaultdict
from pathlib import Path
from random import choice, random

import torch
from mmaction.registry import TRANSFORMS
from mmcv.transforms import BaseTransform


@TRANSFORMS.register_module()
class AttachMaskRatio(BaseTransform):
    def __init__(self, class_index):
        self.class_index = {}

        with open(class_index) as file:
            for line in file:
                id, action = line.split()
                self.class_index[action] = int(id)

    def transform(self, results):
        file_path = Path(results["filename"])
        action_video, _, scene_label = file_path.stem.rpartition("-")

        results["scene_label"] = self.class_index[scene_label]

        return results


@TRANSFORMS.register_module()
class ActorCutMix(BaseTransform):
    def __init__(
        self, video_dir, class_index, mix_prob, mask_ratio_file, min_mask_ratio
    ):
        self.video_dir = Path(video_dir)
        self.mix_prob = mix_prob
        self.min_mask_ratio = min_mask_ratio
        self.video_list = defaultdict(list)
        self.class_index = {}

        with open(class_index) as file:
            for line in file:
                id, action = line.split()
                self.class_index[action] = int(id) - 1

        with open(mask_ratio_file) as file:
            self.mask_ratio = json.load(file)

        with open(self.video_dir / "list.txt") as file:
            for line in file:
                path, class_ = line.split()
                action, filename = path.split("/")
                action_video, _, scene_label = filename.rpartition("-")

                self.video_list[action_video].append(path)

    def transform(self, results):
        file_path = Path(results["filename"])

        if random() < self.mix_prob:
            mask_ratio = self.mask_ratio[file_path.stem]
            results["mask_ratio"] = mask_ratio

            if mask_ratio < self.min_mask_ratio:
                return results

            options = self.video_list[file_path.stem]
            video_pick = self.video_dir / choice(options)
            action_video, _, scene_label = video_pick.stem.rpartition("-")
            scene_id = self.class_index[scene_label]
            # scene_id = torch.tensor(scene_id)#.reshape((1,))

            results["filename"] = str(video_pick)
            results["scene_label"] = scene_id

        return results


@TRANSFORMS.register_module()
class InterCutMix(BaseTransform):

    def __init__(self, video_dir, mix_prob=0.5, min_mask_ratio=0.0):
        self.video_dir = Path(video_dir)
        self.mix_prob = mix_prob
        self.min_mask_ratio = min_mask_ratio
        self.video_list = defaultdict(list)

        with open(self.video_dir / "list.txt") as file:
            for line in file:
                path, class_ = line.split()
                action, filename = path.split("/")
                action_video, _, scene_label = filename.rpartition("-")

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


@TRANSFORMS.register_module()
class InterCutMixIncrProb(BaseTransform):
    def __init__(
        self, train_list, mixed_video_dir, mix_prob, max_epoch, min_mask_ratio
    ):
        self.mixed_video_dir = Path(mixed_video_dir)
        self.mix_prob = mix_prob
        self.max_epoch = max_epoch
        self.min_mask_ratio = min_mask_ratio
        self.video_list = defaultdict(list)
        self.video_count = 0

        with open(train_list, "rb") as file:
            self.len_train = sum(1 for _ in file)

        with open(self.mixed_video_dir / "list.txt") as file:
            for line in file:
                path, class_ = line.split()
                action, filename = path.split("/")
                action_video, _, scene_label = filename.rpartition("-")

                self.video_list[action_video].append(path)

        relevancy_thresh = self.mixed_video_dir
        relevancy_model = self.mixed_video_dir.parent
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
            video_pick = self.mixed_video_dir / choice(options)
            results["filename"] = str(video_pick)

        return results
