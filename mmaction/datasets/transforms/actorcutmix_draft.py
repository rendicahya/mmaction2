import json
from pathlib import Path
from random import choice, random

from mmaction.registry import TRANSFORMS
from mmcv.transforms import BaseTransform


@TRANSFORMS.register_module()
class ActorCutMixDraft(BaseTransform):
    def __init__(self, root, file_list, prob):
        print('ACTORCUTMIX DRAFT')
        self.acm_root = Path(root)
        self.prob = prob

        with open(file_list) as f:
            self.file_list = json.load(f)

    def transform(self, results):
        if random() > self.prob:
            file_path = Path(results["filename"])
            action = file_path.parent.name
            options = [f for f in self.file_list[action] if file_path.stem in f]
            video_pick = self.acm_root / choice(options)
            results["filename"] = str(video_pick)

        return results
