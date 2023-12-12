from pathlib import Path
from random import choice, random

from mmaction.registry import TRANSFORMS
from mmcv.transforms import BaseTransform


@TRANSFORMS.register_module()
class ActorCutMix(BaseTransform):
    def __init__(self, root, prob):
        self.acm_root = Path(root)
        self.prob = prob

    def transform(self, results):
        if random() > self.prob:
            action, file = results["filename"].split("/")[-2:]
            action_dir = self.acm_root / action
            file_stem = file.split(".")[0]

            options = [
                file
                for file in action_dir.iterdir()
                if str(file.stem).startswith(file_stem)
            ]

            pick = choice(options)
            results["filename"] = str(pick)

        return results
