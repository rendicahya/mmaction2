import sys

sys.path.append(".")

import pickle
from pathlib import Path

import cv2
import mmcv
import numpy as np
from assertpy.assertpy import assert_that
from config import settings as conf
from mmaction.apis import init_recognizer
from mmaction.utils import GradCAM
from moviepy.editor import ImageSequenceClip, clips_array
from vis_cam import _resize_frames, build_inputs

dataset = conf.active.dataset
video_dir = Path(conf.datasets[dataset].path)
action_list = [subdir.stem for subdir in sorted(video_dir.iterdir()) if subdir.is_dir()]

with open(conf.cam[dataset].A.dump, "rb") as file:
    dump_A = pickle.load(file)

with open(conf.cam[dataset].B.dump, "rb") as file:
    dump_B = pickle.load(file)

assert_that(len(dump_A) == len(dump_B))

model1 = init_recognizer(
    conf.cam[dataset].A.config, conf.cam[dataset].A.checkpoint, device="cuda"
)
model2 = init_recognizer(
    conf.cam[dataset].B.config, conf.cam[dataset].B.checkpoint, device="cuda"
)

with open(conf.cam[dataset].video_list, "r") as file:
    video_list = file.read().split("\n")

for i in range(len(dump_A)):
    item1 = dump_A[i]
    item2 = dump_B[i]

    item2_correct = item2["pred_label"] == item2["gt_label"]
    item1_wrong = item1["pred_label"] != item1["gt_label"]

    item1_pred = action_list[item1["pred_label"]]
    item2_pred = action_list[item2["pred_label"]]

    if item2_correct and item1_wrong:
        subpath, class_ = video_list[i].split()
        video_path = video_dir / subpath
        video_data = mmcv.VideoReader(str(video_path))
        fps = video_data.fps
        out_dir = Path(conf.cam[dataset].output.dir) / video_path.stem
        out_dir.mkdir(parents=True, exist_ok=True)

        open(out_dir / f"prediction-1: {item1_pred}", "a")
        open(out_dir / f"prediction-2: {item2_pred}", "a")

        w, h = video_data.resolution
        crop_size = min(w, h)
        left = (w - crop_size) // 2
        top = (h - crop_size) // 2
        right = left + crop_size
        bottom = top + crop_size
        collage = []
        models = model1, model1, model2
        alphas = 0, conf.cam.alpha, conf.cam.alpha
        vars_ = "original", "a", "b"

        for var, model, alpha in zip(vars_, models, alphas):
            inputs = build_inputs(model, video_path)
            gradcam = GradCAM(model, conf.cam.target_layer, conf.cam.colormap)
            results = gradcam(inputs, alpha=alpha)
            frames_batches = (results[0] * 255.0).numpy().astype(np.uint8)
            frames = frames_batches.reshape(-1, *frames_batches.shape[-3:])
            frame_list = list(frames)
            frame_list = _resize_frames(frame_list, conf.cam[dataset].output.resolution)

            frame_list_numbered = []

            for i, frame in enumerate(frame_list):
                bgr = mmcv.rgb2bgr(frame)
                numbered = cv2.putText(
                    frame,
                    str(i),
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.75,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA,
                )

                mmcv.imwrite(bgr, f"{out_dir}/{var}/{i}.jpg")
                frame_list_numbered.append(numbered)

            clip = ImageSequenceClip(frame_list_numbered, fps=fps)

            collage.append(clip)

        clips_array([collage]).write_videofile(f"{out_dir}/compare.mp4")
