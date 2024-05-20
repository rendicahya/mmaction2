import sys

sys.path.append(".")

import os.path as osp
import pickle
from pathlib import Path

import click
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
video_dir = Path(conf[dataset].path)

with open(conf.cam.A.dump, "rb") as file:
    dump1 = pickle.load(file)

with open(conf.cam.B.dump, "rb") as file:
    dump2 = pickle.load(file)

assert_that(len(dump1) == len(dump2))

model1 = init_recognizer(conf.cam.A.config, conf.cam.A.checkpoint, device="cuda")
model2 = init_recognizer(conf.cam.B.config, conf.cam.B.checkpoint, device="cuda")

with open(conf.cam.video_list, "r") as file:
    video_list = file.read().split("\n")

for i in range(len(dump1)):
    item1 = dump1[i]
    item2 = dump2[i]

    item2_correct = item2["pred_label"] == item2["gt_label"]
    item1_wrong = item1["pred_label"] != item1["gt_label"]

    if item2_correct and item1_wrong:
        subpath, class_ = video_list[i].split()
        video_path = video_dir / subpath
        video_data = mmcv.VideoReader(str(video_path))
        fps = video_data.fps
        collage = []

        out_dir = Path(conf.cam.output.dir) / video_path.stem
        out_dir.mkdir(parents=True, exist_ok=True)

        w, h = video_data.resolution
        crop_size = min(w, h)
        left = (w - crop_size) // 2
        top = (h - crop_size) // 2
        right = left + crop_size
        bottom = top + crop_size

        for i, frame in enumerate(video_data):
            crop = frame[top:bottom, left:right]
            mmcv.imwrite(crop, f"{out_dir}/0/{i}.jpg")

        for i, model in enumerate((model1, model2), 1):
            inputs = build_inputs(model, video_path)
            gradcam = GradCAM(model, conf.cam.target_layer, conf.cam.output.colormap)
            results = gradcam(inputs, alpha=conf.cam.output.alpha)
            frames_batches = (results[0] * 255.0).numpy().astype(np.uint8)
            frames = frames_batches.reshape(-1, *frames_batches.shape[-3:])
            frame_list = list(frames)
            frame_list = _resize_frames(frame_list, conf.cam.output.resolution)

            for j, frame in enumerate(frame_list):
                bgr = mmcv.rgb2bgr(frame)

                mmcv.imwrite(bgr, f"{out_dir}/{i}/{j}.jpg")

            frame_list_numbered = []

            for j, frame in enumerate(frame_list):
                frame_numbered = cv2.putText(
                    frame,
                    str(j),
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.75,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA,
                )

                frame_list_numbered.append(frame_numbered)

            clip = ImageSequenceClip(frame_list_numbered, fps=fps)

            collage.append(clip)
            clip.write_videofile(f"{out_dir}/{i}.mp4")

        clips_array([collage]).write_videofile(f"{out_dir}/compare.mp4")
