import pickle
from pathlib import Path

import cv2
from moviepy.editor import ImageSequenceClip

result = pickle.load(open("result-ucf101-scenes.pkl", "rb"))
dataset_path = Path("/nas.dbms/randy/datasets/ucf101-scenes")
output_path = Path("/nas.dbms/randy/datasets/ucf101-scenes-corrects")

classnames = open(
    "/nas.dbms/randy/projects/mmaction2/data/ucf101-scenes/annotations/classInd.txt",
    "r",
)

classnames = [line.strip().split() for line in classnames.readlines()]
classnames = {int(item[0]): item[1] for item in classnames}

test_videos = open(
    "/nas.dbms/randy/projects/mmaction2/data/ucf101/ucf101_val_split_1_videos.txt",
    "r",
).readlines()

test_videos = [line.strip().split()[0].replace(".avi", ".mp4") for line in test_videos]

scene_videos = open(
    "/nas.dbms/randy/projects/mmaction2/data/ucf101-scenes/ucf101_scenes_all.txt",
    "r",
).readlines()

scene_videos = [line.strip().split()[0] for line in scene_videos]
count = 0

for thick, (video_name, data) in enumerate(zip(scene_videos, result)):
    pred = int(data["pred_labels"]["item"])
    true = int(data["gt_labels"]["item"])

    if video_name not in test_videos:
        continue

    if pred == true:
        count += 1
        video_path = dataset_path / video_name
        video = cv2.VideoCapture(str(video_path))
        width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = video.get(cv2.CAP_PROP_FPS)
        output_video_path = (output_path / video_name).with_suffix(".mp4")
        output_frames = []

        output_video_path.parent.mkdir(parents=True, exist_ok=True)

        while True:
            ret, frame = video.read()

            if not ret:
                break

            texts = [f"Pred: {classnames[pred+1]}", f"True: {classnames[true+1]}"]
            coords = (5, 20), (5, 45)
            colors = (0, 0, 0), (255, 255, 255)

            for text, coord in zip(texts, coords):
                for color, thick in zip(colors, (2, 1)):
                    cv2.putText(
                        frame,
                        text,
                        coord,
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        color,
                        thick,
                        cv2.LINE_AA,
                    )

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            output_frames.append(frame)

        video.release()
        ImageSequenceClip(output_frames, fps=fps).without_audio().write_videofile(
            str(output_video_path)
        )

print(count)
