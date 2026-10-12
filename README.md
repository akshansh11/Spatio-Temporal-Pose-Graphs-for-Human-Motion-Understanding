# Spatio-Temporal Pose Graphs for Human Motion Understanding

Turning raw video of a multi-person performance into a frame-synchronized sequence of skeleton graphs, where every body joint is a node and every bone is an edge.

## Overview

Human motion is usually stored as pixels, which mix the moving body with lighting, costumes and background. This project strips all of that away. Each person in each frame is reduced to a graph of 17 joints, rendered on a black canvas with the same resolution, frame rate and length as the source video.

The resulting graph sequence is a compact, structured representation of motion. It can be analyzed directly or fed into graph neural networks such as ST-GCN for recognition, comparison and generation tasks.

<img width="1280" height="720" alt="pose" src="https://github.com/user-attachments/assets/30e10d1c-dd1e-4bd1-9ddc-9121e19ae1d6" />


## Pipeline

```
Input video  ──►  YOLOv8m-Pose  ──►  17 keypoints per person  ──►  Skeleton graph  ──►  Output video
(RGB frames)      (per frame)        (x, y, confidence)            (nodes + edges)      (black background)
```

1. **Frame extraction.** The video is read frame by frame with OpenCV.
2. **Pose estimation.** YOLOv8m-Pose detects every person and predicts 17 COCO keypoints for each.
3. **Filtering.** People below 0.35 detection confidence and joints below 0.5 keypoint confidence are discarded.
4. **Graph construction.** Joints become nodes and anatomical links become edges.
5. **Rendering.** Graphs are drawn on a black frame, with line width scaled to each person's size.
6. **Synchronization.** Exactly one output frame is written per input frame, so timing matches the source.

## Graph Representation

Each person in each frame forms a spatial graph G = (V, E).

| Node ID | Joint | Node ID | Joint |
|---|---|---|---|
| 0 | Nose | 9 / 10 | Left / Right wrist |
| 1 / 2 | Left / Right eye | 11 / 12 | Left / Right hip |
| 3 / 4 | Left / Right ear | 13 / 14 | Left / Right knee |
| 5 / 6 | Left / Right shoulder | 15 / 16 | Left / Right ankle |
| 7 / 8 | Left / Right elbow | | |

Edges connect shoulders to elbows to wrists, hips to knees to ankles, the shoulder and hip girdles, the torso sides, and the head to the neck. Linking the same joint across consecutive frames adds temporal edges, which turns the sequence into a spatio-temporal graph.

**Color code**

- 🔵 Blue: left side of the body
- 🟠 Orange: right side of the body
- ⚪ White: torso, neck and head

## Results

| Property | Input | Output |
|---|---|---|
| Resolution | 1920 × 1080 | 1920 × 1080 |
| Frame rate | 30 fps | 30 fps |
| Frames | 742 | 742 |
| Duration | 24.73 s | 24.73 s |
| People per frame | up to 14 | up to 14 skeletons |

Two output videos are produced: one with skeletons only, and one that also carries the original audio track.

## Installation

```bash
pip install ultralytics opencv-python
```

The YOLOv8m-Pose weights download automatically on the first run. `ffmpeg` is optional and only needed for re-encoding and adding audio.

## Usage

```bash
python skeleton_multi.py input.mp4 skeleton_raw.mp4
```

Re-encode for wide compatibility and keep the original audio:

```bash
ffmpeg -i skeleton_raw.mp4 -i input.mp4 -map 0:v -map 1:a? \
       -c:v libx264 -pix_fmt yuv420p -c:a aac -shortest skeleton_final.mp4
```

### Tuning

| Parameter | Default | Effect |
|---|---|---|
| Model | `yolov8m-pose.pt` | Use `yolov8n-pose.pt` for speed or `yolov8x-pose.pt` for accuracy |
| `conf` | 0.35 | Raise it to drop faint or distant people |
| Keypoint threshold | 0.5 | Raise it for cleaner but more incomplete skeletons |

## Project Structure

```
├── skeleton_multi.py               # Pose extraction and graph rendering
├── dance3_skeleton.mp4             # Output, skeletons only
├── dance3_skeleton_with_audio.mp4  # Output with original audio
└── README.md
```

## Limitations

- Heavily overlapping or partly out-of-frame people may get incomplete skeletons or flicker for a few frames.
- Detection is per frame with no identity tracking, so a person's skeleton is not linked across frames by ID.
- Keypoints are 2D only, so depth and rotation toward the camera are not captured.

## Future Work

- **Identity tracking** with ByteTrack or BoT-SORT to give each person a persistent ID.
- **ST-GCN modeling** to learn from the spatio-temporal graphs for movement and style recognition.
- **Synchronization analysis** comparing joint trajectories across people to measure group coordination.
- **Reference comparison** to score a performance against an expert sequence and flag errors.
- **3D lifting** to recover depth from the 2D keypoints.
- **Keypoint export** to JSON or NumPy for direct use in graph learning frameworks.

## References

- Jocher, G. et al. *Ultralytics YOLOv8*. https://github.com/ultralytics/ultralytics
- Yan, S., Xiong, Y., Lin, D. *Spatial Temporal Graph Convolutional Networks for Skeleton-Based Action Recognition*. AAAI 2018.
- Lin, T.-Y. et al. *Microsoft COCO: Common Objects in Context*. ECCV 2014.
