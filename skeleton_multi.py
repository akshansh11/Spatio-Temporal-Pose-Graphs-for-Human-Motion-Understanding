"""
Multi-person dancing skeleton on black background, same resolution/fps/length as input.
Usage: python skeleton_multi.py input.mp4 output.mp4
Needs: pip install ultralytics opencv-python
"""
import sys, cv2, numpy as np
from ultralytics import YOLO

src, dst = sys.argv[1], sys.argv[2]
model = YOLO("yolov8m-pose.pt")

cap = cv2.VideoCapture(src)
fps = cap.get(cv2.CAP_PROP_FPS) or 30
w, h = int(cap.get(3)), int(cap.get(4))
total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
out = cv2.VideoWriter(dst, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

# COCO 17 keypoints: 0 nose,5/6 shoulders,7/8 elbows,9/10 wrists,11/12 hips,13/14 knees,15/16 ankles
EDGES = [(5,7),(7,9),(6,8),(8,10),(5,6),(5,11),(6,12),(11,12),
         (11,13),(13,15),(12,14),(14,16)]
LEFT, RIGHT = {5,7,9,11,13,15}, {6,8,10,12,14,16}
def col(*ids):
    s = set(ids)
    return (255,160,0) if s <= LEFT else (0,140,255) if s <= RIGHT else (255,255,255)

n = 0
while True:
    ok, frame = cap.read()
    if not ok: break
    canvas = np.zeros((h, w, 3), np.uint8)
    res = model(frame, verbose=False, conf=0.35)[0]
    if res.keypoints is not None and res.keypoints.conf is not None:
        boxes = res.boxes.xyxy.cpu().numpy()
        for kp, cf, bx in zip(res.keypoints.xy.cpu().numpy(),
                              res.keypoints.conf.cpu().numpy(), boxes):
            # scale line width to person size so near/far dancers look right
            size = max(bx[3] - bx[1], 1)
            t, r = max(2, int(size / 90)), max(3, int(size / 70))
            pts = {i: (int(x), int(y)) for i, ((x, y), c) in enumerate(zip(kp, cf)) if c > 0.5}
            for a, b in EDGES:
                if a in pts and b in pts:
                    cv2.line(canvas, pts[a], pts[b], col(a, b), t, cv2.LINE_AA)
            if 0 in pts and 5 in pts and 6 in pts:            # neck
                neck = ((pts[5][0]+pts[6][0])//2, (pts[5][1]+pts[6][1])//2)
                cv2.line(canvas, pts[0], neck, (255,255,255), t, cv2.LINE_AA)
            for i, p in pts.items():
                if i in (1, 2, 3, 4):                           # skip eyes/ears
                    continue
                cv2.circle(canvas, p, r, col(i), -1, cv2.LINE_AA)
    out.write(canvas)                                          # 1 output frame per input frame
    n += 1
    if n % 50 == 0:
        print(f"{n}/{total}", flush=True)
cap.release(); out.release()
print(f"done: {n} frames @ {fps} fps")
