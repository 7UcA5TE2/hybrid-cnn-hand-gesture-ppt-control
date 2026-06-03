import os
import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from tqdm import tqdm

#  PATHS 
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR = os.path.join(PROJECT_ROOT, "dataset_raw")
PROCESSED_DIR = os.path.join(PROJECT_ROOT, "dataset_processed")
MODEL_PATH = os.path.join(PROJECT_ROOT, "tools", "hand_landmarker.task")

IMG_SIZE = 224
MARGIN = 20  # pixels around hand

# MEDIAPIPE TASK 
base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=1
)
detector = vision.HandLandmarker.create_from_options(options)

os.makedirs(PROCESSED_DIR, exist_ok=True)

# PROCESS DATASET 
for gesture in os.listdir(RAW_DIR):
    gesture_raw_path = os.path.join(RAW_DIR, gesture)
    gesture_out_path = os.path.join(PROCESSED_DIR, gesture)

    if not os.path.isdir(gesture_raw_path):
        continue

    os.makedirs(gesture_out_path, exist_ok=True)

    print(f"\nProcessing gesture: {gesture}")

    for img_name in tqdm(os.listdir(gesture_raw_path)):
        img_path = os.path.join(gesture_raw_path, img_name)

        img = cv2.imread(img_path)
        if img is None:
            continue

        h, w, _ = img.shape
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=img_rgb
        )

        result = detector.detect(mp_image)

 # Skip if no hand detected
        if not result.hand_landmarks:
            continue

        landmarks = result.hand_landmarks[0]

        xs = [lm.x for lm in landmarks]
        ys = [lm.y for lm in landmarks]

        x_min = int(min(xs) * w) - MARGIN
        y_min = int(min(ys) * h) - MARGIN
        x_max = int(max(xs) * w) + MARGIN
        y_max = int(max(ys) * h) + MARGIN

        x_min = max(0, x_min)
        y_min = max(0, y_min)
        x_max = min(w, x_max)
        y_max = min(h, y_max)

        hand_crop = img[y_min:y_max, x_min:x_max]

        if hand_crop.size == 0:
            continue

        hand_crop = cv2.resize(hand_crop, (IMG_SIZE, IMG_SIZE))

        out_path = os.path.join(gesture_out_path, img_name)
        cv2.imwrite(out_path, hand_crop)

print("\nDataset preprocessing complete.")
