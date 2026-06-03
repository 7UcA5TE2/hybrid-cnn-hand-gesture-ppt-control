import os
import random
import shutil

# PATH SETUP 
project_path = os.path.dirname(os.path.dirname(__file__))

input_dataset = os.path.join(project_path, "dataset_processed")
output_dataset = os.path.join(project_path, "dataset_split")

# Split ratios
train_ratio = 0.7
val_ratio = 0.15
test_ratio = 0.15

random.seed(1)

# CREATE OUTPUT FOLDERS
for folder in ["train", "val", "test"]:
    os.makedirs(os.path.join(output_dataset, folder), exist_ok=True)

# PROCESS EACH GESTURE 
for gesture_name in os.listdir(input_dataset):

    gesture_path = os.path.join(input_dataset, gesture_name)
    if not os.path.isdir(gesture_path):
        continue

    images = os.listdir(gesture_path)
    random.shuffle(images)

    total_images = len(images)

    train_count = int(total_images * train_ratio)
    val_count = int(total_images * val_ratio)

    train_images = images[:train_count]
    val_images = images[train_count:train_count + val_count]
    test_images = images[train_count + val_count:]

    for split_name, split_images in zip(
        ["train", "val", "test"],
        [train_images, val_images, test_images]
    ):
        split_gesture_path = os.path.join(output_dataset, split_name, gesture_name)
        os.makedirs(split_gesture_path, exist_ok=True)

        for img in split_images:
            src_path = os.path.join(gesture_path, img)
            dst_path = os.path.join(split_gesture_path, img)
            shutil.copy(src_path, dst_path)

    print(f"{gesture_name} split completed")

print("\nDataset split finished successfully.")
