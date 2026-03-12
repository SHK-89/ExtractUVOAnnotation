
import json
import os

train_path = r"C:\SCIoI\Myannotation\sam3-main\myutils\VideoDenseSet\Video_train_dense.json"
val_path = r"C:\SCIoI\Myannotation\sam3-main\myutils\VideoDenseSet\Video_val_dense.json"

with open(train_path, "r") as f:
    train_data = json.load(f)

train_videos = train_data["videos"]

with open(val_path, "r") as f:
    val_data = json.load(f)
val_videos = val_data["videos"]

train_video_names = set(video["ytid"] for video in train_videos)
val_video_names = set(video["ytid"] for video in val_videos)


def load_ytids(json_path):
    with open(json_path, "r") as f:
        data = json.load(f)
    return {video["ytid"] for video in data["videos"]}

train_ids = load_ytids(train_path)
val_ids   = load_ytids(val_path)

all_dataset_ids = train_ids.union(val_ids)
print(f"Train videos: {len(train_ids)}")
print(f"Val   videos: {len(val_ids)}")
print(f"Total dataset (train+val): {len(all_dataset_ids)}")


overlap_videos = train_ids.intersection(val_ids)
print(f"Number of overlapping videos: {len(overlap_videos)}")
if overlap_videos:
    print("Overlapping video names:")
    for name in overlap_videos:
        print(name)

# -----------------------------
# Get video files from directory
# -----------------------------
video_dir = r"C:\SCIoI\Myannotation\sam3-main\myutils\uvoVideos"
video_files = [
    f for f in os.listdir(video_dir)
    if f.lower().endswith((".mp4", ".avi", ".mov"))
]

print(f"Found {len(video_files)} videos in directory.")

# -----------------------------
# Get actual video files in directory
# -----------------------------
actual_files = {
    os.path.splitext(f)[0]   # remove .mp4 extension
    for f in os.listdir(video_dir)
    if f.lower().endswith((".mp4", ".avi", ".mov"))
}

print(f"Videos in directory: {len(actual_files)}")

# -----------------------------
# Find missing videos
# -----------------------------
missing_videos = sorted(all_dataset_ids - actual_files)

print("\nVideos that SHOULD exist (in JSON) but are NOT in directory:")
for vid in missing_videos:
    print(vid)

print(f"\nTotal missing videos: {len(missing_videos)}")