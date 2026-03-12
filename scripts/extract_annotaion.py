import json
import os
import pickle


input_path = r"C:\SCIoI\ExtractUVOAnnotation\dataset\VideoDenseSet\Video_val_dense.json"

output_root = r"C:\SCIoI\ExtractUVOAnnotation\results\newAnnotationFiles"
os.makedirs(output_root, exist_ok=True)

with open(input_path, "r") as f:
    data = json.load(f)

videos = data["videos"]

print(f"Total videos found: {len(videos)}")

# Loop over videos and save individually

for video in videos:
    object_id = 0
    video_id = video["id"]
    video_name = video["ytid"]
    frames = video["file_names"] # 99 frames
    # Extract index part (without .png)
    indices = [os.path.splitext(f.split("/")[-1])[0] for f in frames]
    # for each frame, get annotations
    video_annotations = [ann for ann in data["annotations"] if ann["video_id"] == video_id]
    annotationDict = {idx: [] for idx in indices}
    for ann in video_annotations:

        bboxes = ann.get("bboxes", [])
        segmentations = ann.get("segmentations", [])
        object_id += 1
        # Iterate over frames by position
        for pos, frame_idx in enumerate(indices):
            if pos >= len(bboxes) or pos >= len(segmentations): #or pos >= len(areas):
                print(f"Warning: Missing data for video {video_id} at position {pos}. Skipping.")
                continue

            bbox = bboxes[pos]
            seg = segmentations[pos]

            #area = areas[pos]

            seg_info = {
                "size": seg.get("size") if isinstance(seg, dict) else None,
                "counts": seg.get("counts")  if isinstance(seg, dict) else None #check for counts
            }

            annotationDict[frame_idx].append({
                "box": bbox,
                "seg": seg_info,
                "obj_id": object_id
                #"category_id": cat
            })
    #print(annotationDict)
        # Save one .pkl per video
    #folder_path = os.path.join(output_root, video_name)
    #os.makedirs(folder_path, exist_ok=True)

    pkl_path = os.path.join(output_root, f"{video_name}.pkl")
    with open(pkl_path, "wb") as f:
        pickle.dump(annotationDict, f)

    print(f"Saved frame-indexed annotations for {video_id} ({video_name}) → {pkl_path}")

