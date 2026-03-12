import os
import cv2
import pickle

# ------------------------
# Define all directories
# ------------------------
video_dir = r"C:\SCIoI\ExtractUVOAnnotation\dataset\uvoVideos"
anno_root = r"C:\SCIoI\ExtractUVOAnnotation\results\newAnnotationFiles"
output_dir = r"C:\SCIoI\ExtractUVOAnnotation\results\annotatedVideos"

os.makedirs(output_dir, exist_ok=True)

# ------------------------
# List all video files
# ------------------------
video_files = [
    f for f in os.listdir(video_dir)
    if f.lower().endswith((".mp4", ".avi", ".mov"))
]

print(f"Found {len(video_files)} videos.")

# ------------------------
# LOG LISTS
# ------------------------
not_annotation_found = []          # videos with NO pkl
videos_with_missing_bboxes = set() # videos where ann["box"] is None

mark_videos = ['--Kr1PZaPDI', '--dVV4_CSvw', '-081qTv9fXc',
               '-0CNBzthkZ4', '-0Li7rc78jQ', '-0TvmZ9L0ho', '-1QTRLQSzhQ',
               '-1RxG3SJZfY', '-2eq0citAJg', '-2kJpg1NIzM', '-3nc7ljnlFQ',
               '-5FU8vEKtyE', '-5yyKPocddE', '-6BGjxMc1BU', '-85kDMryJ9M',
               '-BGvHxisFII', '-CHrri2fxkk', '-GU9hPMO3mg', '-Gfncisg2GE',
               '-JHvT2pywbo', '-QRY9rDrJf0', '-UJgyiWe500', '-buq7C8Eqic',
               '-d2997jIb3c', '-dSUldceBAA', '-gJlTcI3jRI', '-gun8tHqLIw',
               '-h0RIu0ggZs', '-jLDOqR6XsI', '-ri90Xykrxg', '-sGcmYcU_QI',
               '0AZwSqSgCcc', '0AkA2Ru9qG0', '0CcZv2sSMfs', '0D2YY8fs0ok',
               '0L44b3HExVA', '0M1SkaJJcV0', '0RnlrsxG1hY', '0_VY2tpUC_E',
               '0azWFsmoz38', '0brGLuDMUmk', '0ctdGZygYeU', '0i1lP-WyXLU',
               '0jg1cEhzFL4', '0tHJCi6G4S4', '0u4c8Cel91U', '0zMVDwEJvBo',
               '1BkmCXHttEQ', '1OcA0gQFWjM', '1VfEeERUGKM', '1WEhv0Znmaw',
               '1eaYXyj0sTg', '1eitPYFWFIY', '1m1YYVYysA8', '2DNG46ZD9Ss',
               '2FuCzhX5hWg', '2POcyzPhDnY', '2Zm4Y65FpyM', '2dvIGGnC-LQ',
               '2dxs9ouQLec', '2fojVBo1tv0', '2mFclrl0wo8', '2qxlxhL8wIg',
               '3RdzszuykC4', '3WsJHm2P70M', '43HFIdLD6h8', '4VV5L7lhOjE',
               '5S0S8Tft-hI', '6Cwq13Oaays', 'fBcmTIo8dFE']

#videos = [f for f in video_files if os.path.splitext(f)[0] in mark_videos]
#print(f"Processing {len(videos)} marked videos.")
# ------------------------
# Process each video
# ------------------------
for video_file in video_files:

    video_path = os.path.join(video_dir, video_file)
    video_stem, ext = os.path.splitext(video_file)

    # Expected pkl path
    pkl_path = os.path.join(anno_root, f"{video_stem}.pkl")

    # If annotation does NOT exist, log and skip
    if not os.path.exists(pkl_path):
        print(f"❌ No annotation found for {video_file} → skipping.")
        not_annotation_found.append(video_file)
        continue

    print(f"\nProcessing → {video_file}")
    print(f"Loading annotations from → {pkl_path}")

    # Load annotation dictionary
    with open(pkl_path, "rb") as f:
        ann_dict = pickle.load(f)

    # Open video
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Output file path
    output_path = os.path.join(output_dir, f"{video_stem}_annotated{ext}")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    annotated_frames = sorted(int(idx) for idx in ann_dict.keys())

    missing_bbox_in_video = False  # flag to stop this video if needed

    # ------------ Annotate video frame by frame ------------
    for frame_idx in range(total_frames):

        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            break

        # If this frame has annotations
        if frame_idx in annotated_frames:

            for ann in ann_dict[str(frame_idx)]:

                # Handle missing bounding box → stop entire video
                if ann["box"] is None:
                    print(f"⚠️ Warning: No bounding box for frame {frame_idx} in {video_file} → stopping this video.")
                    videos_with_missing_bboxes.add(video_file)

                    continue

                # Draw bounding box
                x, y, w, h = map(int, ann["box"])
                cv2.rectangle(frame, (x, y), (x + w, y + h),
                              (0, 255, 0), 2)

                # Draw segmentation info text
                seg = ann["seg"]
                object_id = ann["obj_id"]
                #text = f"size:{seg.get('size')} count:{seg.get('counts')}"
                text = f"obj_id:{object_id}"
                cv2.putText(frame, text, (x, y - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (0, 255, 255), 1)



        # Write frame normally
        out.write(frame)

    # Cleanup
    cap.release()
    out.release()



    print(f"✅ Saved annotated video → {output_path}")


# ------------------------
# SAVE LOGS
# ------------------------

# No annotation found log
log_path = os.path.join(output_dir, "missing_annotations.txt")
with open(log_path, "w") as logf:
    for fname in not_annotation_found:
        logf.write(fname + "\n")

print(f"\n📄 Log saved → {log_path}")


print("\n🎉 Batch processing complete!")
