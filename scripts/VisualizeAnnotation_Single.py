import cv2
import pickle
import os

video_path = r"C:\SCIoI\Myannotation\sam3-main\myutils\uvoVideos\0-NiXohf8vM.mp4"
pkl_path   = r"C:\SCIoI\Myannotation\sam3-main\myutils\newAnnotationFiles\0-NiXohf8vM.pkl"
annotated_path = r"C:\SCIoI\Myannotation\sam3-main\myutils\annotatedVideos\\"

with open(pkl_path, "rb") as f:
    ann_dict = pickle.load(f)

cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS)
width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

fourcc = cv2.VideoWriter_fourcc(*"mp4v")

video_filename = os.path.basename(video_path)
video_stem, ext = os.path.splitext(video_filename)

output_path = os.path.join(
    os.path.dirname(annotated_path),
    f"{video_stem}_annotated{ext}"
)

out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# Convert keys to int for indexing
annotated_frames = sorted(int(idx) for idx in ann_dict.keys())

print("Annotating frames:", annotated_frames[:5], " ...")

# Loop over every frame of the video
for frame_idx in range(total_frames):

    # Seek and read exact frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    if not ret:
        break

    # If annotation exists for this frame
    if frame_idx in annotated_frames:
        for ann in ann_dict[str(frame_idx)]:

            x, y, w, h = map(int, ann["box"])
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            seg = ann["seg"]
            object_id = ann["obj_id"]
            #text = f"size:{seg.get('size')} count:{seg.get('counts')} obj_id:{object_id}"
            text = f"obj_id:{object_id}"
            cv2.putText(frame, text, (x, y - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

    out.write(frame)

cap.release()
out.release()
print(f"Annotated video saved → {output_path}")
