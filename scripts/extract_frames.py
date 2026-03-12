import imageio
import os

video_path = r"C:\SCIoI\Myannotation\sam3-main\myutils\uvoVideos\0-b4M8jssX8.mp4"


video_name = os.path.splitext(os.path.basename(video_path))[0]

# Output folder
output_dir = rf"C:\SCIoI\Myannotation\frames\{video_name}"
os.makedirs(output_dir, exist_ok=True)

# Read and save frames
reader = imageio.get_reader(video_path)

for frame_id, img in enumerate(reader):
    frame_path = os.path.join(output_dir, f"{video_name}_{frame_id}.jpg")
    imageio.imwrite(frame_path, img)

print("Done!")