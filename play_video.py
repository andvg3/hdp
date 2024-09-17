import cv2
import os
import sys

# Define source and target directories
SOURCE_FOLDER = "/home/an.vuong/Desktop/ws/CoppeliaSim/"  # Folder containing the .avi files
TARGET_FOLDER = "/home/an.vuong/Desktop/ws/hdp/eval_video"  # Folder to save the frames

def extract_frames(avi_file_name, target_folder, frame_prefix="frame"):
    avi_file_path = os.path.join(SOURCE_FOLDER, avi_file_name)
    
    # Create a subfolder inside the target folder named after the video (without extension)
    subfolder_name = os.path.join(target_folder, avi_file_name.split('.')[0])
    if not os.path.exists(subfolder_name):
        os.makedirs(subfolder_name)

    cap = cv2.VideoCapture(avi_file_path)
    frame_number = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Save the frame as an image file in the subfolder
        frame_file_path = os.path.join(subfolder_name, f'{frame_prefix}_{frame_number}.jpg')
        cv2.imwrite(frame_file_path, frame)
        print(f'Saved {frame_file_path}')
        frame_number += 1

    cap.release()
    print(f"Extracted {frame_number} frames to {subfolder_name}.")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python play_video.py <video_file_name> [<frame_prefix>]")
        sys.exit(1)

    # Parse command-line arguments
    avi_file_name = sys.argv[1]
    frame_prefix = sys.argv[2] if len(sys.argv) > 2 else "frame"

    # Call the function to extract frames
    extract_frames(avi_file_name, TARGET_FOLDER, frame_prefix)
