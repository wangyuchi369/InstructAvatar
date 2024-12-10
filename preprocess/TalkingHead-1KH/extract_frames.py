import cv2
import os
from tqdm import tqdm

# Get the paths of all video files in the folder
folder_path = "/xxxx/li/dataset/TalkingHead-1KH/0613/video_crop_resize_25fps_16k/"
video_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith(".mp4")]

# Loop through all video files
for video_file in tqdm(video_files):
    # Open the video file
    cap = cv2.VideoCapture(video_file)

    # Get the frame rate of the video
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Set the interval for frame extraction
    interval = int(fps)

    # Set the folder to save the extracted frames
    video_name = os.path.split(video_file)[-1].split(".")[0]
    # save_folder = os.path.join('/ml-dl/xxxx/dataset/cctv', video_name)
    save_folder = os.path.join('./data/talkinghead_1kh', video_name)
    # Create the folder to save the extracted frames
    os.makedirs(save_folder, exist_ok=True)

    # Extract and save frames one by one
    count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        # if count % interval == 0:
        file_name = "%05d.jpg" % count
        save_path = os.path.join(save_folder, file_name)
        cv2.imwrite(save_path, frame)
        count += 1

    # Release the video file
    cap.release()