import os
import argparse
import cv2
import numpy as np
from tqdm import tqdm
from headpose.detect import PoseEstimator


def get_headpose_by_headpose(img_path):
    try:
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        est = PoseEstimator()
        est.detect_landmarks(img)
        roll, pitch, yaw = est.pose_from_image(img)
    except:
        roll, pitch, yaw = 0.0, 0.0, 0.0
    return [roll, pitch, yaw]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--frames_dir', type=str, default='/home/aiscuser/HDTF/frames', help='to read frames')
    parser.add_argument('--target_dir', type=str, default='/mnt/shared_data/xxxx/projects/talkingface_LM/datasets/HDTF/clean_videos_split10s_ptcode/headpose_headpose', help='to write npy')
    parser.add_argument('--split', type=int, default=32, help='the number that split the dataset into')
    parser.add_argument('--num', type=int, default=0, help='the start number of the split')
    args = parser.parse_args()

    if not os.path.exists(args.target_dir):
        os.makedirs(args.target_dir)

    paths = sorted(os.listdir(args.frames_dir))[args.num::args.split]
    for sub_dir in tqdm(paths):
        if os.path.exists(f'{args.target_dir}/{sub_dir}.npy'):
            continue
        frames = sorted(os.listdir(f'{args.frames_dir}/{sub_dir}'))
        headpose = []
        for frame in frames:
            out_list = get_headpose_by_headpose(f'{args.frames_dir}/{sub_dir}/{frame}')
            headpose.append(out_list)
        headpose = np.array(headpose)
        np.save(f'{args.target_dir}/{sub_dir}.npy', headpose)