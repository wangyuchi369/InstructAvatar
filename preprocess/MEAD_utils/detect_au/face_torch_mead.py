from facetorch import FaceAnalyzer
from omegaconf import OmegaConf
from torch.nn.functional import cosine_similarity
from typing import Dict
import operator
import torchvision
import os
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

path_config="preprocess/MEAD_utils//detect_au/gpu.config.yml"
cfg = OmegaConf.load(path_config)
analyzer = FaceAnalyzer(cfg.analyzer)

import cv2
import argparse
import random
import os
parser = argparse.ArgumentParser()
parser.add_argument('--split', type=int, default=0, help='the number that split the dataset into')
args = parser.parse_args()
videos_dir = '/mnt/blob/xxxx/TETF/datasets/MEAD_all/MEAD_all_randcrop/'
tmp_frame_save_dir = '/mnt/blob/xxxx/TETF/datasets/MEAD_all/au_detect/'

person = sorted(os.listdir(videos_dir))[args.split]
if not person == 'paths.npy':
    subdir = os.path.join(videos_dir, person, 'video_crop_resize_25fps_16k')
    for video_path in tqdm(os.listdir(subdir)):
        assert video_path.endswith('.mp4')
        current_path = os.path.join(subdir, video_path)

        # 打开视频文件
        video = cv2.VideoCapture(current_path)

        # 获取视频的总帧数
        total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

        # 随机选择三个帧数
        selected_frames = random.sample(range(total_frames), 3)
        saved_dir = os.path.join(tmp_frame_save_dir, person, video_path.split('/')[-1].replace('.mp4', '/'))
        os.makedirs(saved_dir, exist_ok=True)
        each_video_au = []
        for i, frame_num in enumerate(selected_frames):
            # 将视频设置到选定的帧数
            video.set(cv2.CAP_PROP_POS_FRAMES, frame_num)

            # 读取该帧
            ret, frame = video.read()

            # 如果读取成功，保存该帧
            if ret:
                path_input = os.path.join(saved_dir, f'{i}.jpg')
                cv2.imwrite(path_input, frame)
                path_output = os.path.join(saved_dir, f'{i}_output.jpg')
                response = analyzer.run(
                        path_image=path_input,
                        batch_size=cfg.batch_size,
                        fix_img_size=cfg.fix_img_size,
                        return_img_data=False,
                        include_tensors=cfg.include_tensors,
                        # path_output=path_output,
                    )
                if len(response.faces) != 1:
                    print(f'error: {path_input} it has many faces or no face')
                    continue
                action_units = [response.faces[0].preds['au'].label] + response.faces[0].preds['au'].other['multi']
                each_video_au.append(action_units)
                with open(os.path.join(saved_dir, f'{i}_au.txt'), 'w') as f:
                    f.write(','.join(action_units))
                # print({face.indx: face.preds["au"].label + "," + ",".join(face.preds['au'].other['multi']) for face in response.faces})
        if len(each_video_au) == 0:
            print(f'this : {path_input} video has no face detected')
            continue
        intersection_au = set(each_video_au[0]).intersection(*each_video_au)
        import json
        with open(os.path.join(saved_dir, 'intersection.json'), 'w') as f:
            json.dump({"/".join(saved_dir.split('/')[-3:]) : list(intersection_au)}, f)
        # with open(os.path.join(saved_dir, 'intersection_au.txt'), 'w') as f:
        #     f.write(','.join(intersection_au))
        
        # 释放视频文件
        video.release()

