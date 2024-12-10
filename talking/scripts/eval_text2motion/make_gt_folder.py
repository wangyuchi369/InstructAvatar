import json
import os
gt_video = json.load(open('t2m_test_data.json', 'r'))
gt_video_list = list(gt_video.keys())
os.makedirs('/mnt/blob/xxxx/TETF/text2motion/gt', exist_ok=True)
from tqdm import tqdm
for t2m_video in tqdm(gt_video_list):
    person, video_name = '_'.join(t2m_video.split('_')[:4]), '_'.join(t2m_video.split('_')[4:])
    input_path = f'/mnt/blob/xxxx/TETF/datasets/text2motion/text2express/{person}/video_crop_resize_25fps_16k/{video_name}.MP4'
    output = f'/mnt/blob/xxxx/TETF/text2motion/gt/{t2m_video}.mp4'
    os.system(f'cp {input_path} {output}')
    

    
    