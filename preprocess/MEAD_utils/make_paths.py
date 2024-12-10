import numpy as np
import os
from tqdm import tqdm

# a = np.load('/xxxx/dataset/text2motion/MEAD_all_randcrop/paths.npy')
# print(len(a))
# print(a[0:10])
paths = []
for person in tqdm(os.listdir('/xxxx/TETF/datasets/MEAD_all/MEAD_all_randcrop/')):
    if person == 'paths.npy':
        continue
    subdir = os.path.join('/xxxx/TETF/datasets/MEAD_all/MEAD_all_randcrop/', person, 'video_crop_resize_25fps_16k')
    for video in os.listdir(subdir):
        assert video.endswith('.mp4')
        current_path = os.path.join(subdir, video)
        paths.append('/mnt/blob/' + "/".join(current_path.split('/')[2:]))
        print(paths[-1])
       
np.save('/xxxx/TETF/datasets/MEAD_all/MEAD_all_randcrop/paths.npy', paths)
       