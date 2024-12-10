import json
import os
test_id = ['M003', "M019", 'W038', "W018", "M007"]
import random
output = {}

for person in os.listdir('/xxxx/TETF/datasets/MEAD_extend/MEAD_all_randcrop/'):
    if person not in test_id:
        continue
    video_list = os.listdir('/xxxx/TETF/datasets/MEAD_extend/MEAD_all_randcrop/' + person + '/video_crop_resize_25fps_16k/')
    neutral_video = [video for video in video_list if video.split('_')[0] == 'neutral']
    video_list = [video for video in video_list if video.split('_')[0] != 'neutral']
    
    subset_video_list = random.sample(video_list, 28)
    output[person] = subset_video_list
    
with open('./test_set.json', 'w') as f:
    json.dump(output, f)
    