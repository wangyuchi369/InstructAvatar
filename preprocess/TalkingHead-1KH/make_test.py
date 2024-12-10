import os
all_mead_extend_list = list(os.listdir('/mnt/blob/xxxx/TETF/datasets/MEAD_extend/audio_wave2vec_all'))
all_mead_extend_list = [i[:-4] for i in all_mead_extend_list if i.split('_')[1] != 'neutral']
mapping_dict = {}
import random

all_tk1h = os.listdir('/xxxx/TETF/datasets/talkinghead-1kh/frames_256/')
all_tk1h_subset = random.sample(all_tk1h, 120)
for person in all_tk1h_subset:
    corresponding_video = random.choice(all_mead_extend_list)
    mapping_dict[person] = corresponding_video
    
import json
with open('/xxxx/TETF/datasets/talkinghead-1kh/mapping_dict.json', 'w') as f:
    json.dump(mapping_dict, f)
    