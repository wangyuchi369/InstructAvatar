import os
import json
from tqdm import tqdm
from collections import defaultdict


statistic_dict = dict()

for person in tqdm(os.listdir('/xxxx/TETF/datasets/MEAD_all/au_detect/')):
    for video_ in tqdm(os.listdir('/xxxx/TETF/datasets/MEAD_all/au_detect/'+person+'/')):
        emotion_type = video_.split('_')[0]
        emotion_level = video_.split('_')[2]
        if f'{emotion_type}_{emotion_level}' not in statistic_dict.keys():
            statistic_dict[f'{emotion_type}_{emotion_level}'] = defaultdict(int)
        if os.path.exists(os.path.join('/xxxx/TETF/datasets/MEAD_all/au_detect/',person,video_,'intersection.json')):
            action_units = list(json.load(open(os.path.join('/xxxx/TETF/datasets/MEAD_all/au_detect/',person,video_,'intersection.json'))).values())[0]
            for each_au in action_units:
                # print(f'{emotion_type}_{emotion_level}',each_au.strip())
                statistic_dict[f'{emotion_type}_{emotion_level}'][each_au.strip()] += 1
                
                
print(statistic_dict)
with open('au_statistic_all.json','w') as f:
    json.dump(statistic_dict,f)
            