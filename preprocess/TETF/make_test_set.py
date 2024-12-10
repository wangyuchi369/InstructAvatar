import os
from tqdm import tqdm
os.makedirs('/xxxx/TETF/datasets/test_set', exist_ok=True)
os.makedirs('/xxxx/TETF/datasets/test_set/frames', exist_ok=True)
os.makedirs('/xxxx/TETF/datasets/test_set/au_detect', exist_ok=True)
os.makedirs('/xxxx/TETF/datasets/test_set/MEAD_all_randcrop', exist_ok=True)

test_id = ['M003', "M019", 'W038', "W018", "M007"]

for person in tqdm(os.listdir('/xxxx/TETF/datasets/MEAD_extend/frames/')):
    if person not in test_id:
        continue
    os.system('cp -r /xxxx/TETF/datasets/MEAD_extend/frames/' + person + ' /xxxx/TETF/datasets/test_set/frames/')
    os.system('cp -r /xxxx/TETF/datasets/MEAD_extend/au_detect/' + person + ' /xxxx/TETF/datasets/test_set/au_detect/')
    os.system('cp -r /xxxx/TETF/datasets/MEAD_extend/MEAD_all_randcrop//' + person + ' /xxxx/TETF/datasets/test_set/MEAD_all_randcrop/')