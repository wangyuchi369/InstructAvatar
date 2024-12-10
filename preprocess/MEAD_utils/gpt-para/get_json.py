import json
import os
from tqdm import tqdm

au_paraphrase = {}
for person in tqdm(os.listdir('/xxxx/TETF/datasets/MEAD_all/au_detect/')):
    if person == 'au_paraphrase.json':
        continue
    for video in tqdm(os.listdir('/xxxx/TETF/datasets/MEAD_all/au_detect/' + person + '/')):
        try:
            para = open('/xxxx/TETF/datasets/MEAD_all/au_detect/' + person + '/' + video + '/paraphrase.txt').read()
        except:
            continue
        
        au_paraphrase[f'{person}_{video}'] = para
        # print(person + video)
        # print(para)
        
with open('/xxxx/TETF/datasets/MEAD_all/au_detect/au_paraphrase.json', 'w') as f:
    json.dump(au_paraphrase, f)
        
        