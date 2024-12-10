import os
from tqdm import tqdm
for person in tqdm(os.listdir('/xxxx/TETF/datasets/MEAD/agg_MEAD/')):
    os.makedirs('/xxxx/TETF/datasets/MEAD/agg_MEAD/' + person + '/audio_wav/', exist_ok=True)
    print(person)
    for m4a_file in os.listdir('/xxxx/TETF/datasets/MEAD/agg_MEAD/' + person + '/audio/'):
        os.system('ffmpeg -i ' + '/xxxx/TETF/datasets/MEAD/agg_MEAD/' + person + '/audio/' + m4a_file + ' -acodec pcm_s16le -ac 1 -ar 16000 ' + '/xxxx/TETF/datasets/MEAD/agg_MEAD/' + person + '/audio_wav/' + m4a_file[:-4] + '.wav')