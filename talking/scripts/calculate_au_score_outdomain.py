import argparse
import os
from facetorch import FaceAnalyzer
from omegaconf import OmegaConf
import os
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")
import random
import os
import cv2
from sklearn.metrics import precision_score, recall_score, f1_score
import numpy as np
from talking.util import action_units_dict, typical_emotion_dict
import copy
import json

path_config="preprocess/MEAD_utils/detect_au/gpu.config.yml"
cfg = OmegaConf.load(path_config)
analyzer = FaceAnalyzer(cfg.analyzer)

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")

parser = argparse.ArgumentParser()

parser.add_argument(
    "--result_dir",
    type=str,
    # default="configs/vqvae_2enc-hdtf/test_autoencoder_seperate_kl_64x64x3.yaml",
    default="/mnt/blob/xxxx/TETF/eval_results_unify_outdomain/lr1e-5-dp0.08-unified_from_gaia_zero_conv_split_varlen_last/",
    help="the generated video to be evaluated",
)

opt = parser.parse_args()
y_true_list, y_pred_list, emotion_typical_list = [], [], []
for each_video in os.listdir(os.path.join(opt.result_dir,'single_video')):

    
    video_name = each_video[:-4]
    mead_video_name = "_".join(video_name.split('_')[-4:])
    mead_person, mead_video = mead_video_name.split('_')[0], '_'.join(mead_video_name.split('_')[1:])
        # assert video.endswith(".mp4")
    
    current_path = os.path.join(opt.result_dir, 'single_video', each_video)

    # 打开视频文件
    video = cv2.VideoCapture(current_path)
    
    if not os.path.exists(os.path.join(opt.result_dir, 'generated_au', each_video[:-4], 'intersection.json')):
        
        print(f'extracting au from {current_path}')
        # 获取视频的总帧数
        total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

        # 随机选择三个帧数
        selected_frames = random.sample(range(total_frames), 3)
        saved_dir = os.path.join(opt.result_dir, 'generated_au', current_path.split('/')[-1].replace('.mp4', '/'))
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
                # height, width, _ = frame.shape
                # right_half = frame[:, width//2:]
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
            json.dump({"/".join(saved_dir.split('/')[-2:]) : list(intersection_au)}, f)
            
        
    else:
        intersection_au = json.load(open(os.path.join(opt.result_dir, 'generated_au', each_video[:-4], 'intersection.json'), 'r'))['/'.join([each_video[:-4], ''])]
    # calculate score for each video
    gt_au_file = os.path.join('/mnt/blob/xxxx/TETF/datasets/MEAD_extend/au_detect',  mead_person, mead_video, 'intersection.json')
    try:
        gt_au = list(json.load(open(gt_au_file, 'r')).values())[0]
        if len(gt_au) == 0:
            continue
    except:
        continue
    # try:
    emotion = mead_video.split('_')[0]
    if emotion == 'neutral':
        continue
    
    
    y_gt = np.zeros(len(action_units_dict.keys()) + 1)
    y_gt[[action_units_dict[au.strip()] for au in gt_au]] = 1
    y_pred = np.zeros(len(action_units_dict.keys()) + 1)
    y_pred[[action_units_dict[au.strip()] for au in list(intersection_au)]] = 1
    y_true_list.append(y_gt)
    y_pred_list.append(y_pred)
    
    

    emotion_typical_vec = np.zeros(len(action_units_dict.keys()) + 1)
    for au in typical_emotion_dict[emotion]:
        emotion_typical_vec[action_units_dict[au]] = 1
    emotion_typical_list.append(emotion_typical_vec)
    if len(y_true_list) != len(y_pred_list) or len(y_true_list) != len(emotion_typical_list):
        print('error')
        
    # except:
    #     continue
    video.release()
            
        
        
y_true = np.array(y_true_list)
y_pred = np.array(y_pred_list)
emotion_typical = np.array(emotion_typical_list)
print('current eval path: ', opt.result_dir)
print('AU detection score:')
print('precision: {:.3f}'.format(precision_score(y_true, y_pred, average='samples')))
print('recall: {:.3f}'.format(recall_score(y_true, y_pred, average='samples')))
print('f1 score: {:.3f}'.format(f1_score(y_true, y_pred, average='samples')))
print('emotion typical score:')
# print('precision: {:.3f}'.format(precision_score(emotion_typical, y_pred, average='samples')))
print('recall: {:.3f}'.format(recall_score(emotion_typical, y_pred, average='samples')))
# print('f1 score: {:.3f}'.format(f1_score(emotion_typical, y_pred, average='samples')))

with open(os.path.join(opt.result_dir, 'final_results.txt'), 'a') as f:
    f.write(f'{opt.result_dir}\n AU detection score:\n precision: {precision_score(y_true, y_pred, average="samples"):.3f}\n recall: {recall_score(y_true, y_pred, average="samples"):.3f}\n f1 score: {f1_score(y_true, y_pred, average="samples"):.3f}\n emotion typical score:\n recall: {recall_score(emotion_typical, y_pred, average="samples"):.3f}\n')
