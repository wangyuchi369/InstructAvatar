import argparse
import os
import json
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
    default="/mnt/blob/xxxx/TETF/results/tmp/",
    help="the generated video to be evaluated",
)

sync_d_list = []
opt = parser.parse_args()
# opt.result_dir = os.path.join(opt.result_dir, 'single_video')
print(opt.result_dir)
for person in os.listdir(opt.result_dir):
    if person == 'generated_au' or person == 'lip_sync_saved' or person == 'final_results.txt' or person == 'full_video':
        continue
    print(person)
    print(os.path.join(opt.result_dir, person))
    for each_video in os.listdir(os.path.join(opt.result_dir, person)):
        print(each_video)
        video_path = os.path.join(opt.result_dir, person, each_video)
        saved_tmp_path = os.path.join(opt.result_dir, 'lip_sync_saved', person, each_video[:-4])
        os.system(f'bash evaluate_repos/syncnet_python/test.sh {video_path} {saved_tmp_path}')
        try:
            with open(os.path.join(saved_tmp_path, 'result.json')) as f:
                sync_d = json.load(f)['minval']
                print(sync_d)
            sync_d_list.append(sync_d)
        except:
            continue
       
# print(sync_d_list)

with open(os.path.join(opt.result_dir, 'final_results.txt'), 'a') as f:
    f.write(f'{opt.result_dir}\n Sync-D: {sum(sync_d_list)/len(sync_d_list):.3f}\n')