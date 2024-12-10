import os
import numpy as np
import cv2
from PIL import Image
from scipy.signal import convolve
from moviepy.editor import ImageSequenceClip, clips_array, AudioFileClip

def draw_kp(kp, size=(256,256), is_connect=False, color=(255,255,255)):
    frame = np.zeros((size[0], size[1], 3), dtype=np.uint8)
    for i in range(kp.shape[0]):
        x = int((kp[i][0]))
        y = int((kp[i][1]))
        thinkness = 1 if is_connect else 1
        frame = cv2.circle(frame, (x, y), thinkness, color, -1)
    return frame


def smooth_curve(array, weight=0.7):
    for i in range(1, len(array)):
        array[i] = array[i] * weight + array[i - 1] * (1 - weight)
    return array


def moving_average(x, w=5):
    assert w % 2 == 1, "w should be single!"
    kernel = np.ones(tuple([w] + [1] * (x.ndim - 1)))
    a = convolve(x, kernel, 'valid') / w
    return np.vstack((a[0:1, :].repeat(w//2, axis=0), a, a[-1:, :].repeat(w//2, axis=0)))


keypoint_npy = '/data/GAIA/amlt/hdtf_diff_wavn40l_keyla_ldmk_fromccd_keylaenc_default250len-lr-7/hdtf_diff_wavn40l_keyla_ldmk_fromccd_keylaenc_default250len-lr-7/2023-06-12T14-48-31_hdtf_diff_wavn40l_keyla_ldmk_fromccd_keylaenc/latents/val/WRA_VickyHartzler_000-00003_k-latent_rec_gs-110000_e-005499_b-000000.npy'
img_dir = '/data/GAIA/amlt/hdtf_diff_wavn40l_keyla_ldmk_fromccd_keylaenc_default250len-lr-7/hdtf_diff_wavn40l_keyla_ldmk_fromccd_keylaenc_default250len-lr-7/2023-06-12T14-48-31_hdtf_diff_wavn40l_keyla_ldmk_fromccd_keylaenc/images_ldmk/val/WRA_VickyHartzler_000-00003_k-latent_rec_gs-110000_e-005499_b-000000'
img_smoothed_dir = '/data/GAIA/amlt/hdtf_diff_wavn40l_keyla_ldmk_fromccd_keylaenc_default250len-lr-7/hdtf_diff_wavn40l_keyla_ldmk_fromccd_keylaenc_default250len-lr-7/2023-06-12T14-48-31_hdtf_diff_wavn40l_keyla_ldmk_fromccd_keylaenc/images_smoothed_ldmk/val/WRA_VickyHartzler_000-00003_k-latent_rec_gs-110000_e-005499_b-000000'
audio_path = '/data/datasets/HDTF/clean_videos_split10s_ptcode/audios_16k/WRA_VickyHartzler_000-00003.wav'

gen_video_path = '/data/GAIA/amlt/hdtf_diff_wavn40l_keyla_ldmk_fromccd_keylaenc_default250len-lr-7/hdtf_diff_wavn40l_keyla_ldmk_fromccd_keylaenc_default250len-lr-7/2023-06-12T14-48-31_hdtf_diff_wavn40l_keyla_ldmk_fromccd_keylaenc/videos_ldmk/val/WRA_VickyHartzler_000-00003_k-latent_rec_gs-110000_e-005499_b-000000.mp4'

# remove the index of mouth area
mouth_index = list(set([1, 122, 657, 202, 204, 43, 209, 218, 60, 61, 219, 220, 63, 221, 64, 
                        65, 636, 69, 637, 222, 66, 67, 223, 68, 224, 225, 70, 226, 227, 74, 
                        71, 75, 228, 72, 73, 660, 88, 316, 436, 516, 518, 358, 523, 532, 375, 
                        376, 533, 534, 378, 535, 379, 380, 384, 536, 381, 382, 537, 383, 538, 
                        539, 385, 540, 541, 386, 389, 542, 387, 388, 640]))
index = list(range(669))
for i in mouth_index:
    index.remove(i)

if not os.path.exists(img_dir):
    os.makedirs(img_dir)

if not os.path.exists(img_smoothed_dir):
    os.makedirs(img_smoothed_dir)

if not os.path.exists(os.path.dirname(gen_video_path)):
    os.makedirs(os.path.dirname(gen_video_path))

keypoints = np.load(keypoint_npy)
keypoints_smoothed = keypoints.copy().reshape((keypoints.shape[0], 669, 2))
keypoints_smoothed[:, index, :] = moving_average(keypoints_smoothed[:, index, :])
keypoints_smoothed = keypoints_smoothed.reshape((keypoints.shape[0], 669*2))

for i in range(keypoints.shape[0]):
    keypoint_img = Image.fromarray(draw_kp(keypoints[i].reshape((669, 2)), size=(256,256), is_connect=False, color=(255,255,255)))
    keypoint_img.save(os.path.join(img_dir, '%08d.png' % i))

for i in range(keypoints_smoothed.shape[0]):
    keypoint_smoothed_img = Image.fromarray(draw_kp(keypoints_smoothed[i].reshape((669, 2)), size=(256,256), is_connect=False, color=(255,255,255)))
    keypoint_smoothed_img.save(os.path.join(img_smoothed_dir, '%08d.png' % i))

gen_raw_video = ImageSequenceClip(img_dir, fps=25).set_duration(6)
gen_smoothed_video = ImageSequenceClip(img_smoothed_dir, fps=25).set_duration(6)

gen_video = clips_array([[gen_raw_video, gen_smoothed_video]])
gen_video.audio = AudioFileClip(audio_path, fps=16000).set_duration(6)

gen_video.write_videofile(gen_video_path)