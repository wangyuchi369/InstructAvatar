import sys
import os
sys.path.append('src/clip')
sys.path.append('src/taming-transformers')
sys.path.append('.')
# sys.path.append('diffae/src')
# from diffae.src.templates import *
import torch
from PIL import Image
# from imageio.v3 import imread
from glob import glob
from tqdm import tqdm
from pathlib import Path
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from omegaconf import OmegaConf
import numpy as np
import argparse
import cv2

from pytorch_lightning import seed_everything
from torch import autocast
from ldm.util import instantiate_from_config


from onnxruntime import InferenceSession
from typing import Optional, Dict, Any
from face_synthetics_training.runtime.face_detector import FaceDetector
from face_synthetics_training.runtime import landmarks
from face_synthetics_training.runtime.landmarks import get_landmarks
from face_synthetics_training.runtime.models import LDMKS_DENSE_MODEL
from face_synthetics_training.runtime.utils import get_roi_size_multiplier
from face_synthetics_training.runtime.utils import get_input_image_size

class RegressLandmarksFromImages:
    """This class populates probabilistic 2D Landmarks by computing them from images.

    The LKG dense landmark model from https://dev.azure.com/microsoft/Analog/_git/vision.hu.face.synthetics.training
    is used to regress landmarks in each view.
    """

    def __init__(
        self,
        sigma_threshold: Optional[float] = None,
        onnx_model: Optional[Path] = LDMKS_DENSE_MODEL,
        never_from_sequence: Optional[bool] = False,
        override_detector_ags: Optional[Dict[str, Any]] = None,
    ):
        """Construct the Landmarks Regressor.

        Views where the mean landmark sigma is above sigma_threshold will be rejected.
        """
        self._sigma_threshold = sigma_threshold
        self._never_from_sequence = never_from_sequence
        self._dense_landmark_session = InferenceSession(
            str(onnx_model), providers=['CUDAExecutionProvider'])
        # TensorrtExecutionProvider CUDAExecutionProvider

        self._roi_size_multiplier = get_roi_size_multiplier(onnx_model)
        if override_detector_ags is not None:
            # FIXME: this is a hacky way to do this...
            # force the face detector to get set up
            get_landmarks(np.zeros((16, 16, 3)), False,
                          self._dense_landmark_session, self._roi_size_multiplier)
            # replace it with one with modified settings
            landmarks.FACE_TRACKER.detector = FaceDetector(
                **override_detector_ags)

        self.onnx_model = onnx_model

    def regress_landmarks_demo(self, input_img):
        """ Regresses 2D landmarks and sigmas (landmark uncertainty) in an image """

        onnx_sess = self._dense_landmark_session

        input_img_size = get_input_image_size(onnx_sess)

        input_name = onnx_sess.get_inputs()[0].name
        input_shape = onnx_sess.get_inputs()[0].shape
        input_img = np.transpose(input_img, (2, 0, 1)).reshape(
            1, *input_shape[1:])  # HWC to BCHW

        pred = onnx_sess.run(
            None, {input_name: input_img.astype(np.float32)})[0][0]
        num_landmarks = pred.shape[0] // 3
        coords = pred[:num_landmarks * 2].reshape(num_landmarks, 2)
        sigmas = np.exp(pred[num_landmarks * 2:]) * \
            ((input_img_size[0] + input_img_size[1]) / 4.0)

        # map from [-1, 1] to [0, roi_size]
        ldmks_2d = (0.5 * coords + 0.5) * input_img_size

        return ldmks_2d, sigmas

regress_landmarks = RegressLandmarksFromImages(never_from_sequence=True, onnx_model=LDMKS_DENSE_MODEL)

def draw_kp(kp, size=(256,256), is_connect=False, color=(255,255,255)):
    frame = np.zeros((size[0], size[1], 3), dtype=np.uint8)
    for i in range(kp.shape[0]):
        x = int((kp[i][0]))
        y = int((kp[i][1]))
        thinkness = 1 if is_connect else 1
        frame = cv2.circle(frame, (x, y), thinkness, color, -1)
    return frame

if __name__ == '__main__':
    img_path = "data/talkinghead_1kh/frames"   # 30000
    img_output = "data/talkinghead_1kh/frames_output"
    ldmk_output = "data/talkinghead_1kh/ldmks/"
    ldmk_npy_output = "data/talkinghead_1kh/ldmks_npy/"
    #print(len(list(os.listdir(img_path))), len(list(os.listdir(img_output))), len(list(os.listdir(ldmk_output))))
    # img_path = "/ml-dl/xxxx/datasets/ffhq-dataset/images1024x1024/" # 69271
    # img_output = "/ml-dl/v-leyili/dataset/FFHQ/frames/"
    # ldmk_output = "/ml-dl/v-leyili/dataset/FFHQ/ldmks/"
    # print(len(list(os.listdir(img_path))), len(list(os.listdir(img_output))), len(list(os.listdir(ldmk_output))))
    # exit()
    if not os.path.exists(img_output):
        os.makedirs(img_output)
    if not os.path.exists(ldmk_output):
        os.makedirs(ldmk_output)
    for dir in tqdm(os.listdir(img_path)):
        curr_path = os.path.join(img_path, dir)
        img_list = [f for f in os.listdir(curr_path) if f.endswith((".jpg", ".png"))]
        img_out_path = os.path.join(img_output, dir)
        ldmk_output_path = os.path.join(ldmk_output, dir)
        ldmk_npy_output_path = os.path.join(ldmk_npy_output, dir)
        if not os.path.exists(img_out_path):
            os.makedirs(img_out_path)
        if not os.path.exists(ldmk_output_path):
            os.makedirs(ldmk_output_path)
        if not os.path.exists(ldmk_npy_output_path):
            os.makedirs(ldmk_npy_output_path)
        for img in img_list:
            try:
                img_256 = Image.open(os.path.join(curr_path, img)).convert('RGB').resize((256,256))
                img_256.save(os.path.join(img_out_path, img))
                dense_img_256, sigma_img = regress_landmarks.regress_landmarks_demo(np.asarray(img_256)/255.)  # get all the coordinates
                ldmk_npy_path = os.path.join(ldmk_npy_output_path, img.split('.')[0]+'.npy')
                np.save(ldmk_npy_path, dense_img_256)
                gray_img = Image.fromarray(draw_kp(dense_img_256, size=(256,256), is_connect=False, color=(255,255,255)))
                gray_img.save(os.path.join(ldmk_output_path, img))
            except Exception as e:
                print(e, img)

