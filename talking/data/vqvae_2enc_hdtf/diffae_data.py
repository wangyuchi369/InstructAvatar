import os
import logging
import traceback
import numpy as np
from io import BytesIO
from pathlib import Path
import random
import cv2

import lmdb
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

import torchvision.transforms.functional as Ftrans

class BaseLMDB(Dataset):
    def __init__(self, path, original_resolution, zfill: int = 5):
        self.original_resolution = original_resolution
        self.zfill = zfill
        self.env = lmdb.open(
            path,
            max_readers=32,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
        )

        if not self.env:
            raise IOError('Cannot open lmdb dataset', path)

        with self.env.begin(write=False) as txn:
            self.length = int(
                txn.get('length'.encode('utf-8')).decode('utf-8'))

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        with self.env.begin(write=False) as txn:
            key = f'{self.original_resolution}-{str(index).zfill(self.zfill)}'.encode(
                'utf-8')
            img_bytes = txn.get(key)

        buffer = BytesIO(img_bytes)
        img = Image.open(buffer)
        return img

class FFHQlmdb(Dataset):
    def __init__(self,
                 path,
                 image_size=256,
                 original_resolution=256,
                 split=None,
                 as_tensor: bool = True,
                 do_augment: bool = True,
                 do_normalize: bool = True,
                 **kwargs):
        self.original_resolution = original_resolution
        self.data = BaseLMDB(path, original_resolution, zfill=5)
        self.length = len(self.data)

        if split is None:
            self.offset = 0
        elif split == 'train':
            # last 60k
            self.length = self.length - 10000
            self.offset = 10000
        elif split == 'val':
            # first 10k
            self.length = 10000
            self.offset = 0
        else:
            raise NotImplementedError()

        transform = [
            transforms.Resize(image_size),
        ]
        if do_augment:
            transform.append(transforms.RandomHorizontalFlip())
        if as_tensor:
            transform.append(transforms.ToTensor())
        if do_normalize:
            transform.append(
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))
        self.transform = transforms.Compose(transform)

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        assert index < self.length
        index = index + self.offset
        img = self.data[index]
        if self.transform is not None:
            img = self.transform(img)
        return {'image': img, 'index': index}

class FramePoseDataset(Dataset):
    def __init__(
        self,
        frame_folder,
        head_pose_folder,
        image_size=256,
        exts=['mp4'],
        do_augment: bool = False,
        do_transform: bool = True,
        do_normalize: bool = True,
        sort_names=True,
        split=None,
        has_subdir: bool = True,
        num_people: int = -1,
        frame_len: int = 1500,
        use_face_ldmk: bool = True,
    ):
        super().__init__()
        self.frame_folder = frame_folder
        self.head_pose_folder = head_pose_folder
        self.image_size = image_size
        self.use_face_ldmk = use_face_ldmk

        face_ldmk_ids = [0, 1, 2, 515, 516, 3, 518, 4, 5, 6, 8, 523, 9, 10, 524, 15, 16, 17, 18, 19, 532, 533, 534, 535, 536, 537, 538, 539, 540, 541, 542, 22, 23, 29, 30, 31, 32, 549, 550, 551, 552, 553, 554, 43, 35, 36, 556, 44, 48, 49, 517, 568, 569, 56, 57, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 528, 100, 20, 102, 101, 21, 530, 628, 629, 630, 631, 119, 24, 122, 633, 636, 637, 632, 639, 640, 26, 644, 27, 653, 654, 655, 656, 657, 28, 659, 660, 661, 157, 158, 159, 160, 161, 163, 165, 33, 167, 171, 34, 543, 174, 175, 176, 177, 178, 179, 180, 181, 182, 544, 184, 545, 185, 37, 546, 189, 190, 191, 192, 193, 547, 194, 548, 201, 202, 203, 204, 207, 209, 210, 214, 216, 218, 219, 220, 221, 222, 223, 224, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 242, 254, 255, 315, 316, 317, 318, 319, 320, 321, 323, 324, 325, 329, 330, 331, 332, 333, 334, 335, 336, 337, 338, 339, 341, 342, 343, 344, 345, 346, 347, 348, 349, 350, 351, 352, 172, 14, 358, 173, 359, 371, 372, 375, 376, 377, 378, 379, 380, 381, 382, 383, 384, 385, 386, 387, 388, 389, 390, 391, 392, 393, 394, 395, 396, 397, 398, 399, 400, 401, 402, 7, 414, 415, 416, 433, 436, 471, 472, 473, 474, 475, 477, 479, 481, 485, 486, 487, 488, 489, 490, 491, 492, 493, 494, 495, 496, 498, 499, 503, 504, 505, 506, 507, 508]

        # relative paths (make it shorter, saves memory and faster to sort)
        self.paths = []
        # self.pos_paths = []

        sub_dirs = sorted([name for name in os.listdir(frame_folder) if os.path.isdir(os.path.join(frame_folder, name))])[:num_people]
        for sub_dir in sub_dirs:
            if os.path.exists(os.path.join(head_pose_folder, sub_dir+".npy")):
                self.paths.append(sub_dir)

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

        def draw_kp(kp, size=(256,256), is_connect=False, color=(255,255,255)):
            frame = np.zeros((size[0], size[1], 3), dtype=np.uint8)
            for i in range(kp.shape[0]):
                x = int((kp[i][0]))
                y = int((kp[i][1]))
                thinkness = 1 if is_connect else 1
                frame = cv2.circle(frame, (x, y), thinkness, color, -1)
            return frame

        regress_landmarks = RegressLandmarksFromImages(never_from_sequence=True, onnx_model=LDMKS_DENSE_MODEL)
        
        def transfer_to_ldmk(img_file):
            img_256 = Image.open(img_file).convert('RGB').resize((256,256))
            dense_img_256, sigma_img = regress_landmarks.regress_landmarks_demo(np.asarray(img_256)/255.)
            if self.use_face_ldmk:
                new_dense_img_256 = []
                for i in face_ldmk_ids:
                    new_dense_img_256.append(dense_img_256[i])
                dense_img_256 = np.array(new_dense_img_256)
            gray_img = Image.fromarray(draw_kp(dense_img_256, size=(256,256), is_connect=False, color=(255,255,255)))
            return gray_img
        
        def random_hdtf_frame(hdtf_path, head_pose_path):
            files = os.listdir(hdtf_path)
            random_frame_number, random_pos_number = random.sample(range(0, len(files) - 1), 2)

            img = Image.open(os.path.join(hdtf_path, files[random_frame_number])).convert('RGB')
            img_pos = Image.open(os.path.join(hdtf_path, files[random_pos_number])).convert('RGB')

            ldmk_img = transfer_to_ldmk(os.path.join(hdtf_path, files[random_frame_number]))
            ldmk_img_pos = transfer_to_ldmk(os.path.join(hdtf_path, files[random_pos_number]))

            head_pose = np.load(head_pose_path)
            img_pose = head_pose[random_frame_number]
            img_pos_pose = head_pose[random_pos_number]
            return img, img_pos, ldmk_img, ldmk_img_pos, img_pose, img_pos_pose

        self.length = len(self.paths)
        if split is None:
            self.offset = 0
        elif split == 'train':
            # last 60k
            self.length = self.length - 10
            self.offset = 10
        elif split == 'val':
            # first 10k
            self.length = 10
            self.offset = 0
        else:
            raise NotImplementedError()

        transform = [
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
        ]
        if do_augment:
            transform.append(transforms.RandomHorizontalFlip())
        if do_transform:
            transform.append(transforms.ToTensor())
        if do_normalize:
            transform.append(
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))
        self.transform = transforms.Compose(transform)
        self.random_hdtf_frame = random_hdtf_frame

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        offset_index = index + self.offset

        frame_path = os.path.join(self.frame_folder, self.paths[offset_index])
        head_pose_path = os.path.join(self.head_pose_folder, self.paths[offset_index]+".npy")

        try:
            img, img_pos, ldmk_img, ldmk_img_pos, img_pose, img_pos_pose = self.random_hdtf_frame(frame_path, head_pose_path)
        except:
            print(f"Wrong in {frame_path}, retry")
            return self.__getitem__(index)

        img = self.transform(img)
        img_pos = self.transform(img_pos)
        ldmk_img = self.transform(ldmk_img)
        ldmk_img_pos = self.transform(ldmk_img_pos)

        img_pose = torch.from_numpy(img_pose).float()
        img_pos_pose = torch.from_numpy(img_pos_pose).float()

        return {'image': img, 'positive_img': img_pos, 
                'ldmk_img': ldmk_img, 'ldmk_pos_img': ldmk_img_pos,
                'img_pose': img_pose, 'img_pos_pose': img_pos_pose,
                'index': index}

class VideoDataset(Dataset):
    def __init__(
        self,
        folder,
        image_size=256,
        exts=['mp4'],
        do_augment: bool = True,
        do_transform: bool = True,
        do_normalize: bool = True,
        sort_names=True,
        split=None,
        has_subdir: bool = True,
        num_people: int = -1,
        frame_len: int = 1500,
        use_contrastive: bool = False,
        contrast_mode: str = "motion", # motion or appearance
    ):
        super().__init__()
        self.folder = folder
        self.image_size = image_size
        self.use_contrastive = use_contrastive
        self.contrast_mode = contrast_mode

        # relative paths (make it shorter, saves memory and faster to sort)
        self.paths = [
            p.relative_to(folder) for ext in exts
            for p in Path(f'{folder}').glob(f'*.{ext}')
        ]
        if sort_names:
            self.paths = sorted(self.paths)
        
        self.length = len(self.paths)
        if split is None:
            self.offset = 0
        elif split == 'train':
            # last 60k
            self.length = self.length - 1000
            self.offset = 1000
        elif split == 'val':
            # first 10k
            self.length = 1000
            self.offset = 0
        else:
            raise NotImplementedError()

        transform = [
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
        ]
        if do_augment:
            transform.append(transforms.RandomHorizontalFlip())
        if do_transform:
            transform.append(transforms.ToTensor())
        if do_normalize:
            transform.append(
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))
        self.transform = transforms.Compose(transform)

        def random_frame(video_path):
            video = cv2.VideoCapture(video_path)

            # Get the total number of frames
            total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

            # Generate a random frame number
            random_frame_number = random.randint(0, total_frames - 1)

            # Set the video to the random frame
            video.set(cv2.CAP_PROP_POS_FRAMES, random_frame_number)

            # Read the frame
            ret, frame = video.read()

            # Release the video
            video.release()

            # Convert the frame to PIL format
            pil_frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            return pil_frame
        
        def positive_frame(video_path, mode="motion"):
            # positive pair for motion or appearance
            video = cv2.VideoCapture(video_path)

            total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
            
            if mode == "motion":
                random_frame_number = random.randint(0, total_frames - 1)
                positive_frame_numer = random_frame_number + 1 if random_frame_number + 1 < total_frames else random_frame_number - 1
            elif mode == "appearance":
                random_frame_number, positive_frame_numer = random.sample(range(total_frames), 2)
            else:
                raise NotImplementedError()

            video.set(cv2.CAP_PROP_POS_FRAMES, random_frame_number)
            ret, frame = video.read()

            video.set(cv2.CAP_PROP_POS_FRAMES, positive_frame_numer)
            ret, positive_frame = video.read()

            # Release the video
            video.release()

            # Convert the frame to PIL format
            pil_frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            positive_frame = Image.fromarray(cv2.cvtColor(positive_frame, cv2.COLOR_BGR2RGB))

            return pil_frame, positive_frame
        
        self.random_frame = random_frame
        self.positive_frame = positive_frame

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        index = index + self.offset
        path = os.path.join(self.folder, self.paths[index])
        # random select a frame of video
        try:
            if not self.use_contrastive:
                img = self.random_frame(path)
                img = self.transform(img)
                positive_img = None
            else:
                if self.contrast_mode == "motion":
                    img, positive_img = self.positive_frame(path, mode="motion")
                elif self.contrast_mode == "appearance":
                    img, positive_img = self.positive_frame(path, mode="appearance")
                else:
                    raise NotImplementedError()
                img = self.transform(img)
                positive_img = self.transform(positive_img)
        except:
            # remove the corrupted video
            self.paths.pop(index)
            if index >= len(self.paths):
                index = len(self.paths) - 1
            return self.__getitem__(index)
            
        return {'image': img, 'positive_img': positive_img, 'index': index}

class FrameDataset(Dataset):
    def __init__(
        self,
        folder,
        image_size=256,
        exts=['mp4'],
        do_augment: bool = False,
        do_transform: bool = True,
        do_normalize: bool = True,
        sort_names=True,
        split=None,
        has_subdir: bool = True,
        num_people: int = -1,
        frame_len: int = 1500,
        use_contrastive: bool = False,
        use_ldmk: bool = False,
        contrast_mode: str = "appearance", # motion or appearance
    ):
        super().__init__()
        self.folder = folder
        self.image_size = image_size
        self.use_contrastive = use_contrastive
        self.contrast_mode = contrast_mode
        self.use_ldmk = use_ldmk

        # relative paths (make it shorter, saves memory and faster to sort)
        self.paths = []
        self.pos_paths = []
        errors = ['WDA_JohnLewis0_000/05988.jpg', 'WRA_CoryGardner1_000/00053.jpg']
        # random select {num_people} people

        if not os.path.exists(f'{self.folder}/paths_numpeople{num_people}_framelen{frame_len}.npy'):
            sub_dirs = sorted(os.listdir(folder))[:num_people]
            for sub_dir in sub_dirs:
                if not os.path.isdir(f'{folder}/{sub_dir}'):
                    continue
                # random select continuous frames with length {frame_len} from each person
                files = sorted(os.listdir(f'{folder}/{sub_dir}'))
                if len(files) > frame_len:
                    start = random.randint(0, len(files) - frame_len)
                    files = files[start:start + frame_len]

                if self.use_contrastive:
                    if self.contrast_mode == "motion":
                        for i in range(len(files) - 1):
                            self.paths.append(f'{sub_dir}/{files[i]}')
                            self.pos_paths.append(f'{sub_dir}/{files[i + 1]}')
                        self.paths.append(f'{sub_dir}/{files[-1]}')
                        self.pos_paths.append(f'{sub_dir}/{files[-2]}')
                    elif self.contrast_mode == "appearance":
                        for i in range(len(files)):
                            select_pool = list(range(0, i)) + list(range(i+1, len(files)))
                            pos_i = random.choice(select_pool)
                            if f'{sub_dir}/{files[i]}' in errors or f'{sub_dir}/{files[pos_i]}' in errors:
                                continue
                            self.paths.append(f'{sub_dir}/{files[i]}')
                            self.pos_paths.append(f'{sub_dir}/{files[pos_i]}')
                    else:
                        raise NotImplementedError()
                else:
                    for file in files:
                        if f'{sub_dir}/{file}' in errors:
                            continue
                        self.paths.append(f'{sub_dir}/{file}')
                        self.pos_paths.append(f'{sub_dir}/{file}')

            if sort_names:
                sorted_pairs = sorted(zip(self.paths, self.pos_paths), key=lambda pair: pair[0])
                self.paths, self.pos_paths = zip(*sorted_pairs)
            
            np.save(f'{self.folder}/paths_numpeople{num_people}_framelen{frame_len}.npy', np.array(self.paths))
            np.save(f'{self.folder}/pos_paths_numpeople{num_people}_framelen{frame_len}.npy', np.array(self.pos_paths))
        else:
            self.paths = list(np.load(f'{self.folder}/paths_numpeople{num_people}_framelen{frame_len}.npy', allow_pickle=True))
            self.pos_paths = list(np.load(f'{self.folder}/pos_paths_numpeople{num_people}_framelen{frame_len}.npy', allow_pickle=True))
        
        self.paths, self.pos_paths = list(self.paths), list(self.pos_paths)

        self.length = len(self.paths)
        if split is None:
            self.offset = 0
        elif split == 'train':
            # last 60k
            self.length = self.length - 1000
            self.offset = 1000
        elif split == 'val':
            # first 10k
            self.length = 1000
            self.offset = 0
        else:
            raise NotImplementedError()

        transform = [
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
        ]
        if do_augment:
            transform.append(transforms.RandomHorizontalFlip())
        if do_transform:
            transform.append(transforms.ToTensor())
        if do_normalize:
            transform.append(
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))
        self.transform = transforms.Compose(transform)

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        offset_index = index + self.offset
        # if offset_index >= len(self.paths):
        #     if offset_index % len(self.paths) + self.offset < len(self.paths):
        #         offset_index = offset_index % len(self.paths) + self.offset
        #     else:
        #         offset_index = offset_index % len(self.paths)

        path = os.path.join(self.folder, self.paths[offset_index])
        pos_path = os.path.join(self.folder, self.pos_paths[offset_index])
        img = Image.open(path)
        pos_img = Image.open(pos_path)
        # if the image is 'rgba'!
        img = img.convert('RGB')
        pos_img = pos_img.convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
            pos_img = self.transform(pos_img)
        
        if self.use_ldmk:
            # try:
            ldmk_img = Image.open(path.replace("frames", "ldmks")).convert('RGB')
            ldmk_pos_img = Image.open(pos_path.replace("frames", "ldmks")).convert('RGB')
            ldmk_img = self.transform(ldmk_img)
            ldmk_pos_img = self.transform(ldmk_pos_img)
            # except:
            #     print(f"Pop file {self.paths[offset_index]}")
            #     # remove the corrupted video
            #     self.paths.pop(offset_index)
            #     if index >= len(self.paths):
            #         index = len(self.paths) - 1
            #     return self.__getitem__(index)

            return {'image': img, 'positive_img': pos_img, 
                    'ldmk_img': ldmk_img, 'ldmk_pos_img': ldmk_pos_img,
                    'index': index}
        else:
            return {'image': img, 'positive_img': pos_img, 'index': index}

class FrameDatasetwithMouthMask(Dataset):
    def __init__(
        self,
        folder,
        image_size=256,
        exts=['mp4'],
        do_augment: bool = False,
        do_transform: bool = True,
        do_normalize: bool = True,
        sort_names=True,
        split=None,
        has_subdir: bool = True,
        num_people: int = -1,
        frame_len: int = 1500,
        use_contrastive: bool = False,
        use_ldmk: bool = False,
        contrast_mode: str = "appearance", # motion or appearance
        ldmk_mode: str = "ldmks_smooth_nomouth_moveavg",
    ):
        super().__init__()
        self.folder = folder
        self.image_size = image_size
        self.use_contrastive = use_contrastive
        self.contrast_mode = contrast_mode
        self.use_ldmk = use_ldmk
        self.ldmk_mode = ldmk_mode

        import json

        try:
            with open(os.path.join(os.path.dirname(folder), 'HDTF_mouth_rect.json'), 'r') as f:
                self.mouth_rect = json.load(f)

            with open(os.path.join(os.path.dirname(folder), 'HDTF_video_list.json'), 'r') as f:
                self.video_list = json.load(f)
        except:
            raise FileNotFoundError('''Cannot find 'HDTF_mouth_rect.json' or 'HDTF_video_list.json'. 
                                       You can find them on blob at "ml-dl/v-leyili/dataset/HDTF/"''')
        
        # relative paths (make it shorter, saves memory and faster to sort)
        self.paths = []
        self.pos_paths = []
        errors = ['WDA_JohnLewis0_000/05988.jpg', 'WRA_CoryGardner1_000/00053.jpg']
        # random select {num_people} people

        sub_dirs = sorted(self.video_list)[:num_people]
        for sub_dir in sub_dirs:
            if sub_dir.endswith('.npy'):
                continue
            # random select continuous frames with length {frame_len} from each person
            files = sorted(os.listdir(f'{folder}/{sub_dir}'))
            if len(files) > frame_len:
                start = random.randint(0, len(files) - frame_len)
                files = files[start:start + frame_len]

            if self.use_contrastive:
                if self.contrast_mode == "motion":
                    for i in range(len(files) - 1):
                        self.paths.append(f'{sub_dir}/{files[i]}')
                        self.pos_paths.append(f'{sub_dir}/{files[i + 1]}')
                    self.paths.append(f'{sub_dir}/{files[-1]}')
                    self.pos_paths.append(f'{sub_dir}/{files[-2]}')
                elif self.contrast_mode == "appearance":
                    for i in range(len(files)):
                        select_pool = list(range(0, i)) + list(range(i+1, len(files)))
                        pos_i = random.choice(select_pool)
                        if f'{sub_dir}/{files[i]}' in errors or f'{sub_dir}/{files[pos_i]}' in errors:
                            continue
                        self.paths.append(f'{sub_dir}/{files[i]}')
                        self.pos_paths.append(f'{sub_dir}/{files[pos_i]}')
                else:
                    raise NotImplementedError()
            else:
                for file in files:
                    if f'{sub_dir}/{file}' in errors:
                        continue
                    self.paths.append(f'{sub_dir}/{file}')
                    self.pos_paths.append(f'{sub_dir}/{file}')

        if sort_names:
            sorted_pairs = sorted(zip(self.paths, self.pos_paths), key=lambda pair: pair[0])
            self.paths, self.pos_paths = zip(*sorted_pairs)
        
        self.paths, self.pos_paths = list(self.paths), list(self.pos_paths)

        self.length = len(self.paths)
        if split is None:
            self.offset = 0
        elif split == 'train':
            # last 60k
            self.length = self.length - 1000
            self.offset = 1000
        elif split == 'val':
            # first 10k
            self.length = 1000
            self.offset = 0
        else:
            raise NotImplementedError()

        transform = [
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
        ]
        if do_augment:
            transform.append(transforms.RandomHorizontalFlip())
        if do_transform:
            transform.append(transforms.ToTensor())
        if do_normalize:
            transform.append(
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))
        self.transform = transforms.Compose(transform)

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        offset_index = index + self.offset
        # if offset_index >= len(self.paths):
        #     if offset_index % len(self.paths) + self.offset < len(self.paths):
        #         offset_index = offset_index % len(self.paths) + self.offset
        #     else:
        #         offset_index = offset_index % len(self.paths)

        path = os.path.join(self.folder, self.paths[offset_index])
        pos_path = os.path.join(self.folder, self.pos_paths[offset_index])
        mouth_rect = self.mouth_rect[self.paths[offset_index]]
        mouth_mask = torch.zeros((1, 256, 256))
        mouth_mask[0, mouth_rect[1]:mouth_rect[3], mouth_rect[0]:mouth_rect[2]] = 1
        img = Image.open(path)
        pos_img = Image.open(pos_path)
        # if the image is 'rgba'!
        img = img.convert('RGB')
        pos_img = pos_img.convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
            pos_img = self.transform(pos_img)
        
        if self.use_ldmk:
            # try:
            ldmk_img = Image.open(path.replace("frames", self.ldmk_mode)).convert('RGB')
            ldmk_pos_img = Image.open(pos_path.replace("frames", self.ldmk_mode)).convert('RGB')
            ldmk_img = self.transform(ldmk_img)
            ldmk_pos_img = self.transform(ldmk_pos_img)
            # except:
            #     print(f"Pop file {self.paths[offset_index]}")
            #     # remove the corrupted video
            #     self.paths.pop(offset_index)
            #     if index >= len(self.paths):
            #         index = len(self.paths) - 1
            #     return self.__getitem__(index)

            return {'image': img, 'positive_img': pos_img, 
                    'ldmk_img': ldmk_img, 'ldmk_pos_img': ldmk_pos_img,
                    'index': index, 'mouth_mask': mouth_mask, 'path': path, 'pos_path': pos_path}
        else:
            return {'image': img, 'positive_img': pos_img, 'index': index, 'mouth_mask': mouth_mask, 'path': path, 'pos_path': pos_path}

class FrameDatasetWithTemporal(FrameDataset):
    def __init__(self, folder, image_size=256, exts=['mp4'], do_augment: bool = False, 
                 do_transform: bool = True, do_normalize: bool = True, sort_names=True, 
                 split=None, has_subdir: bool = True, num_people: int = -1, frame_len: int = 1500, 
                 use_contrastive: bool = False, use_ldmk: bool = False, contrast_mode: str = "appearance", 
                 temporal_len: int = 5):
        super().__init__(folder, image_size, exts, do_augment, do_transform, do_normalize, sort_names, split, has_subdir, num_people, frame_len, use_contrastive, use_ldmk, contrast_mode)
        self.temporal_len = temporal_len
    def __len__(self):
        return super().__len__() // self.temporal_len
    def __getitem__(self, index):
        temporal_list = []
        for i in range(index * self.temporal_len, (index+1) * self.temporal_len):
            temporal_list.append(super().__getitem__(i))
        return {'image': torch.stack([temporal_list[i]['image'] for i in range(self.temporal_len)]),
                'positive_img': torch.stack([temporal_list[i]['positive_img'] for i in range(self.temporal_len)]), 
                'ldmk_img': torch.stack([temporal_list[i]['ldmk_img'] for i in range(self.temporal_len)]),
                'ldmk_pos_img': torch.stack([temporal_list[i]['ldmk_pos_img'] for i in range(self.temporal_len)]),
                'index': index}

class HDTFandFaceDataset(Dataset):
    def __init__(
        self,
        hdtf_folder = None,
        hhfq_folder = None,
        celeba_folder = None,
        image_size=256,
        exts=['jpg', 'png'],
        do_augment: bool = False,
        do_transform: bool = True,
        do_normalize: bool = True,
        sort_names=True,
        split=None,
        has_subdir: bool = True,
        num_people: int = -1,
        frame_len: int = 1500,      # 100   TODO: here can change small to accelerate
        use_contrastive: bool = False,
        use_ldmk: bool = False,
        contrast_mode: str = "appearance", # motion or appearance
    ):
        super().__init__()
        self.hdtf_folder = hdtf_folder

        # face dataset
        self.hhfq_folder = hhfq_folder
        self.celeba_folder = celeba_folder

        self.image_size = image_size
        self.use_contrastive = use_contrastive
        self.contrast_mode = contrast_mode
        self.use_ldmk = use_ldmk

        # relative paths (make it shorter, saves memory and faster to sort)
        self.paths = []
        self.pos_paths = []
        errors = ['WDA_JohnLewis0_000/05988.jpg', 'WRA_CoryGardner1_000/00053.jpg']
        # random select {num_people} people

        # hdtf
        if self.hdtf_folder is not None:
            if not os.path.exists(f'{self.hdtf_folder}/paths_numpeople{num_people}_framelen{frame_len}.npy'):
                sub_dirs = [dir for dir in os.listdir(hdtf_folder) if os.path.isdir(os.path.join(hdtf_folder, dir))]
                sub_dirs = sorted(sub_dirs)[:num_people]
                for sub_dir in sub_dirs:
                    # random select continuous frames with length {frame_len} from each person
                    files = sorted(os.listdir(f'{hdtf_folder}/{sub_dir}'))
                    if len(files) > frame_len:
                        start = random.randint(0, len(files) - frame_len)
                        files = files[start:start + frame_len]

                    if self.use_contrastive:
                        if self.contrast_mode == "motion":
                            for i in range(len(files) - 1):
                                self.paths.append(f'{sub_dir}/{files[i]}')
                                self.pos_paths.append(f'{sub_dir}/{files[i + 1]}')
                            self.paths.append(f'{sub_dir}/{files[-1]}')
                            self.pos_paths.append(f'{sub_dir}/{files[-2]}')
                        elif self.contrast_mode == "appearance":
                            for i in range(len(files)):
                                select_pool = list(range(0, i)) + list(range(i+1, len(files)))
                                pos_i = random.choice(select_pool)
                                if f'{sub_dir}/{files[i]}' in errors or f'{sub_dir}/{files[pos_i]}' in errors:
                                    continue
                                self.paths.append(f'{sub_dir}/{files[i]}')
                                self.pos_paths.append(f'{sub_dir}/{files[pos_i]}')
                        else:
                            raise NotImplementedError()
                    else:
                        for file in files:
                            if f'{sub_dir}/{file}' in errors:
                                continue
                            self.paths.append(f'{sub_dir}/{file}')
                            self.pos_paths.append(f'{sub_dir}/{file}')

                if sort_names:
                    sorted_pairs = sorted(zip(self.paths, self.pos_paths), key=lambda pair: pair[0])
                    self.paths, self.pos_paths = zip(*sorted_pairs)
                
                np.save(f'{self.hdtf_folder}/paths_numpeople{num_people}_framelen{frame_len}.npy', np.array(self.paths))
                np.save(f'{self.hdtf_folder}/pos_paths_numpeople{num_people}_framelen{frame_len}.npy', np.array(self.pos_paths))
            else:
                self.paths = list(np.load(f'{self.hdtf_folder}/paths_numpeople{num_people}_framelen{frame_len}.npy', allow_pickle=True))
                self.pos_paths = list(np.load(f'{self.hdtf_folder}/pos_paths_numpeople{num_people}_framelen{frame_len}.npy', allow_pickle=True))
            
        # face dataset
        face_paths = []
        if self.hhfq_folder is not None:
            if not os.path.exists(f'{self.hhfq_folder}/paths.npy'):
                paths = []
                # files = sorted([f for f in os.listdir(self.hhfq_folder) if os.path.isfile(os.path.join(self.hhfq_folder, f))])
                files = sorted([f.name for ext in exts for f in Path(self.hhfq_folder).glob(f'*.{ext}')])
                for file in files:
                    if f'{file}' in errors:
                        continue
                    paths.append(f'{file}')
                np.save(f'{self.hhfq_folder}/paths.npy', np.array(paths))
            else:
                paths = list(np.load(f'{self.hhfq_folder}/paths.npy', allow_pickle=True))
            paths = [os.path.join(self.hhfq_folder, f) for f in paths]
            face_paths = face_paths + paths

        if self.celeba_folder is not None:
            if not os.path.exists(f'{self.celeba_folder}/paths.npy'):
                paths = []
                # files = sorted([f for f in os.listdir(self.celeba_folder) if os.path.isfile(os.path.join(self.celeba_folder, f))])
                files = sorted([f.name for ext in exts for f in Path(self.celeba_folder).glob(f'*.{ext}')])
                for file in files:
                    if f'{file}' in errors:
                        continue
                    paths.append(f'{file}')
                if sort_names:
                    paths = sorted(paths)
                np.save(f'{self.celeba_folder}/paths.npy', np.array(paths))
            else:
                paths = list(np.load(f'{self.celeba_folder}/paths.npy', allow_pickle=True))
            paths = [os.path.join(self.celeba_folder, f) for f in paths]
            face_paths = face_paths + paths
        
        self.indicator = [0] * len(self.paths) + [1] * len(face_paths)
        print(len(face_paths), len(self.paths))     # 99271 frame_len * 411
        self.paths = self.paths + face_paths
        self.pos_paths = self.pos_paths + face_paths
        self.paths, self.pos_paths = list(self.paths), list(self.pos_paths)
        
        self.length = len(self.paths)   # all frame
        
        if split is None:
            self.offset = 0
        elif split == 'train':
            # last 60k
            self.length = self.length - 1000
            self.offset = 1000
        elif split == 'val':
            # first 10k
            self.length = 1000
            self.offset = 0
        else:
            raise NotImplementedError()

        transform = [
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
        ]
        if do_augment:
            transform.append(transforms.RandomHorizontalFlip())
        if do_transform:
            transform.append(transforms.ToTensor())
        if do_normalize:
            transform.append(
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))
        self.transform = transforms.Compose(transform)

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        offset_index = index + self.offset
        # if offset_index >= len(self.paths):
        #     if offset_index % len(self.paths) + self.offset < len(self.paths):
        #         offset_index = offset_index % len(self.paths) + self.offset
        #     else:
        #         offset_index = offset_index % len(self.paths)
        try:
            if self.indicator[offset_index] == 0:   # hdtf
                path = os.path.join(self.hdtf_folder, self.paths[offset_index])
                pos_path = os.path.join(self.hdtf_folder, self.pos_paths[offset_index])
                img = Image.open(path)
                pos_img = Image.open(pos_path)
                img = img.convert('RGB')
                pos_img = pos_img.convert('RGB')
            else:   # face only one
                path = self.paths[offset_index]
                pos_path = path
                img = Image.open(path)
                img = img.convert('RGB')
                pos_img = img
        except Exception as error:
            print(error, f"Pop file {self.paths[offset_index]}")
            # remove the corrupted video
            self.paths.pop(offset_index)
            self.indicator.pop(offset_index)
            if index >= len(self.paths):
                index = len(self.paths) - 1
            return self.__getitem__(index)
        
        if self.transform is not None:
            img = self.transform(img)
            pos_img = self.transform(pos_img)
        
        if self.use_ldmk:
            ldmk_img = Image.open(path.replace("frames", "ldmks")).convert('RGB')
            ldmk_pos_img = Image.open(pos_path.replace("frames", "ldmks")).convert('RGB')
            ldmk_img = self.transform(ldmk_img)
            ldmk_pos_img = self.transform(ldmk_pos_img)

            return {'image': img, 'positive_img': pos_img, 
                    'ldmk_img': ldmk_img, 'ldmk_pos_img': ldmk_pos_img,
                    'index': index}
        else:
            return {'image': img, 'positive_img': pos_img, 'index': index}

class ExtendedAvenaDataset(Dataset):
    def __init__(
        self,
        hdtf_folder,
        tk1k_folder,
        avena_folder,
        image_size=256,
        do_augment: bool = False,
        do_transform: bool = True,
        do_normalize: bool = True,
        split=None,
        exts=['mp4', 'MP4'],
        has_subdir: bool = True,
        num_people: int = -1,
        frame_len: int = 1500,
        use_contrastive: bool = False,
        only_mouth: bool = False,
        dense_ldmk: bool = False,
        contrast_mode: str = "appearance", # motion or appearance
        sparse_outline: bool = False,
    ):
        super().__init__()
        self.image_size = image_size
        self.hdtf_folder = hdtf_folder

        # CCD dataset
        self.v2_folder = tk1k_folder
        self.v1_folder = tk1k_folder.replace("CCDv2_processed", "CCDv1_processed")

        self.image_size = image_size
        self.contrast_mode = contrast_mode
        self.only_mouth = only_mouth
        self.dense_ldmk = dense_ldmk
        self.sparse_outline = sparse_outline
        
        v1_sub_dirs = sorted(os.listdir(self.v1_folder))
        v2_sub_dirs = sorted(os.listdir(self.v2_folder))

        path_file = f'{self.v2_folder}/paths_nump{num_people}.npy'
        if not os.path.exists(path_file):
            paths = []
            for sub_dir in v2_sub_dirs:
                video_path = os.path.join(self.v2_folder, sub_dir, "video_crop_resize_25fps_16k")
                now_paths = [
                    p.relative_to(self.v2_folder) for ext in exts
                    for p in Path(f'{video_path}').glob(f'*.{ext}')
                ]
                for path in now_paths:
                    try:
                        # check if the video is corrupted
                        video = cv2.VideoCapture(os.path.join(self.v2_folder, path))
                        total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
                        ret, frame = video.read()
                        video.release()
                        if total_frames < 5 or ret is False:
                            continue
                    except Exception as e: 
                        print(f"error: {e} in loading {os.path.join(self.v2_folder, path)}")
                        continue
                    paths.append(os.path.join(self.v2_folder, path))

            for sub_dir in v1_sub_dirs:
                video_path = os.path.join(self.v1_folder, sub_dir, "video_crop_resize_25fps_16k")
                # ldmk_path = os.path.join(folder, sub_dir, "video_ldmk")
                now_paths = [
                    p.relative_to(self.v1_folder) for ext in exts
                    for p in Path(f'{video_path}').glob(f'*.{ext}')
                ]
                for path in now_paths:
                    try:
                        video = cv2.VideoCapture(os.path.join(self.v1_folder, path))
                        total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
                        ret, frame = video.read()
                        video.release()
                        if total_frames < 25 or ret is False:
                            continue
                    except Exception as e: 
                        print(f"error: {e} in loading {os.path.join(self.v1_folder, path)}")
                        continue
                    paths.append(os.path.join(self.v1_folder, path))
            np.save(path_file, np.array(paths))
        else:
            paths = list(np.load(path_file, allow_pickle=True))

        self.tk1k_paths = sorted(paths)[:num_people]
        hdtf_paths = sorted([name for name in os.listdir(hdtf_folder) if os.path.isdir(os.path.join(hdtf_folder, name))]) #[:num_people]
        val_names = ["WRA_LamarAlexander0_000", "WRA_MarcoRubio_000", "WRA_ReincePriebus_000", "WRA_ReneeEllmers_000", "WRA_TimScott_000", "WRA_VickyHartzler_000"]
        self.hdtf_paths = []
        self.hdtf_paths_val = []
        for path in hdtf_paths:
            if path in val_names:
                self.hdtf_paths_val.append(path)
            else:
                self.hdtf_paths.append(path)
        
        avena_paths = list(np.load(os.path.join(avena_folder, "paths_nump-1.npy"), allow_pickle=True))[:-100]
        self.paths = avena_paths + self.tk1k_paths + self.hdtf_paths
        self.indicator = [0] * len(avena_paths) + [0] * len(self.tk1k_paths) + [1] * len(self.hdtf_paths)
        
        self.length = len(self.paths)
        if split is None:
            self.offset = 0
        elif split == 'train':
            # last 60k
            self.length = self.length - 10
            self.offset = 10
        elif split == 'val':
            # first 10k
            self.length = 10
            self.offset = 0
        else:
            raise NotImplementedError()

        transform = [
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
        ]
        if do_augment:
            transform.append(transforms.RandomHorizontalFlip())
        if do_transform:
            transform.append(transforms.ToTensor())
        if do_normalize:
            transform.append(
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))
        self.transform = transforms.Compose(transform)

        import dlib
        from onnxruntime import InferenceSession
        from typing import Optional, Dict, Any
        from face_synthetics_training.runtime.face_detector import FaceDetector
        from face_synthetics_training.runtime import landmarks
        from face_synthetics_training.runtime.landmarks import get_landmarks
        from face_synthetics_training.runtime.models import LDMKS_DENSE_MODEL
        from face_synthetics_training.runtime.utils import get_roi_size_multiplier
        from face_synthetics_training.runtime.utils import get_input_image_size

        mouth_idx = list(set([
            1, 4, 37, 56, 60, 63, 64, 65, 66, 67, 68, 69, 122, 157, 159, 165, 204, 209, 210, 214, 218, 221, 222, 223, 316, 319, 352, 371, 375, 378, 379, 380, 381, 382, 383, 384, 436, 471, 473, 479, 518, 523, 524, 528, 532, 535, 536, 537, 629, 633, 636, 637, 640, 657, # upper_lip_and_philtrum
            1, 43, 60, 61, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 88, 122, 202, 204, 209, 218, 219, 220, 221, 222, 223, 224, 225, 226, 227, 228, 316, 358, 375, 376, 378, 379, 380, 381, 382, 383, 384, 385, 386, 387, 388, 389, 436, 516, 518, 523, 532, 533, 534, 535, 536, 537, 538, 539, 540, 541, 542, 636, 637, 640, 657, 660,
            6, 7, 17, 48, 49, 177, 178, 207,      # mouth
            14, 43, 44, 57, 70, 71, 72, 73, 74, 88, 171, 172, 173, 202, 203, 216, 219, 220, 224, 225, 226, 228, 329, 358, 359, 372, 385, 386, 387, 388, 485, 486, 487, 516, 517, 530, 533, 534, 538, 539, 540, 542, 639, 660, 661 # chin_and_lower_lip
            ]))
        face_idx = [0, 1, 2, 515, 516, 3, 518, 4, 5, 6, 8, 523, 9, 10, 524, 15, 16, 17, 18, 19, 532, 533, 534, 535, 536, 537, 538, 539, 540, 541, 542, 22, 23, 29, 30, 31, 32, 549, 550, 551, 552, 553, 554, 43, 35, 36, 556, 44, 48, 49, 517, 568, 569, 
                    56, 57, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 528, 100, 20, 102, 101, 21, 530, 628, 629, 630, 631, 119, 24, 122, 633, 636, 637, 632, 639, 640, 26, 
                    644, 27, 653, 654, 655, 656, 657, 28, 659, 660, 661, 157, 158, 159, 160, 161, 163, 165, 33, 167, 171, 34, 543, 174, 175, 176, 177, 178, 179, 180, 181, 182, 544, 184, 545, 185, 37, 546, 189, 190, 191, 192, 193, 547, 194, 548, 
                    201, 202, 203, 204, 207, 209, 210, 214, 216, 218, 219, 220, 221, 222, 223, 224, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 242, 254, 255, 315, 316, 317, 318, 319, 320, 321, 323, 324, 325, 
                    329, 330, 331, 332, 333, 334, 335, 336, 337, 338, 339, 341, 342, 343, 344, 345, 346, 347, 348, 349, 350, 351, 352, 172, 14, 358, 173, 359, 371, 372, 375, 376, 377, 378, 379, 380, 381, 382, 383, 384, 385, 386, 387, 388, 389, 390, 391, 392, 393, 394, 395, 396, 397, 398, 399, 400, 401, 402, 
                    7, 414, 415, 416, 433, 436, 471, 472, 473, 474, 475, 477, 479, 481, 485, 486, 487, 488, 489, 490, 491, 492, 493, 494, 495, 496, 498, 499, 503, 504, 505, 506, 507, 508]
        sparse_outline_idx = [0, 1, 2, 3, 4, 5, 516, 518, 8, 513, 10, 523, 525, 532, 533, 534, 535, 536, 537, 26, 27, 28, 538, 539, 540, 541, 542, 544, 547, 36, 549, 550, 551, 552, 553, 554, 43, 556, 548, 41, 47, 39, 42, 52, 53, 568, 569, 60, 61, 62, 
                             63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 100, 101, 102, 104, 105, 21, 22, 628, 629, 630, 119, 117, 633, 634, 636, 637, 643, 644, 654, 655, 656, 657, 660, 157, 158, 159, 160, 161, 165, 543, 545, 185, 546, 189, 199, 202, 
                             204, 209, 211, 218, 219, 220, 221, 222, 223, 224, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 242, 254, 255, 259, 267, 271, 272, 273, 276, 277, 315, 316, 317, 318, 319, 320, 323, 573, 325, 336, 337, 341, 342, 343, 351, 354, 579, 356, 357, 358, 362, 
                             581, 367, 368, 375, 376, 377, 378, 379, 380, 381, 382, 383, 384, 385, 386, 387, 388, 389, 390, 391, 392, 393, 394, 395, 396, 397, 398, 399, 400, 401, 402, 587, 586, 414, 415, 416, 417, 418, 429, 431, 433, 434, 471, 472, 473, 474, 475, 479, 499, 503, 508]
        detector = dlib.get_frontal_face_detector()
        predictor = dlib.shape_predictor("/mnt/blob/xxxx/diffusion/shape_predictor_68_face_landmarks.dat")

        class RegressLandmarksFromImages:
            def __init__(
                self,
                sigma_threshold: Optional[float] = None,
                onnx_model: Optional[Path] = LDMKS_DENSE_MODEL,
                never_from_sequence: Optional[bool] = False,
                override_detector_ags: Optional[Dict[str, Any]] = None,
            ):
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

        def process_one(img_256):
            # img_256 = Image.open(img_file).convert('RGB').resize((256,256))
            # img_256 = Image.fromarray(array).convert('RGB').resize((256,256))
            # sparse ldmks
            img_array = np.asarray(img_256)
            faces = detector(img_array)
            if self.dense_ldmk or self.only_mouth or len(faces) != 1 or self.sparse_outline:
                sparse_coords = []
            else:
                sparse_ldmks = predictor(img_array, faces[0])
                sparse_coords = np.zeros((sparse_ldmks.num_parts, 2), dtype=np.uint8)
                for i in range(sparse_ldmks.num_parts):
                    sparse_coords[i] = (sparse_ldmks.part(i).x, sparse_ldmks.part(i).y) # 68 sparse ldmks

            dense_img_256, sigma_img = regress_landmarks.regress_landmarks_demo(np.asarray(img_256)/255.)
            if not self.dense_ldmk:
                new_dense_img_256 = []
                if self.only_mouth:
                    preserve_id = face_idx
                elif self.sparse_outline:
                    preserve_id = sparse_outline_idx
                else:
                    preserve_id = mouth_idx
                for i in preserve_id:
                    new_dense_img_256.append(dense_img_256[i])
                new_dense_img_256.extend(sparse_coords)
                dense_img_256 = np.array(new_dense_img_256)
            
            ldmks = draw_kp(dense_img_256, size=(256,256), is_connect=False, color=(255,255,255))
            return ldmks

        def random_hdtf_frame(hdtf_path):
            files = os.listdir(hdtf_path)
            file_num = len(files)
            random_frame_number, random_pos_number = random.sample(range(0, file_num - 1), 2)

            img = Image.open(os.path.join(hdtf_path, files[random_frame_number])).convert('RGB')
            img_pos = Image.open(os.path.join(hdtf_path, files[random_pos_number])).convert('RGB')
            ldmk_img = Image.fromarray(process_one(img.resize((256,256))))
            ldmk_img_pos = Image.fromarray(process_one(img_pos.resize((256,256))))
            return img, img_pos, ldmk_img, ldmk_img_pos

        def random_video_frame(video_path):
            # ldmk_video_path = video_path.replace("video_crop_resize_25fps_16k", "video_ldmk")
            video = cv2.VideoCapture(video_path)
            # ldmk_video = cv2.VideoCapture(ldmk_video_path)
            # total_frames_ldmk = int(ldmk_video.get(cv2.CAP_PROP_FRAME_COUNT))
            total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
            # total_frames = min(total_frames, total_frames_ldmk)
            random_frame_number, random_pos_number = random.sample(range(0, total_frames - 1), 2)
            video.set(cv2.CAP_PROP_POS_FRAMES, random_frame_number)
            ret, frame = video.read()
            video.set(cv2.CAP_PROP_POS_FRAMES, random_pos_number)
            ret, frame_pos = video.read()
            video.release()

            ldmk_frame = process_one(Image.fromarray(frame).convert('RGB').resize((256,256)))
            ldmk_frame_pos = process_one(Image.fromarray(frame_pos).convert('RGB').resize((256,256)))

            pil_frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            pil_frame_pos = Image.fromarray(cv2.cvtColor(frame_pos, cv2.COLOR_BGR2RGB))
            ldmk_pil_frame = Image.fromarray(cv2.cvtColor(ldmk_frame, cv2.COLOR_BGR2RGB))
            ldmk_pil_frame_pos = Image.fromarray(cv2.cvtColor(ldmk_frame_pos, cv2.COLOR_BGR2RGB))
            return pil_frame, pil_frame_pos, ldmk_pil_frame, ldmk_pil_frame_pos
        
        self.random_hdtf_frame = random_hdtf_frame
        self.random_video_frame = random_video_frame
    
    def __len__(self):
        return self.length

    def __getitem__(self, index):
        offset_index = index + self.offset
        # if offset_index >= len(self.paths):
        #     if offset_index % len(self.paths) + self.offset < len(self.paths):
        #         offset_index = offset_index % len(self.paths) + self.offset
        #     else:
        #         offset_index = offset_index % len(self.paths)

        
        try:
            if self.indicator[offset_index] == 0:
                img, img_pos, ldmk_img, ldmk_img_pos = self.random_video_frame(self.paths[offset_index])
            else:
                path = os.path.join(self.hdtf_folder, self.paths[offset_index])
                img, img_pos, ldmk_img, ldmk_img_pos = self.random_hdtf_frame(path)
        except:
            print(f"Retry file {self.paths[offset_index]}")
            # remove the corrupted video
            # self.paths.pop(offset_index)
            # self.indicator.pop(offset_index)
            # if index >= len(self.paths):
            #     index = len(self.paths) - 1
            return self.__getitem__(index)
        

        img = self.transform(img)
        img_pos = self.transform(img_pos)
        ldmk_img = self.transform(ldmk_img)
        ldmk_img_pos = self.transform(ldmk_img_pos)
        return {'image': img, 'positive_img': img_pos, 
                    'ldmk_img': ldmk_img, 'ldmk_pos_img': ldmk_img_pos,
                    'index': index}

class ExtendedHDTFandCCDDataset(Dataset):
    def __init__(
        self,
        hdtf_folder,
        tk1k_folder,
        image_size=256,
        do_augment: bool = False,
        do_transform: bool = True,
        do_normalize: bool = True,
        split=None,
        exts=['mp4', 'MP4'],
        has_subdir: bool = True,
        num_people: int = -1,
        frame_len: int = 1500,
        use_contrastive: bool = False,
        only_mouth: bool = False,
        dense_ldmk: bool = False,
        contrast_mode: str = "appearance", # motion or appearance
        sparse_outline: bool = False,
    ):
        super().__init__()
        self.image_size = image_size
        self.hdtf_folder = hdtf_folder

        # CCD dataset
        self.v2_folder = tk1k_folder
        self.v1_folder = tk1k_folder.replace("CCDv2_processed", "CCDv1_processed")

        self.image_size = image_size
        self.contrast_mode = contrast_mode
        self.only_mouth = only_mouth
        self.dense_ldmk = dense_ldmk
        self.sparse_outline = sparse_outline
        
        v1_sub_dirs = sorted(os.listdir(self.v1_folder))
        v2_sub_dirs = sorted(os.listdir(self.v2_folder))

        path_file = f'{self.v2_folder}/paths_nump{num_people}.npy'
        if not os.path.exists(path_file):
            paths = []
            for sub_dir in v2_sub_dirs:
                video_path = os.path.join(self.v2_folder, sub_dir, "video_crop_resize_25fps_16k")
                now_paths = [
                    p.relative_to(self.v2_folder) for ext in exts
                    for p in Path(f'{video_path}').glob(f'*.{ext}')
                ]
                for path in now_paths:
                    try:
                        # check if the video is corrupted
                        video = cv2.VideoCapture(os.path.join(self.v2_folder, path))
                        total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
                        ret, frame = video.read()
                        video.release()
                        if total_frames < 5 or ret is False:
                            continue
                    except Exception as e: 
                        print(f"error: {e} in loading {os.path.join(self.v2_folder, path)}")
                        continue
                    paths.append(os.path.join(self.v2_folder, path))

            for sub_dir in v1_sub_dirs:
                video_path = os.path.join(self.v1_folder, sub_dir, "video_crop_resize_25fps_16k")
                # ldmk_path = os.path.join(folder, sub_dir, "video_ldmk")
                now_paths = [
                    p.relative_to(self.v1_folder) for ext in exts
                    for p in Path(f'{video_path}').glob(f'*.{ext}')
                ]
                for path in now_paths:
                    try:
                        video = cv2.VideoCapture(os.path.join(self.v1_folder, path))
                        total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
                        ret, frame = video.read()
                        video.release()
                        if total_frames < 25 or ret is False:
                            continue
                    except Exception as e: 
                        print(f"error: {e} in loading {os.path.join(self.v1_folder, path)}")
                        continue
                    paths.append(os.path.join(self.v1_folder, path))
            np.save(path_file, np.array(paths))
        else:
            paths = list(np.load(path_file, allow_pickle=True))

        self.tk1k_paths = sorted(paths)[:num_people]
        hdtf_paths = sorted([name for name in os.listdir(hdtf_folder) if os.path.isdir(os.path.join(hdtf_folder, name))]) #[:num_people]
        val_names = ["WRA_LamarAlexander0_000", "WRA_MarcoRubio_000", "WRA_ReincePriebus_000", "WRA_ReneeEllmers_000", "WRA_TimScott_000", "WRA_VickyHartzler_000"]
        self.hdtf_paths = []
        self.hdtf_paths_val = []
        for path in hdtf_paths:
            if path in val_names:
                self.hdtf_paths_val.append(path)
            else:
                self.hdtf_paths.append(path)
        
        self.paths = self.tk1k_paths + self.hdtf_paths
        self.indicator = [0] * len(self.tk1k_paths) + [1] * len(self.hdtf_paths)
        
        self.length = len(self.paths)
        if split is None:
            self.offset = 0
        elif split == 'train':
            # last 60k
            self.length = self.length - 10
            self.offset = 10
        elif split == 'val':
            # first 10k
            self.length = 10
            self.offset = 0
        else:
            raise NotImplementedError()

        transform = [
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
        ]
        if do_augment:
            transform.append(transforms.RandomHorizontalFlip())
        if do_transform:
            transform.append(transforms.ToTensor())
        if do_normalize:
            transform.append(
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))
        self.transform = transforms.Compose(transform)

        import dlib
        from onnxruntime import InferenceSession
        from typing import Optional, Dict, Any
        from face_synthetics_training.runtime.face_detector import FaceDetector
        from face_synthetics_training.runtime import landmarks
        from face_synthetics_training.runtime.landmarks import get_landmarks
        from face_synthetics_training.runtime.models import LDMKS_DENSE_MODEL
        from face_synthetics_training.runtime.utils import get_roi_size_multiplier
        from face_synthetics_training.runtime.utils import get_input_image_size

        mouth_idx = list(set([
            1, 4, 37, 56, 60, 63, 64, 65, 66, 67, 68, 69, 122, 157, 159, 165, 204, 209, 210, 214, 218, 221, 222, 223, 316, 319, 352, 371, 375, 378, 379, 380, 381, 382, 383, 384, 436, 471, 473, 479, 518, 523, 524, 528, 532, 535, 536, 537, 629, 633, 636, 637, 640, 657, # upper_lip_and_philtrum
            1, 43, 60, 61, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 88, 122, 202, 204, 209, 218, 219, 220, 221, 222, 223, 224, 225, 226, 227, 228, 316, 358, 375, 376, 378, 379, 380, 381, 382, 383, 384, 385, 386, 387, 388, 389, 436, 516, 518, 523, 532, 533, 534, 535, 536, 537, 538, 539, 540, 541, 542, 636, 637, 640, 657, 660,
            6, 7, 17, 48, 49, 177, 178, 207,      # mouth
            14, 43, 44, 57, 70, 71, 72, 73, 74, 88, 171, 172, 173, 202, 203, 216, 219, 220, 224, 225, 226, 228, 329, 358, 359, 372, 385, 386, 387, 388, 485, 486, 487, 516, 517, 530, 533, 534, 538, 539, 540, 542, 639, 660, 661 # chin_and_lower_lip
            ]))
        face_idx = [0, 1, 2, 515, 516, 3, 518, 4, 5, 6, 8, 523, 9, 10, 524, 15, 16, 17, 18, 19, 532, 533, 534, 535, 536, 537, 538, 539, 540, 541, 542, 22, 23, 29, 30, 31, 32, 549, 550, 551, 552, 553, 554, 43, 35, 36, 556, 44, 48, 49, 517, 568, 569, 
                    56, 57, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 528, 100, 20, 102, 101, 21, 530, 628, 629, 630, 631, 119, 24, 122, 633, 636, 637, 632, 639, 640, 26, 
                    644, 27, 653, 654, 655, 656, 657, 28, 659, 660, 661, 157, 158, 159, 160, 161, 163, 165, 33, 167, 171, 34, 543, 174, 175, 176, 177, 178, 179, 180, 181, 182, 544, 184, 545, 185, 37, 546, 189, 190, 191, 192, 193, 547, 194, 548, 
                    201, 202, 203, 204, 207, 209, 210, 214, 216, 218, 219, 220, 221, 222, 223, 224, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 242, 254, 255, 315, 316, 317, 318, 319, 320, 321, 323, 324, 325, 
                    329, 330, 331, 332, 333, 334, 335, 336, 337, 338, 339, 341, 342, 343, 344, 345, 346, 347, 348, 349, 350, 351, 352, 172, 14, 358, 173, 359, 371, 372, 375, 376, 377, 378, 379, 380, 381, 382, 383, 384, 385, 386, 387, 388, 389, 390, 391, 392, 393, 394, 395, 396, 397, 398, 399, 400, 401, 402, 
                    7, 414, 415, 416, 433, 436, 471, 472, 473, 474, 475, 477, 479, 481, 485, 486, 487, 488, 489, 490, 491, 492, 493, 494, 495, 496, 498, 499, 503, 504, 505, 506, 507, 508]
        sparse_outline_idx = [0, 1, 2, 3, 4, 5, 516, 518, 8, 513, 10, 523, 525, 532, 533, 534, 535, 536, 537, 26, 27, 28, 538, 539, 540, 541, 542, 544, 547, 36, 549, 550, 551, 552, 553, 554, 43, 556, 548, 41, 47, 39, 42, 52, 53, 568, 569, 60, 61, 62, 
                             63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 100, 101, 102, 104, 105, 21, 22, 628, 629, 630, 119, 117, 633, 634, 636, 637, 643, 644, 654, 655, 656, 657, 660, 157, 158, 159, 160, 161, 165, 543, 545, 185, 546, 189, 199, 202, 
                             204, 209, 211, 218, 219, 220, 221, 222, 223, 224, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 242, 254, 255, 259, 267, 271, 272, 273, 276, 277, 315, 316, 317, 318, 319, 320, 323, 573, 325, 336, 337, 341, 342, 343, 351, 354, 579, 356, 357, 358, 362, 
                             581, 367, 368, 375, 376, 377, 378, 379, 380, 381, 382, 383, 384, 385, 386, 387, 388, 389, 390, 391, 392, 393, 394, 395, 396, 397, 398, 399, 400, 401, 402, 587, 586, 414, 415, 416, 417, 418, 429, 431, 433, 434, 471, 472, 473, 474, 475, 479, 499, 503, 508]
        detector = dlib.get_frontal_face_detector()
        predictor = dlib.shape_predictor("/mnt/blob/xxxx/diffusion/shape_predictor_68_face_landmarks.dat")

        class RegressLandmarksFromImages:
            def __init__(
                self,
                sigma_threshold: Optional[float] = None,
                onnx_model: Optional[Path] = LDMKS_DENSE_MODEL,
                never_from_sequence: Optional[bool] = False,
                override_detector_ags: Optional[Dict[str, Any]] = None,
            ):
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

        def process_one(img_256):
            # img_256 = Image.open(img_file).convert('RGB').resize((256,256))
            # img_256 = Image.fromarray(array).convert('RGB').resize((256,256))
            # sparse ldmks
            img_array = np.asarray(img_256)
            faces = detector(img_array)
            if self.dense_ldmk or self.only_mouth or len(faces) != 1 or self.sparse_outline:
                sparse_coords = []
            else:
                sparse_ldmks = predictor(img_array, faces[0])
                sparse_coords = np.zeros((sparse_ldmks.num_parts, 2), dtype=np.uint8)
                for i in range(sparse_ldmks.num_parts):
                    sparse_coords[i] = (sparse_ldmks.part(i).x, sparse_ldmks.part(i).y) # 68 sparse ldmks

            dense_img_256, sigma_img = regress_landmarks.regress_landmarks_demo(np.asarray(img_256)/255.)
            if not self.dense_ldmk:
                new_dense_img_256 = []
                if self.only_mouth:
                    preserve_id = face_idx
                elif self.sparse_outline:
                    preserve_id = sparse_outline_idx
                else:
                    preserve_id = mouth_idx
                for i in preserve_id:
                    new_dense_img_256.append(dense_img_256[i])
                new_dense_img_256.extend(sparse_coords)
                dense_img_256 = np.array(new_dense_img_256)
            
            ldmks = draw_kp(dense_img_256, size=(256,256), is_connect=False, color=(255,255,255))
            return ldmks

        def random_hdtf_frame(hdtf_path):
            files = os.listdir(hdtf_path)
            file_num = len(files)
            random_frame_number, random_pos_number = random.sample(range(0, file_num - 1), 2)

            img = Image.open(os.path.join(hdtf_path, files[random_frame_number])).convert('RGB')
            img_pos = Image.open(os.path.join(hdtf_path, files[random_pos_number])).convert('RGB')
            ldmk_img = Image.fromarray(process_one(img.resize((256,256))))
            ldmk_img_pos = Image.fromarray(process_one(img_pos.resize((256,256))))
            return img, img_pos, ldmk_img, ldmk_img_pos

        def random_video_frame(video_path):
            # ldmk_video_path = video_path.replace("video_crop_resize_25fps_16k", "video_ldmk")
            video = cv2.VideoCapture(video_path)
            # ldmk_video = cv2.VideoCapture(ldmk_video_path)
            # total_frames_ldmk = int(ldmk_video.get(cv2.CAP_PROP_FRAME_COUNT))
            total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
            # total_frames = min(total_frames, total_frames_ldmk)
            random_frame_number, random_pos_number = random.sample(range(0, total_frames - 1), 2)
            video.set(cv2.CAP_PROP_POS_FRAMES, random_frame_number)
            ret, frame = video.read()
            video.set(cv2.CAP_PROP_POS_FRAMES, random_pos_number)
            ret, frame_pos = video.read()
            video.release()

            ldmk_frame = process_one(Image.fromarray(frame).convert('RGB').resize((256,256)))
            ldmk_frame_pos = process_one(Image.fromarray(frame_pos).convert('RGB').resize((256,256)))

            pil_frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            pil_frame_pos = Image.fromarray(cv2.cvtColor(frame_pos, cv2.COLOR_BGR2RGB))
            ldmk_pil_frame = Image.fromarray(cv2.cvtColor(ldmk_frame, cv2.COLOR_BGR2RGB))
            ldmk_pil_frame_pos = Image.fromarray(cv2.cvtColor(ldmk_frame_pos, cv2.COLOR_BGR2RGB))
            return pil_frame, pil_frame_pos, ldmk_pil_frame, ldmk_pil_frame_pos
        
        self.random_hdtf_frame = random_hdtf_frame
        self.random_video_frame = random_video_frame
    
    def __len__(self):
        return self.length

    def __getitem__(self, index):
        offset_index = index + self.offset
        # if offset_index >= len(self.paths):
        #     if offset_index % len(self.paths) + self.offset < len(self.paths):
        #         offset_index = offset_index % len(self.paths) + self.offset
        #     else:
        #         offset_index = offset_index % len(self.paths)

        
        try:
            if self.indicator[offset_index] == 0:
                img, img_pos, ldmk_img, ldmk_img_pos = self.random_video_frame(self.paths[offset_index])
            else:
                path = os.path.join(self.hdtf_folder, self.paths[offset_index])
                img, img_pos, ldmk_img, ldmk_img_pos = self.random_hdtf_frame(path)
        except:
            print(f"Retry file {self.paths[offset_index]}")
            # remove the corrupted video
            # self.paths.pop(offset_index)
            # self.indicator.pop(offset_index)
            # if index >= len(self.paths):
            #     index = len(self.paths) - 1
            return self.__getitem__(index)
        

        img = self.transform(img)
        img_pos = self.transform(img_pos)
        ldmk_img = self.transform(ldmk_img)
        ldmk_img_pos = self.transform(ldmk_img_pos)
        return {'image': img, 'positive_img': img_pos, 
                    'ldmk_img': ldmk_img, 'ldmk_pos_img': ldmk_img_pos,
                    'index': index}


class HDTFandCCDDataset(Dataset):
    def __init__(
        self,
        hdtf_folder,
        tk1k_folder,
        image_size=256,
        do_augment: bool = False,
        do_transform: bool = True,
        do_normalize: bool = True,
        split=None,
        exts=['mp4', 'MP4'],
        has_subdir: bool = True,
        num_people: int = -1,
        frame_len: int = 1500,
        use_contrastive: bool = False,
        contrast_mode: str = "appearance", # motion or appearance
    ):
        super().__init__()
        self.image_size = image_size
        self.hdtf_folder = hdtf_folder

        # CCD dataset
        self.v2_folder = tk1k_folder
        self.v1_folder = tk1k_folder.replace("CCDv2_processed", "CCDv1_processed")

        self.image_size = image_size
        self.contrast_mode = contrast_mode

        v1_sub_dirs = sorted(os.listdir(self.v1_folder))
        v2_sub_dirs = sorted(os.listdir(self.v2_folder))
        
        error_videos = ["CC_part_12_4/video_crop_resize_25fps_16k/0150_06_0.MP4",
                        "CC_part_5_1/video_crop_resize_25fps_16k/1421_10_0.MP4",
                        "CC_part_6_4/video_crop_resize_25fps_16k/0290_07_0.MP4",
                        "CC_part_2_2/video_crop_resize_25fps_16k/1366_12_0.MP4",
                        "CC_part_17_4/video_crop_resize_25fps_16k/0423_10_2.MP4",
                        "CC_part_1_1/video_crop_resize_25fps_16k/1157_14_1.MP4",
                        "CC_part_6_2/video_crop_resize_25fps_16k/1805_12_1.MP4",
                        "CC_part_7_3/video_crop_resize_25fps_16k/1030_13_0.MP4",
                        "CC_part_10_3/video_crop_resize_25fps_16k/0001_07_0.MP4",
                        "CC_part_14_4/video_crop_resize_25fps_16k/0489_07_0.MP4",
                        "CC_part_16_1/video_crop_resize_25fps_16k/0843_03_0.MP4",
                        "CC_part_19_2/video_crop_resize_25fps_16k/0176_10_1.MP4",
                        "CC_part_1_4/video_crop_resize_25fps_16k/0129_05_0.MP4",
                        "CC_part_4_2/video_crop_resize_25fps_16k/1616_12_0.MP4",
                        "CC_part_2_3/video_crop_resize_25fps_16k/1734_12_0.MP4",
                        "CC_part_9_4/video_crop_resize_25fps_16k/0081_05_0.MP4",
                        "CC_part_14_2/video_crop_resize_25fps_16k/1013_09_0.MP4",
                        "CC_part_2_3/video_crop_resize_25fps_16k/1734_12_0.MP4",
                        "CC_part_3_2/video_crop_resize_25fps_16k/1500_07_0.MP4",
                        "CC_part_17_3/video_crop_resize_25fps_16k/0246_03_1.MP4",
                        "CC_part_15_3/video_crop_resize_25fps_16k/0443_09_0.MP4",
                        "CC_part_4_4/video_crop_resize_25fps_16k/1031_13_0.MP4",
                        "CC_part_4_2/video_crop_resize_25fps_16k/1687_10_3.MP4",
                        "CC_part_12_1/video_crop_resize_25fps_16k/1377_09_1.MP4",
                        "CC_part_13_4/video_crop_resize_25fps_16k/0435_05_0.MP4",
                        "CC_part_5_2/video_crop_resize_25fps_16k/1787_13_0.MP4",
                        "CC_part_1_4/video_crop_resize_25fps_16k/0519_05_0.MP4",
                        "CC_part_2_3/video_crop_resize_25fps_16k/1379_10_3.MP4",
                        "CC_part_13_3/video_crop_resize_25fps_16k/0092_09_0.MP4",
                        "CC_part_3_4/video_crop_resize_25fps_16k/0348_10_0.MP4",
                        "CC_part_19_1/video_crop_resize_25fps_16k/0059_10_0.MP4",
                        "CC_part_8_1/video_crop_resize_25fps_16k/1713_11_0.MP4",
                        "CC_part_15_4/video_crop_resize_25fps_16k/0337_10_0.MP4",
                        "CC_part_11_2/video_crop_resize_25fps_16k/0692_04_0.MP4",
                        "CC_part_8_1/video_crop_resize_25fps_16k/1842_13_0.MP4",
                        ]

        if not os.path.exists(f'{self.v2_folder}/paths.npy'):
            paths = []
            for sub_dir in v2_sub_dirs:
                video_path = os.path.join(self.v2_folder, sub_dir, "video_crop_resize_25fps_16k")
                now_paths = [
                    p.relative_to(self.v2_folder) for ext in exts
                    for p in Path(f'{video_path}').glob(f'*.{ext}')
                ]
                for path in now_paths:
                    if str(path) in error_videos:
                        continue
                    video = cv2.VideoCapture(os.path.join(self.v2_folder, path))
                    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
                    video.release()
                    if total_frames < 25:
                        continue
                    paths.append(os.path.join(self.v2_folder, path))

            for sub_dir in v1_sub_dirs:
                video_path = os.path.join(self.v1_folder, sub_dir, "video_crop_resize_25fps_16k")
                # ldmk_path = os.path.join(folder, sub_dir, "video_ldmk")
                now_paths = [
                    p.relative_to(self.v1_folder) for ext in exts
                    for p in Path(f'{video_path}').glob(f'*.{ext}')
                ]
                for path in now_paths:
                    if str(path) in error_videos:
                        continue
                    video = cv2.VideoCapture(os.path.join(self.v1_folder, path))
                    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
                    video.release()
                    if total_frames < 25:
                        continue
                    paths.append(os.path.join(self.v1_folder, path))
            
            np.save(f'{self.v2_folder}/paths.npy', np.array(paths))
        else:
            paths = list(np.load(f'{self.v2_folder}/paths.npy', allow_pickle=True))

        self.tk1k_paths = sorted(paths)[:num_people]
        hdtf_paths = sorted([name for name in os.listdir(hdtf_folder) if os.path.isdir(os.path.join(hdtf_folder, name))]) #[:num_people]
        val_names = ["WRA_LamarAlexander0_000", "WRA_MarcoRubio_000", "WRA_ReincePriebus_000", "WRA_ReneeEllmers_000", "WRA_TimScott_000", "WRA_VickyHartzler_000"]
        self.hdtf_paths = []
        self.hdtf_paths_val = []
        for path in hdtf_paths:
            if path in val_names:
                self.hdtf_paths_val.append(path)
            else:
                self.hdtf_paths.append(path)
        self.paths = self.tk1k_paths + self.hdtf_paths
        self.indicator = [0] * len(self.tk1k_paths) + [1] * len(self.hdtf_paths)
        
        self.length = len(self.paths)
        if split is None:
            self.offset = 0
        elif split == 'train':
            # last 60k
            self.length = self.length - 10
            self.offset = 10
        elif split == 'val':
            # first 10k
            self.length = 10
            self.offset = 0
        else:
            raise NotImplementedError()

        transform = [
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
        ]
        if do_augment:
            transform.append(transforms.RandomHorizontalFlip())
        if do_transform:
            transform.append(transforms.ToTensor())
        if do_normalize:
            transform.append(
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))
        self.transform = transforms.Compose(transform)

        def random_hdtf_frame(hdtf_path):
            files = os.listdir(hdtf_path)
            ldmk_path = hdtf_path.replace("frames", "ldmks")
            ldmk_files = os.listdir(ldmk_path)
            file_num = min(len(files), len(ldmk_files))
            random_frame_number, random_pos_number = random.sample(range(0, file_num - 1), 2)

            img = Image.open(os.path.join(hdtf_path, files[random_frame_number])).convert('RGB')
            img_pos = Image.open(os.path.join(hdtf_path, files[random_pos_number])).convert('RGB')
            ldmk_img = Image.open(os.path.join(ldmk_path, files[random_frame_number])).convert('RGB')
            ldmk_img_pos = Image.open(os.path.join(ldmk_path, files[random_pos_number])).convert('RGB')
            return img, img_pos, ldmk_img, ldmk_img_pos

        def random_video_frame(video_path):
            ldmk_video_path = video_path.replace("video_crop_resize_25fps_16k", "video_ldmk")
            video = cv2.VideoCapture(video_path)
            ldmk_video = cv2.VideoCapture(ldmk_video_path)
            total_frames_ldmk = int(ldmk_video.get(cv2.CAP_PROP_FRAME_COUNT))
            total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
            total_frames = min(total_frames, total_frames_ldmk)
            random_frame_number, random_pos_number = random.sample(range(0, total_frames - 1), 2)
            video.set(cv2.CAP_PROP_POS_FRAMES, random_frame_number)
            ret, frame = video.read()
            video.set(cv2.CAP_PROP_POS_FRAMES, random_pos_number)
            ret, frame_pos = video.read()
            video.release()

            ldmk_video.set(cv2.CAP_PROP_POS_FRAMES, random_frame_number)
            ret, ldmk_frame = ldmk_video.read()
            ldmk_video.set(cv2.CAP_PROP_POS_FRAMES, random_pos_number)
            ret, ldmk_frame_pos = ldmk_video.read()
            ldmk_video.release()

            pil_frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            pil_frame_pos = Image.fromarray(cv2.cvtColor(frame_pos, cv2.COLOR_BGR2RGB))
            ldmk_pil_frame = Image.fromarray(cv2.cvtColor(ldmk_frame, cv2.COLOR_BGR2RGB))
            ldmk_pil_frame_pos = Image.fromarray(cv2.cvtColor(ldmk_frame_pos, cv2.COLOR_BGR2RGB))
            return pil_frame, pil_frame_pos, ldmk_pil_frame, ldmk_pil_frame_pos
        
        self.random_hdtf_frame = random_hdtf_frame
        self.random_video_frame = random_video_frame
    
    def __len__(self):
        return self.length

    def __getitem__(self, index):
        offset_index = index + self.offset
        # if offset_index >= len(self.paths):
        #     if offset_index % len(self.paths) + self.offset < len(self.paths):
        #         offset_index = offset_index % len(self.paths) + self.offset
        #     else:
        #         offset_index = offset_index % len(self.paths)

        
        try:
            if self.indicator[offset_index] == 0:
                img, img_pos, ldmk_img, ldmk_img_pos = self.random_video_frame(self.paths[offset_index])
            else:
                path = os.path.join(self.hdtf_folder, self.paths[offset_index])
                img, img_pos, ldmk_img, ldmk_img_pos = self.random_hdtf_frame(path)
        except:
            print(f"Retry file {self.paths[offset_index]}")
            # remove the corrupted video
            # self.paths.pop(offset_index)
            # self.indicator.pop(offset_index)
            # if index >= len(self.paths):
            #     index = len(self.paths) - 1
            return self.__getitem__(index)
        

        img = self.transform(img)
        img_pos = self.transform(img_pos)
        ldmk_img = self.transform(ldmk_img)
        ldmk_img_pos = self.transform(ldmk_img_pos)
        return {'image': img, 'positive_img': img_pos, 
                    'ldmk_img': ldmk_img, 'ldmk_pos_img': ldmk_img_pos,
                    'index': index}

class HDTFandCCDTemporalDataset(Dataset):
    def __init__(
        self,
        hdtf_folder,
        tk1k_folder,
        image_size=256,
        do_augment: bool = False,
        do_transform: bool = True,
        do_normalize: bool = True,
        split=None,
        exts=['mp4', 'MP4'],
        has_subdir: bool = True,
        num_people: int = -1,
        frame_len: int = 1500,
        use_contrastive: bool = False,
        contrast_mode: str = "appearance", # motion or appearance
        temporal_len: int = 5,
    ):
        super().__init__()
        self.image_size = image_size
        self.hdtf_folder = hdtf_folder

        # CCD dataset
        self.v2_folder = tk1k_folder
        self.v1_folder = tk1k_folder.replace("CCDv2_processed", "CCDv1_processed")

        self.image_size = image_size
        self.contrast_mode = contrast_mode
        self.temporal_len = temporal_len

        v1_sub_dirs = sorted(os.listdir(self.v1_folder))
        v2_sub_dirs = sorted(os.listdir(self.v2_folder))

        error_videos = []

        if not os.path.exists(f'{self.v2_folder}/paths.npy'):
            paths = []
            for sub_dir in v2_sub_dirs:
                video_path = os.path.join(self.v2_folder, sub_dir, "video_crop_resize_25fps_16k")
                now_paths = [
                    p.relative_to(self.v2_folder) for ext in exts
                    for p in Path(f'{video_path}').glob(f'*.{ext}')
                ]
                for path in now_paths:
                    if str(path) in error_videos:
                        continue
                    video = cv2.VideoCapture(os.path.join(self.v2_folder, path))
                    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
                    video.release()
                    if total_frames < 25:
                        continue
                    paths.append(os.path.join(self.v2_folder, path))

            for sub_dir in v1_sub_dirs:
                video_path = os.path.join(self.v1_folder, sub_dir, "video_crop_resize_25fps_16k")
                # ldmk_path = os.path.join(folder, sub_dir, "video_ldmk")
                now_paths = [
                    p.relative_to(self.v1_folder) for ext in exts
                    for p in Path(f'{video_path}').glob(f'*.{ext}')
                ]
                for path in now_paths:
                    if str(path) in error_videos:
                        continue
                    video = cv2.VideoCapture(os.path.join(self.v1_folder, path))
                    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
                    video.release()
                    if total_frames < 25:
                        continue
                    paths.append(os.path.join(self.v1_folder, path))
            
            np.save(f'{self.v2_folder}/paths.npy', np.array(paths))
        else:
            paths = list(np.load(f'{self.v2_folder}/paths.npy', allow_pickle=True))

        self.tk1k_paths = sorted(paths)[:num_people]
        hdtf_paths = sorted([name for name in os.listdir(hdtf_folder) if os.path.isdir(os.path.join(hdtf_folder, name))]) #[:num_people]
        val_names = ["WRA_LamarAlexander0_000", "WRA_MarcoRubio_000", "WRA_ReincePriebus_000", "WRA_ReneeEllmers_000", "WRA_TimScott_000", "WRA_VickyHartzler_000"]
        self.hdtf_paths = []
        self.hdtf_paths_val = []
        for path in hdtf_paths:
            if path in val_names:
                self.hdtf_paths_val.append(path)
            else:
                self.hdtf_paths.append(path)
        self.paths = self.tk1k_paths + self.hdtf_paths
        self.indicator = [0] * len(self.tk1k_paths) + [1] * len(self.hdtf_paths)
        
        self.length = len(self.paths)
        if split is None:
            self.offset = 0
        elif split == 'train':
            # last 60k
            self.length = self.length - 10
            self.offset = 10
        elif split == 'val':
            # first 10k
            self.length = 10
            self.offset = 0
        else:
            raise NotImplementedError()

        transform = [
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
        ]
        if do_augment:
            transform.append(transforms.RandomHorizontalFlip())
        if do_transform:
            transform.append(transforms.ToTensor())
        if do_normalize:
            transform.append(
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))
        self.transform = transforms.Compose(transform)

        def random_hdtf_frame(hdtf_path):
            files = os.listdir(hdtf_path)
            ldmk_path = hdtf_path.replace("frames", "ldmks")
            ldmk_files = os.listdir(ldmk_path)
            file_num = min(len(files), len(ldmk_files))
            start = random.randint(0, file_num - temporal_len - 1)
            frames = []
            ldmk_frames = []
            for _ in range(temporal_len):
                img = Image.open(os.path.join(hdtf_path, files[start])).convert('RGB')
                img = self.transform(img)
                frames.append(img)

                ldmk_img = Image.open(os.path.join(ldmk_path, files[start])).convert('RGB')
                ldmk_img = self.transform(ldmk_img)
                ldmk_frames.append(ldmk_img)
                
                start += 1

            start = random.randint(0, file_num - temporal_len - 1)
            pos_frames = []
            ldmk_pos_frames = []
            for _ in range(temporal_len):
                img = Image.open(os.path.join(hdtf_path, files[start])).convert('RGB')
                img = self.transform(img)
                pos_frames.append(img)

                ldmk_img = Image.open(os.path.join(ldmk_path, files[start])).convert('RGB')
                ldmk_img = self.transform(ldmk_img)
                ldmk_pos_frames.append(ldmk_img)
                
                start += 1

            return frames, pos_frames, ldmk_frames, ldmk_pos_frames

        def random_video_frame(video_path):
            ldmk_video_path = video_path.replace("video_crop_resize_25fps_16k", "video_ldmk")
            video = cv2.VideoCapture(video_path)
            ldmk_video = cv2.VideoCapture(ldmk_video_path)
            total_frames_ldmk = int(ldmk_video.get(cv2.CAP_PROP_FRAME_COUNT))
            total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
            total_frames = min(total_frames, total_frames_ldmk)

            frames = []
            start = random.randint(0, total_frames - temporal_len - 1)
            video.set(cv2.CAP_PROP_POS_FRAMES, start)
            for _ in range(temporal_len):
                ret, frame = video.read()
                frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                frame = self.transform(frame)
                frames.append(frame)
            
            pos_frames = []
            pos_start = random.randint(0, total_frames - temporal_len - 1)
            video.set(cv2.CAP_PROP_POS_FRAMES, pos_start)
            for _ in range(temporal_len):
                ret, frame = video.read()
                frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                frame = self.transform(frame)
                pos_frames.append(frame)
            video.release()

            ldmk_frames = []
            ldmk_video.set(cv2.CAP_PROP_POS_FRAMES, start)
            for _ in range(temporal_len):
                ret, frame = ldmk_video.read()
                frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                frame = self.transform(frame)
                ldmk_frames.append(frame)
            
            ldmk_pos_frames = []
            ldmk_video.set(cv2.CAP_PROP_POS_FRAMES, pos_start)
            for _ in range(temporal_len):
                ret, frame = ldmk_video.read()
                frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                frame = self.transform(frame)
                ldmk_pos_frames.append(frame)
            ldmk_video.release()

            return frames, pos_frames, ldmk_frames, ldmk_pos_frames
        
        self.random_hdtf_frame = random_hdtf_frame
        self.random_video_frame = random_video_frame
    
    def __len__(self):
        return self.length

    def __getitem__(self, index):
        temporal_list = []
        for i in range(index * self.temporal_len, (index+1) * self.temporal_len):
            temporal_list.append(super().__getitem__(i))
        return {'image': torch.stack([temporal_list[i]['image'] for i in range(self.temporal_len)]),
                'positive_img': torch.stack([temporal_list[i]['positive_img'] for i in range(self.temporal_len)]), 
                'ldmk_img': torch.stack([temporal_list[i]['ldmk_img'] for i in range(self.temporal_len)]),
                'ldmk_pos_img': torch.stack([temporal_list[i]['ldmk_pos_img'] for i in range(self.temporal_len)]),
                'index': index}

    def __getitem__(self, index):
        offset_index = index + self.offset

        try:
            if self.indicator[offset_index] == 0:
                frames, pos_frames, ldmk_frames, ldmk_pos_frames = self.random_video_frame(self.paths[offset_index])
            else:
                path = os.path.join(self.hdtf_folder, self.paths[offset_index])
                frames, pos_frames, ldmk_frames, ldmk_pos_frames = self.random_hdtf_frame(path)
        except:
            print(f"Retry file {self.paths[offset_index]}")
            return self.__getitem__(index)
        
        return {'image': torch.stack(frames), 'positive_img': torch.stack(pos_frames), 
                'ldmk_img': torch.stack(ldmk_frames), 'ldmk_pos_img': torch.stack(ldmk_pos_frames),
                'index': index}

class HDTFCCDFACEDataset(Dataset):
    def __init__(
        self,
        hdtf_folder,
        tk1k_folder,
        hhfq_folder = None,
        celeba_folder = None,
        image_size=256,
        face_fold=2000,
        do_augment: bool = False,
        do_transform: bool = True,
        do_normalize: bool = True,
        split=None,
        exts=['mp4', 'MP4'],
        has_subdir: bool = True,
        num_people: int = -1,
        frame_len: int = 1500,
        use_contrastive: bool = False,
        contrast_mode: str = "appearance", # motion or appearance
    ):
        super().__init__()
        self.image_size = image_size
        self.hdtf_folder = hdtf_folder

        # face dataset
        self.hhfq_folder = hhfq_folder
        self.celeba_folder = celeba_folder

        # CCD dataset
        self.v2_folder = tk1k_folder
        self.v1_folder = tk1k_folder.replace("CCDv2_processed", "CCDv1_processed")

        self.image_size = image_size
        self.contrast_mode = contrast_mode

        v1_sub_dirs = sorted(os.listdir(self.v1_folder))
        v2_sub_dirs = sorted(os.listdir(self.v2_folder))
        
        error_videos = []

        if not os.path.exists(f'{self.v2_folder}/paths.npy'):
            paths = []
            for sub_dir in v2_sub_dirs:
                video_path = os.path.join(self.v2_folder, sub_dir, "video_crop_resize_25fps_16k")
                now_paths = [
                    p.relative_to(self.v2_folder) for ext in exts
                    for p in Path(f'{video_path}').glob(f'*.{ext}')
                ]
                for path in now_paths:
                    if str(path) in error_videos:
                        continue
                    video = cv2.VideoCapture(os.path.join(self.v2_folder, path))
                    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
                    video.release()
                    if total_frames < 25:
                        continue
                    paths.append(os.path.join(self.v2_folder, path))

            for sub_dir in v1_sub_dirs:
                video_path = os.path.join(self.v1_folder, sub_dir, "video_crop_resize_25fps_16k")
                # ldmk_path = os.path.join(folder, sub_dir, "video_ldmk")
                now_paths = [
                    p.relative_to(self.v1_folder) for ext in exts
                    for p in Path(f'{video_path}').glob(f'*.{ext}')
                ]
                for path in now_paths:
                    if str(path) in error_videos:
                        continue
                    video = cv2.VideoCapture(os.path.join(self.v1_folder, path))
                    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
                    video.release()
                    if total_frames < 25:
                        continue
                    paths.append(os.path.join(self.v1_folder, path))
            
            np.save(f'{self.v2_folder}/paths.npy', np.array(paths))
        else:
            paths = list(np.load(f'{self.v2_folder}/paths.npy', allow_pickle=True))

        self.tk1k_paths = sorted(paths)[:num_people]
        hdtf_paths = sorted([name for name in os.listdir(hdtf_folder) if os.path.isdir(os.path.join(hdtf_folder, name))]) #[:num_people]
        val_names = ["WRA_LamarAlexander0_000", "WRA_MarcoRubio_000", "WRA_ReincePriebus_000", "WRA_ReneeEllmers_000", "WRA_TimScott_000", "WRA_VickyHartzler_000"]
        self.hdtf_paths = []
        self.hdtf_paths_val = []
        for path in hdtf_paths:
            if path in val_names:
                self.hdtf_paths_val.append(path)
            else:
                self.hdtf_paths.append(path)
        
        # face dataset
        face_paths = []
        if self.hhfq_folder is not None:
            if not os.path.exists(f'{self.hhfq_folder}/paths.npy'):
                paths = []
                files = sorted([f.name for ext in ['jpg', 'png'] for f in Path(self.hhfq_folder).glob(f'*.{ext}')])
                for file in files:
                    paths.append(f'{file}')
                paths = sorted(paths)
                np.save(f'{self.hhfq_folder}/paths.npy', np.array(paths))
            else:
                paths = list(np.load(f'{self.hhfq_folder}/paths.npy', allow_pickle=True))
            paths = [os.path.join(self.hhfq_folder, f) for f in paths]
            face_paths = face_paths + paths

        if self.celeba_folder is not None:
            if not os.path.exists(f'{self.celeba_folder}/paths.npy'):
                paths = []
                files = sorted([f.name for ext in ['jpg', 'png'] for f in Path(self.celeba_folder).glob(f'*.{ext}')])
                for file in files:
                    paths.append(f'{file}')
                paths = sorted(paths)
                np.save(f'{self.celeba_folder}/paths.npy', np.array(paths))
            else:
                paths = list(np.load(f'{self.celeba_folder}/paths.npy', allow_pickle=True))
            paths = [os.path.join(self.celeba_folder, f) for f in paths]
            face_paths = face_paths + paths
        
        folded_face_paths = []
        # split face_paths into {face_fold} folds
        for i in range(face_fold):
            folded_face_paths.append(face_paths[i::face_fold])

        self.paths = self.tk1k_paths + self.hdtf_paths + folded_face_paths
        self.indicator = [0] * len(self.tk1k_paths) + [1] * len(self.hdtf_paths) + [2] * len(folded_face_paths)
        
        self.length = len(self.paths)
        if split is None:
            self.offset = 0
        elif split == 'train':
            # last 60k
            self.length = self.length - 10
            self.offset = 10
        elif split == 'val':
            # first 10k
            self.length = 10
            self.offset = 0
        else:
            raise NotImplementedError()

        transform = [
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
        ]
        if do_augment:
            transform.append(transforms.RandomHorizontalFlip())
        if do_transform:
            transform.append(transforms.ToTensor())
        if do_normalize:
            transform.append(
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))
        self.transform = transforms.Compose(transform)

        def random_hdtf_frame(hdtf_path):
            files = os.listdir(hdtf_path)
            ldmk_path = hdtf_path.replace("frames", "ldmks")
            ldmk_files = os.listdir(ldmk_path)
            file_num = min(len(files), len(ldmk_files))
            random_frame_number, random_pos_number = random.sample(range(0, file_num - 1), 2)

            img = Image.open(os.path.join(hdtf_path, files[random_frame_number])).convert('RGB')
            img_pos = Image.open(os.path.join(hdtf_path, files[random_pos_number])).convert('RGB')
            ldmk_img = Image.open(os.path.join(ldmk_path, files[random_frame_number])).convert('RGB')
            ldmk_img_pos = Image.open(os.path.join(ldmk_path, files[random_pos_number])).convert('RGB')
            return img, img_pos, ldmk_img, ldmk_img_pos

        def random_video_frame(video_path):
            ldmk_video_path = video_path.replace("video_crop_resize_25fps_16k", "video_ldmk")
            video = cv2.VideoCapture(video_path)
            ldmk_video = cv2.VideoCapture(ldmk_video_path)
            total_frames_ldmk = int(ldmk_video.get(cv2.CAP_PROP_FRAME_COUNT))
            total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
            total_frames = min(total_frames, total_frames_ldmk)
            random_frame_number, random_pos_number = random.sample(range(0, total_frames - 1), 2)
            video.set(cv2.CAP_PROP_POS_FRAMES, random_frame_number)
            ret, frame = video.read()
            video.set(cv2.CAP_PROP_POS_FRAMES, random_pos_number)
            ret, frame_pos = video.read()
            video.release()

            ldmk_video.set(cv2.CAP_PROP_POS_FRAMES, random_frame_number)
            ret, ldmk_frame = ldmk_video.read()
            ldmk_video.set(cv2.CAP_PROP_POS_FRAMES, random_pos_number)
            ret, ldmk_frame_pos = ldmk_video.read()
            ldmk_video.release()

            pil_frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            pil_frame_pos = Image.fromarray(cv2.cvtColor(frame_pos, cv2.COLOR_BGR2RGB))
            ldmk_pil_frame = Image.fromarray(cv2.cvtColor(ldmk_frame, cv2.COLOR_BGR2RGB))
            ldmk_pil_frame_pos = Image.fromarray(cv2.cvtColor(ldmk_frame_pos, cv2.COLOR_BGR2RGB))
            return pil_frame, pil_frame_pos, ldmk_pil_frame, ldmk_pil_frame_pos
        
        self.random_hdtf_frame = random_hdtf_frame
        self.random_video_frame = random_video_frame
    
    def __len__(self):
        return self.length

    def __getitem__(self, index):
        offset_index = index + self.offset  

        try:
            if self.indicator[offset_index] == 0:
                img, img_pos, ldmk_img, ldmk_img_pos = self.random_video_frame(self.paths[offset_index])
            elif self.indicator[offset_index] == 1:
                path = os.path.join(self.hdtf_folder, self.paths[offset_index])
                img, img_pos, ldmk_img, ldmk_img_pos = self.random_hdtf_frame(path)
            elif self.indicator[offset_index] == 2:
                face_fold = self.paths[offset_index]
                path = random.choice(face_fold)
                img = Image.open(path).convert('RGB')
                img_pos = img
                ldmk_img = Image.open(path.replace("frames", "ldmks")).convert('RGB')
                ldmk_img_pos = ldmk_img
            else:
                raise NotImplementedError()
        except:
            print(f"Retry file {self.paths[offset_index]}")
            return self.__getitem__(index)
        

        img = self.transform(img)
        img_pos = self.transform(img_pos)
        ldmk_img = self.transform(ldmk_img)
        ldmk_img_pos = self.transform(ldmk_img_pos)
        return {'image': img, 'positive_img': img_pos, 
                    'ldmk_img': ldmk_img, 'ldmk_pos_img': ldmk_img_pos,
                    'index': index}


class HDTFandTK1KHDataset(Dataset):
    def __init__(
        self,
        hdtf_folder,
        tk1k_folder,
        image_size=256,
        do_augment: bool = True,
        do_transform: bool = True,
        do_normalize: bool = True,
        split=None,
        exts=['mp4'],
        has_subdir: bool = True,
        num_people: int = -1,
        frame_len: int = 1500,
        use_contrastive: bool = False,
        contrast_mode: str = "motion", # motion or appearance
    ):
        super().__init__()
        self.image_size = image_size
        # self.use_contrastive = use_contrastive
        # self.contrast_mode = contrast_mode
        # self.tk1k_folder = tk1k_folder
        self.hdtf_folder = hdtf_folder

        # CCD dataset
        self.v2_folder = tk1k_folder
        self.v1_folder = tk1k_folder.replace("CCDv2_processed", "CCDv1_processed")

        self.image_size = image_size
        self.contrast_mode = contrast_mode

        v1_sub_dirs = sorted(os.listdir(self.v1_folder))
        v2_sub_dirs = sorted(os.listdir(self.v2_folder))
        
        error_videos = ["CC_part_12_4/video_crop_resize_25fps_16k/0150_06_0.MP4",
                        "CC_part_5_1/video_crop_resize_25fps_16k/1421_10_0.MP4",
                        "CC_part_6_4/video_crop_resize_25fps_16k/0290_07_0.MP4",
                        "CC_part_2_2/video_crop_resize_25fps_16k/1366_12_0.MP4",
                        "CC_part_17_4/video_crop_resize_25fps_16k/0423_10_2.MP4",
                        "CC_part_1_1/video_crop_resize_25fps_16k/1157_14_1.MP4",
                        "CC_part_6_2/video_crop_resize_25fps_16k/1805_12_1.MP4",
                        "CC_part_7_3/video_crop_resize_25fps_16k/1030_13_0.MP4",
                        "CC_part_10_3/video_crop_resize_25fps_16k/0001_07_0.MP4",
                        "CC_part_14_4/video_crop_resize_25fps_16k/0489_07_0.MP4",
                        "CC_part_16_1/video_crop_resize_25fps_16k/0843_03_0.MP4",
                        "CC_part_19_2/video_crop_resize_25fps_16k/0176_10_1.MP4",
                        "CC_part_1_4/video_crop_resize_25fps_16k/0129_05_0.MP4",
                        ]

        if not os.path.exists(f'{self.v2_folder}/frame_buckets.npy'):
            paths = []
            for sub_dir in v2_sub_dirs:
                video_path = os.path.join(self.v2_folder, sub_dir, "video_crop_resize_25fps_16k")
                # ldmk_path = os.path.join(folder, sub_dir, "video_ldmk")
                now_paths = [
                    p.relative_to(self.v2_folder) for ext in exts
                    for p in Path(f'{video_path}').glob(f'*.{ext}')
                ]
                for path in now_paths:
                    if str(path) in error_videos:
                        continue
                    paths.append(os.path.join(self.v2_folder, path))
            
            for sub_dir in v1_sub_dirs:
                video_path = os.path.join(self.v1_folder, sub_dir, "video_crop_resize_25fps_16k")
                # ldmk_path = os.path.join(folder, sub_dir, "video_ldmk")
                now_paths = [
                    p.relative_to(self.v1_folder) for ext in exts
                    for p in Path(f'{video_path}').glob(f'*.{ext}')
                ]
                for path in now_paths:
                    if str(path) in error_videos:
                        continue
                    paths.append(os.path.join(self.v1_folder, path))

            paths = sorted(paths)
            self.tk1k_paths = []
            self.frame_buckets = []
            accumulate_frames = 0
            for i in range(len(paths)):
                try:
                    video = cv2.VideoCapture(paths[i])
                    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
                    video.release()
                    if total_frames < 25:
                        continue
                    accumulate_frames += total_frames
                    self.tk1k_paths.append(paths[i])
                    self.frame_buckets.append(accumulate_frames)
                except:
                    logging.info(f'Error in {paths[i]}')
                    logging.info(traceback.format_exc())
                    print(f'Error in {paths[i]}')
                    
            np.save(f'{self.v2_folder}/frame_buckets.npy', np.array(self.frame_buckets))
            np.save(f'{self.v2_folder}/paths.npy', np.array(self.tk1k_paths))
        else:
            self.tk1k_paths = np.load(f'{self.v2_folder}/paths.npy', allow_pickle=True)
            self.frame_buckets = np.load(f'{self.v2_folder}/frame_buckets.npy', allow_pickle=True)
            accumulate_frames = self.frame_buckets[-1]

        # self.tk1k_paths = sorted([
        #     p.relative_to(tk1k_folder) for ext in exts
        #     for p in Path(f'{tk1k_folder}').glob(f'*.{ext}')
        # ])[:num_people]
        self.tk1k_paths = list(self.tk1k_paths)
        self.hdtf_paths = sorted(os.listdir(hdtf_folder))[:num_people]
        self.paths = self.tk1k_paths + self.hdtf_paths
        self.indicator = [0] * len(self.tk1k_paths) + [1] * len(self.hdtf_paths)
        
        self.length = len(self.paths)
        if split is None:
            self.offset = 0
        elif split == 'train':
            # last 60k
            self.length = self.length - 10
            self.offset = 10
        elif split == 'val':
            # first 10k
            self.length = 10
            self.offset = 0
        else:
            raise NotImplementedError()

        transform = [
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
        ]
        if do_augment:
            transform.append(transforms.RandomHorizontalFlip())
        if do_transform:
            transform.append(transforms.ToTensor())
        if do_normalize:
            transform.append(
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))
        self.transform = transforms.Compose(transform)

        def random_hdtf_frame(hdtf_path):
            files = os.listdir(hdtf_path)
            random_frame_number = random.randint(0, len(files) - 1)
            img = Image.open(os.path.join(hdtf_path, files[random_frame_number]))
            img = img.convert('RGB')
            return img

        def random_video_frame(video_path):
            video = cv2.VideoCapture(video_path)
            total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
            random_frame_number = random.randint(0, total_frames - 1)
            video.set(cv2.CAP_PROP_POS_FRAMES, random_frame_number)
            ret, frame = video.read()
            video.release()
            pil_frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            return pil_frame
        
        self.random_hdtf_frame = random_hdtf_frame
        self.random_video_frame = random_video_frame
    
    def __len__(self):
        return self.length

    def __getitem__(self, index):
        index = index + self.offset
        if self.indicator[index] == 0:
            try:
                # path = os.path.join(self.tk1k_folder, self.paths[index])
                img = self.random_video_frame(self.paths[index])
            except:
                # remove the corrupted video
                self.paths.pop(index)
                self.indicator.pop(index)
                if index >= len(self.paths):
                    index = len(self.paths) - 1
                return self.__getitem__(index)
        else:
            path = os.path.join(self.hdtf_folder, self.paths[index])
            img = self.random_hdtf_frame(path)

        
        img = self.transform(img)  
        return {'image': img, 'index': index}