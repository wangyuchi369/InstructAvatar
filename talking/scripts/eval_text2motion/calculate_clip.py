import cv2
from PIL import Image
import torch
from transformers import CLIPProcessor, CLIPModel
import json
import os
from tqdm import tqdm

# 检查是否有可用的CUDA设备
device = "cuda" if torch.cuda.is_available() else "cpu"

# 加载模型和处理器，并将模型移动到GPU上
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

input_dir = "/mnt/blob/xxxx/TETF/text2motion/unified_rand/"
mapping_dict = json.load(open('t2m_test_data.json', 'r'))

def get_video_scores(video_path, prompt):
 import cv2
from PIL import Image
import torch
from transformers import CLIPProcessor, CLIPModel
import json
import os
from tqdm import tqdm

# 检查是否有可用的CUDA设备
device = "cuda" if torch.cuda.is_available() else "cpu"

# 加载模型和处理器，并将模型移动到GPU上
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

input_dir = "/mnt/blob/xxxx/TETF/text2motion/unified_rand/"
mapping_dict = json.load(open('t2m_test_data.json', 'r'))

def get_video_scores(video_path, prompt):
    video = cv2.VideoCapture(video_path)
    # 初始化最大CLIP分数
    max_clip_score = 0
    texts = [prompt]
    while True:
        # 读取一帧
        ret, frame = video.read()

        # 如果读取成功，计算CLIP分数
        if ret:
            # 将帧转换为PIL图像
            image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            # 使用处理器预处理图片和文本，并将数据移动到GPU上
            inputs = processor(text=texts, images=[image], return_tensors="pt", padding=True).to(device)

            # 计算CLIP分数
            logits_per_image = model(**inputs).logits_per_image
            clip_score = logits_per_image.item()

            # 更新最大CLIP分数
            max_clip_score = max(max_clip_score, clip_score)
        else:
            break

    # 释放视频文件
    video.release()

    return max_clip_score

scores = []
for each_video in tqdm(os.listdir(input_dir)):
    # 加载图片
    video_path = os.path.join(input_dir, each_video)
    

    # 准备文本
    texts = mapping_dict[each_video[:-4]]
    
    score = get_video_scores(video_path, texts)
    print(score)
    scores.append(score)

print(f"Average text similarity: {sum(scores)/len(scores)}")




# gt: 0.23136341722705697

# ours : 0.23042625716969936

# 22.238

