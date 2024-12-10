import numpy as np
import soundfile as sf

# 设置参数
sample_rate = 16000  # 采样率
duration = 20  # 持续时间（秒）

# # 生成白噪声
# samples = np.random.normal(0, 1, int(sample_rate * duration))

# # 写入wav文件
# sf.write('preprocess/empty_audio/white_noise.wav', samples, sample_rate)


samples = np.zeros(int(sample_rate * duration))

# 写入wav文件
sf.write('preprocess/empty_audio/empty_audio_15.wav', samples, sample_rate)