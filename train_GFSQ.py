#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F 
from torch.utils.data import DataLoader, TensorDataset,Dataset
from modules.wavenet import WaveNet
from modules.dvae import GFSQ,DVAEDecoder
from modules.spectrogram import LogMelSpectrogram
import os
import torchaudio
from torch.utils.data import random_split
import librosa
import logging

# 设置设备
device = torch.device('cuda:0' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
# 设置日志
logger = logging.getLogger()


# In[2]:


class AudioDataset(Dataset):
    def __init__(self, audio_files, sample_rate=44100):
        self.audio_files = audio_files
        self.mel_spec = LogMelSpectrogram(
            sample_rate=44100,
            n_fft=2048,
            win_length=2048,
            hop_length=512,
            n_mels=128,
            f_min=0.0,
            f_max=8000.0,
        )
        self.sample_rate = sample_rate

    def __len__(self):
        return len(self.audio_files)

    def __getitem__(self, idx):
        mel_spectrogram = self.load_mel_spectrogram(self.audio_files[idx])
        return mel_spectrogram

    def load_mel_spectrogram(self, file_path):
        waveform, sample_rate = torchaudio.load(file_path)
        if sample_rate != self.sample_rate:
            waveform = torchaudio.transforms.Resample(sample_rate, self.sample_rate)(waveform)
        mel_spectrogram = torchaudio.transforms.MelSpectrogram(sample_rate=self.sample_rate)(waveform)
        with torch.no_grad():
            return mel_spectrogram.squeeze(0)

    # def load_mel_spectrogram(self, file_path):
    #     audio, _ = librosa.load(file_path, sr=self.sample_rate, mono=True)
    #     audio = torch.from_numpy(audio)
    #     audio = audio.float()
    #     audio = audio[:, None, :]
    #     # 生成mel频谱图
    #     with torch.no_grad():
    #         return self.mel_spec(audio)


# In[3]:


# 初始化模型参数
model_params = {
    "WaveNet": {"input_channels": 128, "output_channels": 1024, 'residual_layers':20,'dilation_cycle':4},
    "GFSQ": {"dim": 1024, "levels": [8,5,5,5], "G": 2, "R": 1},
    "DVAEDecoder": {"idim": 1024, "odim": 128}
}


# In[4]:


# 实例化模型
wavenet = WaveNet(**model_params["WaveNet"]).to(device)
gfsq = GFSQ(**model_params["GFSQ"]).to(device)
decoder = DVAEDecoder(**model_params["DVAEDecoder"]).to(device)

# 定义损失函数和优化器
criterion = nn.MSELoss()
optimizer = optim.Adam(gfsq.parameters(), lr=1e-4)


# In[5]:


def get_audio_files(root_dir):
    audio_files = []
    for root, _, files in os.walk(root_dir):
        for file in files:
            if file.endswith(".wav"):
                file_path = os.path.join(root, file)
                duration = torchaudio.info(file_path).num_frames / torchaudio.info(file_path).sample_rate
                if 2 <= duration <= 30:
                    audio_files.append(file_path)
    return audio_files

def collate_fn(batch):
    max_len = max(tensor.shape[1] for tensor in batch)
    padded_batch = []
    for tensor in batch:
        if tensor.shape[1] == max_len:
            padded_batch.append(tensor)
        else:
            # 跳过无法对齐的数据
            logger.warning(f"Skipping tensor with shape {tensor.shape}")
    if len(padded_batch) == 0:
        raise ValueError("All tensors in the batch were skipped. Check your data preprocessing.")
    batch_tensor = torch.stack(padded_batch)
    return batch_tensor



# 加载数据集并拆分为训练集和验证集
root_dir = "/tmp/three_moon/"
audio_files = get_audio_files(root_dir)
dataset = AudioDataset(audio_files)

# 计算训练集和验证集的大小
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size

# 使用 random_split 进行数据集拆分
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

# 创建数据加载器
train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False, collate_fn=collate_fn)

# 打印训练数量和验证数量
logger.info(f"Train size: {len(train_dataset)} \t Val size: {len(val_dataset)}")


# In[6]:


# 训练循环
num_epochs = 100
for epoch in range(num_epochs):
    # 训练阶段
    wavenet.train()
    gfsq.train()
    decoder.train()
    for mel_spectrogram in train_loader:
        # 确保输入数据的形状为 [batch_size, channels, time_steps]
        mel_spectrogram = mel_spectrogram.to(device)

        # 提取特征
        features = wavenet(mel_spectrogram)
        
        # 量化特征
        _, quantized_features, _, _, quantized_indices = gfsq(features)
        
        # 解码量化特征
        quantized_features = quantized_features.transpose(1, 2)
        decoded_features = decoder(quantized_features)
        decoded_features = decoded_features.transpose(1, 2)
        
        # 计算损失
        loss = criterion(decoded_features, mel_spectrogram)
        
        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    # 验证阶段
    wavenet.eval()
    gfsq.eval()
    decoder.eval()
    val_loss = 0
    with torch.no_grad():
        for mel_spectrogram in val_loader:
            mel_spectrogram = mel_spectrogram.to(device)
            
            # 提取特征
            features = wavenet(mel_spectrogram)
            
            # 量化特征
            _, quantized_features, _, _, quantized_indices = gfsq(features)
            
            # 解码量化特征
            quantized_features = quantized_features.transpose(1, 2)
            decoded_features = decoder(quantized_features)
            decoded_features = decoded_features.transpose(1, 2)
            
            # 计算损失
            loss = criterion(decoded_features, mel_spectrogram)
            val_loss += loss.item()
    
    val_loss /= len(val_loader)
    logger.info(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item()}, Val Loss: {val_loss}")

logger.info("训练完成")

