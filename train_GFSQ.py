#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#!/usr/bin/env python
# coding: utf-8


# In[ ]:


import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from modules.wavenet import WaveNet
from modules.dvae import GFSQ, DVAEDecoder
from modules.spectrogram import LogMelSpectrogram
import os
from torch.utils.tensorboard import SummaryWriter
import librosa
import torchaudio
from torch.utils.data import random_split
import logging


# 设置设备，优先使用CUDA，其次是MPS（Mac上的GPU加速），最后是CPU

# In[ ]:


device = torch.device('cuda:0' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
# 设置日志级别为INFO
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()
logger.info(f"Use device: {device}")
log_dir = "runs/experiment1"  # 指定日志目录
writer = SummaryWriter(log_dir=log_dir)


# In[ ]:


class AudioDataset(Dataset):
    def __init__(self, audio_files, sample_rate=44100):
        # 初始化音频文件列表和Mel谱图转换器
        self.audio_files = audio_files
        # self.mel_spec = LogMelSpectrogram(
        #     sample_rate=44100,
        #     n_fft=2048,
        #     win_length=2048,
        #     hop_length=512,
        #     n_mels=128,
        #     f_min=0.0,
        #     f_max=8000.0,
        # )
        self.mel_spectrogram = torchaudio.transforms.MelSpectrogram(sample_rate=sample_rate)
        self.sample_rate = sample_rate
    def __len__(self):
        # 返回数据集中的音频文件数量
        return len(self.audio_files)
    def __getitem__(self, idx):
        # 加载并返回指定索引的音频文件的Mel谱图
        mel_spectrogram = self.load_mel_spectrogram(self.audio_files[idx])
        return mel_spectrogram
    def load_mel_spectrogram(self, file_path):
        # 加载音频文件并转换为Mel谱图
        waveform, _ = librosa.load(file_path, sr=44100, mono=True)
        audio = torch.from_numpy(waveform)
        mel_spectrogram = self.mel_spectrogram(audio)
        return mel_spectrogram.squeeze(0)


# In[ ]:


def get_audio_files(root_dir):
    """# 从指定目录加载所有符合条件的音频文件
    """
    audio_files = []
    for root, _, files in os.walk(root_dir):
        for file in files:
            if file.endswith(".wav"):
                file_path = os.path.join(root, file)
                duration = torchaudio.info(file_path).num_frames / torchaudio.info(file_path).sample_rate
                if 2 <= duration <= 30:
                    audio_files.append(file_path)
    return audio_files


# In[ ]:


def dynamic_collate_fn(batch):
    # 按照音频长度排序
    batch.sort(key=lambda x: x.shape[1], reverse=True)
    max_len = batch[0].shape[1]
    
    # 填充所有张量到相同的长度
    padded_batch = []
    for tensor in batch:
        padded_tensor = torch.nn.functional.pad(tensor, (0, max_len - tensor.shape[1]), mode='constant', value=0)
        padded_batch.append(padded_tensor)
    
    if len(padded_batch) == 0:
        raise ValueError("All tensors in the batch were skipped. Check your data preprocessing.")
    
    batch_tensor = torch.stack(padded_batch)
    return batch_tensor


# 初始化模型参数

# In[ ]:


model_params = {
    "WaveNet": {"input_channels": 128, "output_channels": 1024, 'residual_layers': 20, 'dilation_cycle': 4},
    "GFSQ": {"dim": 1024, "levels": [8, 5, 5, 5], "G": 2, "R": 1},
    "DVAEDecoder": {"idim": 1024, "odim": 128}
}


# 实例化模型

# In[ ]:


wavenet = WaveNet(**model_params["WaveNet"]).to(device)
gfsq = GFSQ(**model_params["GFSQ"]).to(device)
decoder = DVAEDecoder(**model_params["DVAEDecoder"]).to(device)


# 定义损失函数和优化器

# In[ ]:


criterion = nn.MSELoss()
optimizer = optim.Adam(list(wavenet.parameters()) + list(gfsq.parameters()) + list(decoder.parameters()), lr=1e-5)  # 调整学习率


# 使用学习率调度器

# In[ ]:


scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)  # 调整调度器参数


# 梯度累积设置

# In[ ]:


accumulation_steps = 8


# 加载数据集并拆分为训练集和验证集

# In[ ]:


root_dir = "/tmp/three_moon/"
audio_files = get_audio_files(root_dir)
dataset = AudioDataset(audio_files)


# 切割分成训练集和校验集

# In[ ]:


train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])


# In[ ]:


train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, collate_fn=dynamic_collate_fn, )
val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False, collate_fn=dynamic_collate_fn, )


# In[ ]:


logger.info(f"Train size: {len(train_dataset)} \t Val size: {len(val_dataset)}")


# 查找是否有记录点

# In[ ]:


import glob  # 用于查找模型文件


# 定义 resume 变量

# In[ ]:


resume = True  # 如果需要从最新检查点恢复训练，则设置为 True


# 获取最新的检查点

# In[ ]:


if resume:
    checkpoint_files = glob.glob('checkpoint_epoch_*.pth')
    if checkpoint_files:
        latest_checkpoint = max(checkpoint_files, key=os.path.getctime)
        checkpoint = torch.load(latest_checkpoint)
        wavenet.load_state_dict(checkpoint['wavenet_state_dict'])
        gfsq.load_state_dict(checkpoint['gfsq_state_dict'])
        decoder.load_state_dict(checkpoint['decoder_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
        logger.info(f"Resumed training from epoch {start_epoch}")
    else:
        start_epoch = 0
        logger.info("No checkpoint found, starting from scratch.")
else:
    start_epoch = 0


# 训练循环

# In[ ]:


# 训练循环
num_epochs = 10000  # 定义训练的总轮数
for epoch in range(start_epoch, num_epochs):
    wavenet.train()  # 设置WaveNet模型为训练模式
    gfsq.train()  # 设置GFSQ模型为训练模式
    decoder.train()  # 设置DVAEDecoder模型为训练模式
    
    for i, mel_spectrogram in enumerate(train_loader):
        mel_spectrogram = mel_spectrogram.to(device)  # 将mel谱图数据移动到指定设备（GPU或CPU）
        optimizer.zero_grad()  # 清空梯度
        
        # 前向传播
        features = wavenet(mel_spectrogram)  # 通过WaveNet提取特征
        _, quantized_features, _, _, quantized_indices = gfsq(features)  # 通过GFSQ量化特征
        quantized_features = quantized_features.transpose(1, 2)  # 转置量化特征以适应解码器输入
        decoded_features = decoder(quantized_features)  # 通过DVAEDecoder解码特征
        decoded_features = decoded_features.transpose(1, 2)  # 转置解码后的特征以匹配原始mel谱图
        
        # 计算损失
        loss = criterion(decoded_features, mel_spectrogram)  # 计算解码后的特征与原始mel谱图之间的均方误差损失
        (loss / accumulation_steps).backward()  # 反向传播并进行梯度累积
        
        if (i + 1) % accumulation_steps == 0:
            optimizer.step()  # 每 accumulation_steps 步更新一次模型参数

        # 打印每5 steps的信息
        if (i + 1) % 5 == 0:
            logger.info(f"Epoch [{epoch+1}/{num_epochs}], Step [{i+1}], Loss: {loss.item()}")
            writer.add_scalar('training_loss', loss.item(), epoch * len(train_loader) + i)  # 记录训练损失到TensorBoard

        # 每1000 steps保存一次模型
        if (i + 1) % 1000 == 0:
            checkpoint_path = f'checkpoint_epoch_{epoch+1}_step_{i+1}.pth'
            torch.save({
                'epoch': epoch,
                'wavenet_state_dict': wavenet.state_dict(),
                'gfsq_state_dict': gfsq.state_dict(),
                'decoder_state_dict': decoder.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, checkpoint_path)
            logger.info(f"Model saved to {checkpoint_path}")

    scheduler.step()  # 每个epoch结束后更新学习率

    # 记录当前学习率到TensorBoard
    current_lr = scheduler.get_last_lr()[0]
    writer.add_scalar('learning_rate', current_lr, epoch)

    # 验证模型
    wavenet.eval()  # 设置WaveNet模型为评估模式
    gfsq.eval()  # 设置GFSQ模型为评估模式
    decoder.eval()  # 设置DVAEDecoder模型为评估模式
    val_loss = 0  # 初始化验证损失
    with torch.no_grad():  # 禁用梯度计算
        for mel_spectrogram in val_loader:
            mel_spectrogram = mel_spectrogram.to(device)  # 将mel谱图数据移动到指定设备
            features = wavenet(mel_spectrogram)  # 通过WaveNet提取特征
            _, quantized_features, _, _, quantized_indices = gfsq(features)  # 通过GFSQ量化特征
            quantized_features = quantized_features.transpose(1, 2)  # 转置量化特征以适应解码器输入
            decoded_features = decoder(quantized_features)  # 通过DVAEDecoder解码特征
            decoded_features = decoded_features.transpose(1, 2)  # 转置解码后的特征以匹配原始mel谱图
            loss = criterion(decoded_features, mel_spectrogram)  # 计算解码后的特征与原始mel谱图之间的均方误差损失
            val_loss += loss.item()  # 累加验证损失

    val_loss /= len(val_loader)  # 计算平均验证损失
    logger.info(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item()}, Val Loss: {val_loss}")
    writer.add_scalar('validation_loss', val_loss, epoch)  # 记录验证损失到TensorBoard

logger.info("训练完成")  # 训练完成后打印日志
writer.close()  # 关闭TensorBoard日志记录器


# In[ ]:


logger.info("训练完成")  # 训练完成后打印日志

