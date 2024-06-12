#!/usr/bin/env python
# coding: utf-8

# In[1]:


#!/usr/bin/env python
# coding: utf-8


# In[2]:


import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from modules.wavenet import WaveNet
from modules.dvae import GFSQ, DVAEDecoder
import os
from torch.utils.tensorboard import SummaryWriter
import librosa
import torchaudio
from torch.utils.data import random_split
import logging
from torch.cuda.amp import autocast, GradScaler
from modules.feature_extractors import MelSpectrogramFeatures


# 设置设备，优先使用CUDA，其次是MPS（Mac上的GPU加速），最后是CPU

# In[3]:


device = torch.device('cuda:0' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
# 设置日志级别为INFO
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()
logger.info(f"Use device: {device}")
log_dir = "runs/experiment1"  # 指定日志目录
writer = SummaryWriter(log_dir=log_dir)


# In[4]:


class AudioDataset(Dataset):
    def __init__(self, audio_files, sample_rate=24000,n_fft =1024,hop_length=512,n_mels=100):
        # 初始化音频文件列表和Mel谱图转换器
        self.audio_files = audio_files
        # self.mel_spectrogram = torchaudio.transforms.MelSpectrogram(sample_rate=sample_rate)
        self.mel_spectrogram = MelSpectrogramFeatures(sample_rate=sample_rate, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)
        self.sample_rate = sample_rate
    def __len__(self):
        # 返回数据集中的音频文件数量
        return len(self.audio_files)
    def __getitem__(self, idx):
        # 加载并返回指定索引的音频文件的Mel谱图
        mel_spectrogram = self.load_mel_spectrogram(self.audio_files[idx])
        return mel_spectrogram
    # def load_mel_spectrogram(self, file_path):
    #     # 加载音频文件并转换为Mel谱图
    #     waveform, sr = librosa.load(file_path, sr=self.sample_rate, mono=True)
    #     S = librosa.feature.melspectrogram(y=waveform, sr=sr, n_mels=128,n_fft=1024,hop_length=256,)
    #     return torch.from_numpy(S)
    def load_mel_spectrogram(self, file_path):
        # 加载音频文件并转换为Mel谱图
        waveform, sr = torchaudio.load(file_path)
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0)
        if sr != self.sample_rate:
            waveform = torchaudio.transforms.Resample(sr, self.sample_rate)(waveform)
        mel_spectrogram = self.mel_spectrogram(waveform)
        return mel_spectrogram[0]


# In[5]:


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


# In[6]:


def dynamic_collate_fn(batch):
    # Filter out tensors that do not have 2 dimensions
    batch = [tensor for tensor in batch if len(tensor.shape) == 2]

    # Ensure the batch is not empty after filtering
    if len(batch) == 0:
        raise ValueError("All tensors in the batch were skipped. Check your data preprocessing.")
    
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

# In[7]:


model_params = {
    "WaveNet": {"input_channels": 100, "output_channels": 512, 'residual_layers': 12, 'dilation_cycle': 4,},
    "GFSQ": {"dim": 512, "levels": [8,8,5,5], "G": 4, "R": 4},
    "DVAEDecoder": {"idim": 256, "odim": 100, "n_layer":12, "bn_dim": 128, "hidden":512}
}


# 实例化模型

# In[8]:


wavenet = WaveNet(**model_params["WaveNet"]).to(device)
gfsq = GFSQ(**model_params["GFSQ"]).to(device)
decoder = DVAEDecoder(**model_params["DVAEDecoder"]).to(device)


# 定义损失函数和优化器

# In[9]:


loss_type = 'MSE'
if loss_type == 'MSE':
    criterion = nn.MSELoss()
else:
    criterion = nn.L1Loss()

optimizer = optim.Adam(
    list(wavenet.parameters()) + list(gfsq.parameters()) + list(decoder.parameters()), 
    lr=4e-4,
    betas=(0.8, 0.99),
    eps=1e-6,
)


# 使用学习率调度器

# In[10]:


import math

scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.999999)  # 调整调度器参数
# 在模型定义后添加以下代码
T_max = 100  # 余弦退火的最大周期为总轮数
eta_min = 1e-6  # 最小学习率为1e-6

# scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_max, eta_min=eta_min)
# scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)

# def get_cosine_schedule_with_warmup_lr_lambda(
#     current_step: int,
#     num_warmup_steps: int,
#     num_training_steps: int,
#     num_cycles: float = 0.5,
#     final_lr_ratio: float = 0.0,
# ):
#     if current_step < num_warmup_steps:
#         return float(current_step) / float(max(1, num_warmup_steps))

#     progress = float(current_step - num_warmup_steps) / float(
#         max(1, num_training_steps - num_warmup_steps)
#     )

#     return max(
#         final_lr_ratio,
#         0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)),
#     )

# # 创建 LambdaLR 调度器
# num_warmup_steps = 100
# num_training_steps = 10000
# final_lr_ratio = 0
# scheduler = torch.optim.lr_scheduler.LambdaLR(
#     optimizer, 
#     lr_lambda=lambda step: get_cosine_schedule_with_warmup_lr_lambda(
#         step, 
#         num_warmup_steps=num_warmup_steps, 
#         num_training_steps=num_training_steps,
#         final_lr_ratio = 0
#     )
# )


# 梯度累积设置

# In[11]:


accumulation_steps = 8


# 加载数据集并拆分为训练集和验证集

# In[12]:


root_dir = "/tmp/three_moon/"
audio_files = get_audio_files(root_dir)
dataset = AudioDataset(audio_files)


# 切割分成训练集和校验集

# In[13]:


train_size = int(0.95 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
logger.info(f"Train size: {len(train_dataset)} \t Val size: {len(val_dataset)}")


# In[14]:


if 'cuda' in str(device):
    batch_size = 8
else:
    batch_size = 1

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=dynamic_collate_fn, )
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=dynamic_collate_fn, )


# 查找是否有记录点

# In[15]:


import glob  # 用于查找模型文件


# 定义 resume 变量

# In[16]:


resume = False  # 如果需要从最新检查点恢复训练，则设置为 True


# 获取最新的检查点

# In[17]:


def convert_state_dict_to_float(state_dict):
    """
    将 state_dict 中的所有张量从 fp16 转换为 fp32
    """
    new_state_dict = {}
    for k, v in state_dict.items():
        new_state_dict[k] = v.float()  # 将每个张量转换为float32
    return new_state_dict


if resume:
    checkpoint_files = glob.glob('checkpoint_epoch_*.pth')
    if checkpoint_files:
        latest_checkpoint = max(checkpoint_files, key=os.path.getctime)
        checkpoint = torch.load(latest_checkpoint)
        wavenet.load_state_dict(convert_state_dict_to_float(checkpoint['wavenet_state_dict']))
        gfsq.load_state_dict(convert_state_dict_to_float(checkpoint['gfsq_state_dict']))
        decoder.load_state_dict(convert_state_dict_to_float(checkpoint['decoder_state_dict']))
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
        logger.info(f"Resumed training from epoch {start_epoch}")
    else:
        start_epoch = 0
        logger.info("No checkpoint found, starting from scratch.")
else:
    start_epoch = 0
    

# 创建 GradScaler，转换为fp16
scaler = GradScaler()


# In[18]:


import librosa
import numpy as np
import torch

def mel_to_audio(mel_spectrogram, sr=24000, n_fft=1024, hop_length=256, win_length=None):
    """将 Mel 频谱图转换回音频信号"""
    # 确保输入为 NumPy 数组
    if isinstance(mel_spectrogram, torch.Tensor):
        mel_spectrogram = mel_spectrogram.cpu().numpy()
    
    # 使用 librosa 的功能进行逆 Mel 频谱变换
    mel_decompress = librosa.feature.inverse.mel_to_audio(mel_spectrogram, sr=sr, n_fft=n_fft, hop_length=hop_length, win_length=win_length)
    return mel_decompress


# In[19]:


# 定义混合损失函数
def mixed_loss(decoded_features, mel_spectrogram):
    loss_mse = F.mse_loss(decoded_features, mel_spectrogram)
    loss_l1 = F.l1_loss(decoded_features, mel_spectrogram)
    return loss_mse*0.5 + 0.5 * loss_l1

# 时间步翻倍
def double_time_steps(mel_spectrogram):
    batch_size, n_mels, time_steps = mel_spectrogram.shape
    mel_spectrogram = mel_spectrogram.unsqueeze(1)  # 添加通道维度
    doubled_mel = F.interpolate(mel_spectrogram, size=(n_mels, time_steps * 2), mode='bilinear', align_corners=False)
    return doubled_mel.squeeze(1)  # 移除通道维度


# 训练循环

# In[ ]:


# 训练循环
num_epochs = 10000  # 定义训练的总轮数
i = 0
for epoch in range(start_epoch, num_epochs):
    wavenet.train()  # 设置WaveNet模型为训练模式
    gfsq.train()  # 设置GFSQ模型为训练模式
    decoder.train()  # 设置DVAEDecoder模型为训练模式
    
    for _, mel_spectrogram in enumerate(train_loader):
        mel_spectrogram = mel_spectrogram.to(device)  # 将mel谱图数据移动到指定设备（GPU或CPU）
        optimizer.zero_grad()  # 清空梯度
        
        # 前向传播
        with autocast():
            features = wavenet(mel_spectrogram)  # 通过WaveNet提取特征
            _, quantized_features, perplexity, _, quantized_indices = gfsq(features)# 通过GFSQ量化特征

            # 将特征沿着 dim=1 维度分成两部分，得到两个形状为 (1, 512, 121) 的张量。
            temp = torch.chunk(quantized_features, 2, dim=1) # flatten trick :)
            #将这两个张量堆叠在一起，得到形状为 (1, 512, 121, 2) 的张量
            temp = torch.stack(temp, -1)
            # 重新调整特征形状，得到 vq_feats 形状为 (1, 512, 242)
            quantized_features = temp.reshape(*temp.shape[:2], -1)
            # 将特征 vq_feats 转置,转置后的特征 vq_feats 形状为 (1, 242, 512)
            # quantized_features = vq_feats.transpose(1, 2)
            
            decoded_features = decoder(quantized_features)  # 通过DVAEDecoder解码特征
            # 解码的时间步会翻倍，所以也需要将原来的Mel频谱图翻倍处理
            mel_spectrogram = double_time_steps(mel_spectrogram)

            # 计算损失
            loss = criterion(decoded_features, mel_spectrogram)  # 计算解码后的特征与原始mel谱图之间的均方误差损失
            # loss = mixed_loss(decoded_features, mel_spectrogram)
            # (loss / accumulation_steps).backward()  # 反向传播并进行梯度累积
        scaler.scale(loss).backward()
        
        if (i + 1) % accumulation_steps == 0:
            # optimizer.step()  # 每 accumulation_steps 步更新一次模型参数
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        # 打印每100 steps的信息
        if (i + 1) % 100 == 0:
            logger.info(f"Epoch [{epoch+1}/{num_epochs}], Step [{i+1}], Loss: {loss.item()}, Perplexity: {perplexity.mean().item()}")
            writer.add_scalar('training_perplexity', perplexity.mean().item(), epoch * len(train_loader) + i)

        # 每500 steps保存一次模型
        if (i + 1) % 1000 == 0 or (i+1) == len(train_loader):
            checkpoint_path = f'checkpoint_epoch_{epoch+1}_step_{i+1}.pth'
            torch.save({
                'epoch': epoch,
                'wavenet_state_dict': wavenet.state_dict(),
                'gfsq_state_dict': gfsq.state_dict(), 
                'decoder_state_dict': decoder.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scaler_state_dict': scaler.state_dict(),  # 保存 GradScaler 状态
            }, checkpoint_path)
            logger.info(f"Model saved to {checkpoint_path}")
        i += 1   # 更新迭代计数器

    scheduler.step()  # 每个epoch结束后更新学习率

    # 验证模型
    wavenet.eval()  # 设置WaveNet模型为评估模式
    gfsq.eval()  # 设置GFSQ模型为评估模式
    decoder.eval()  # 设置DVAEDecoder模型为评估模式
    val_loss_mse = 0  # 初始化验证MSE损失
    val_loss_l1 = 0  # 初始化验证L1损失
    with torch.no_grad():  # 禁用梯度计算
        for batch_index, mel_spectrogram in enumerate(val_loader):
            mel_spectrogram = mel_spectrogram.to(device)  # 将mel谱图数据移动到指定设备
            with autocast():
                features = wavenet(mel_spectrogram)  # 通过WaveNet提取特征
                _, quantized_features, perplexity, _, quantized_indices = gfsq(features) # 通过GFSQ量化特征
    
                # 将特征沿着 dim=1 维度分成两部分，得到两个形状为 (1, 512, 121) 的张量。
                temp = torch.chunk(quantized_features, 2, dim=1) # flatten trick :)
                #将这两个张量堆叠在一起，得到形状为 (1, 512, 121, 2) 的张量
                temp = torch.stack(temp, -1)
                # 重新调整特征形状，得到 vq_feats 形状为 (1, 512, 242)
                quantized_features = temp.reshape(*temp.shape[:2], -1)
                # 将特征 vq_feats 转置,转置后的特征 vq_feats 形状为 (1, 242, 512)
                # quantized_features = vq_feats.transpose(1, 2)
                
                decoded_features = decoder(quantized_features)  # 通过DVAEDecoder解码特征
                # 解码的时间步会翻倍，所以也需要将原来的Mel频谱图翻倍处理
                mel_spectrogram = double_time_steps(mel_spectrogram)
                
                # 计算MSE损失
                loss_mse = F.mse_loss(decoded_features, mel_spectrogram)  # 计算解码后的特征与原始mel谱图之间的均方误差损失
                val_loss_mse += loss_mse.item()  # 累加验证MSE损失

                # 计算L1损失
                loss_l1 = F.l1_loss(decoded_features, mel_spectrogram)  # 计算解码后的特征与原始mel谱图之间的L1损失
                val_loss_l1 += loss_l1.item()  # 累加验证L1损失

    val_loss_mse /= len(val_loader)  # 计算平均验证MSE损失
    val_loss_l1 /= len(val_loader)  # 计算平均验证L1损失

    logger.info(f"Epoch [{epoch+1}/{num_epochs}], MSE Loss: {val_loss_mse}, L1 Loss: {val_loss_l1}")
    writer.add_scalar('validation_mse_loss', val_loss_mse, epoch)  # 记录验证MSE损失到TensorBoard
    writer.add_scalar('validation_l1_loss', val_loss_l1, epoch)  # 记录验证L1损失到TensorBoard

logger.info("训练完成")  # 训练完成后打印日志
writer.close()  # 关闭TensorBoard日志记录器

