#!/usr/bin/env python
# coding: utf-8

# In[1]:


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
from modules.discriminator import DynamicAudioDiscriminator


# 设置设备，优先使用CUDA，其次是MPS（Mac上的GPU加速），最后是CPU

# In[2]:


device = torch.device('cuda:0' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
# 设置日志级别为INFO
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()
logger.info(f"Use device: {device}")
log_dir = "runs/experiment1"  # 指定日志目录
writer = SummaryWriter(log_dir=log_dir)


# In[3]:


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


# In[4]:


def get_audio_files(root_dir):
    """# 从指定目录加载所有符合条件的音频文件
    """
    audio_files = []
    for root, _, files in os.walk(root_dir):
        for file in files:
            if file.endswith(".wav"):
                file_path = os.path.join(root, file)
                duration = torchaudio.info(file_path).num_frames / torchaudio.info(file_path).sample_rate
                if 1 <= duration <= 30:
                    audio_files.append(file_path)
    return audio_files


# In[5]:


def dynamic_collate_fn(batch):
    # Filter out tensors that do not have 2 dimensions
    batch = [tensor for tensor in batch if len(tensor.shape) == 2]

    # If the batch is empty after filtering, return None to skip this batch
    if len(batch) == 0:
        return None
    
    # 按照音频长度排序
    batch.sort(key=lambda x: x.shape[1], reverse=True)
    max_len = batch[0].shape[1]
    
    # 填充所有张量到相同的长度
    padded_batch = []
    for tensor in batch:
        padded_tensor = torch.nn.functional.pad(tensor, (0, max_len - tensor.shape[1]), mode='constant', value=0)
        padded_batch.append(padded_tensor)
    
    batch_tensor = torch.stack(padded_batch)
    return batch_tensor


# 初始化模型参数

# In[6]:


model_params = {
    "WaveNet": {"input_channels": 100, "output_channels": 1024, 'residual_layers': 12, 'dilation_cycle': 4,},
    "GFSQ": {"dim": 1024, "levels": [16,16,8,8], "G": 8, "R": 4},
    "DVAEDecoder": {"idim": 1024, "odim": 100, "n_layer":12, "bn_dim": 128, "hidden":512},
    "DynamicAudioDiscriminator":{"num_mels":100}
}


# 实例化模型

# In[7]:


wavenet = WaveNet(**model_params["WaveNet"]).to(device)
gfsq = GFSQ(**model_params["GFSQ"]).to(device)
decoder = DVAEDecoder(**model_params["DVAEDecoder"]).to(device)
# 初始化判别器
discriminator = DynamicAudioDiscriminator(**model_params["DynamicAudioDiscriminator"]).to(device)


# 定义损失函数和优化器

# In[8]:


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
# 定义判别器的优化器
optimizer_d = torch.optim.Adam(discriminator.parameters(), lr=0.0001, betas=(0.5, 0.999))


# 使用学习率调度器

# In[9]:


import math

scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.999999)  # 调整调度器参数
# 在模型定义后添加以下代码
T_max = 100  # 余弦退火的最大周期为总轮数
eta_min = 1e-6  # 最小学习率为1e-6

# scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_max, eta_min=eta_min)
# scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)


# 梯度累积设置

# In[10]:


accumulation_steps = 8


# 加载数据集并拆分为训练集和验证集

# In[11]:


root_dir = "/tmp/three_moon/"
audio_files = get_audio_files(root_dir)
dataset = AudioDataset(audio_files)


# 切割分成训练集和校验集

# In[12]:


train_size = int(0.95 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
logger.info(f"Train size: {len(train_dataset)} \t Val size: {len(val_dataset)}")


# In[13]:


if 'cuda' in str(device):
    batch_size = 8
else:
    batch_size = 1

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=dynamic_collate_fn, )
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=dynamic_collate_fn, )

# 创建 GradScaler，转换为fp16
scaler = GradScaler()


# 查找是否有记录点

# In[14]:


import glob  # 用于查找模型文件


# 定义 resume 变量

# In[15]:


resume = False  # 如果需要从最新检查点恢复训练，则设置为 True


# 获取最新的检查点

# In[16]:


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
        discriminator.load_state_dict(convert_state_dict_to_float(checkpoint['discriminator_state_dict']))
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        optimizer_d.load_state_dict(checkpoint['optimizer_d_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        scaler.load_state_dict(checkpoint['scaler_state_dict'])
        start_epoch = checkpoint['epoch']
        logger.info(f"Resumed training from epoch {start_epoch}")
    else:
        start_epoch = 0
        logger.info("No checkpoint found, starting from scratch.")
else:
    start_epoch = 0


# In[17]:


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


# In[18]:


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

# 定义判别器的损失函数
def discriminator_loss_fn(outputs_real, outputs_fake):
    real_loss = F.binary_cross_entropy_with_logits(outputs_real, torch.ones_like(outputs_real))
    fake_loss = F.binary_cross_entropy_with_logits(outputs_fake, torch.zeros_like(outputs_fake))
    return (real_loss + fake_loss) / 2

# 定义生成器的损失函数
def generator_loss_fn(outputs_fake):
    return F.binary_cross_entropy_with_logits(outputs_fake, torch.ones_like(outputs_fake))

# 定义判别器损失函数
criterion_d = nn.BCELoss()


# In[19]:


def evaluate(wavenet, gfsq, decoder, val_loader, device):
    wavenet.eval()
    gfsq.eval()
    decoder.eval()
    val_loss_mse = 0
    val_loss_l1 = 0
    with torch.no_grad():
        for mel_spectrogram in val_loader:
            if mel_spectrogram is None:
                continue  # Skip this batch
            
            mel_spectrogram = mel_spectrogram.to(device)
            features = wavenet(mel_spectrogram)
            _, quantized_features, _, _, _ = gfsq(features)
            decoded_features = decoder(quantized_features)
            loss_mse = F.mse_loss(decoded_features, mel_spectrogram)
            loss_l1 = F.l1_loss(decoded_features, mel_spectrogram)
            val_loss_mse += loss_mse.item()
            val_loss_l1 += loss_l1.item()
    val_loss_mse /= len(val_loader)
    val_loss_l1 /= len(val_loader)
    return val_loss_mse, val_loss_l1


# 训练循环

# In[20]:


# 训练循环
num_epochs = 10000  # 定义训练的总轮数
d_step = 0  # 判别器步骤计数器
g_step = 0  # 生成器步骤计数器

# 训练循环
for epoch in range(start_epoch, num_epochs):
    # 训练判别器
    wavenet.eval()
    gfsq.eval()
    decoder.eval()
    discriminator.train()
    
    for mel_spectrogram in train_loader:
        if mel_spectrogram is None:
            continue  # Skip this batch
        
        mel_spectrogram = mel_spectrogram.to(device)
        
        optimizer_d.zero_grad()
        with autocast():
            real_labels = torch.ones(mel_spectrogram.size(0), device=device)
            fake_labels = torch.zeros(mel_spectrogram.size(0), device=device)
            
            try:
                real_output = discriminator(mel_spectrogram)
            except RuntimeError as e:
                print(f"Error during discriminator forward pass: {e}")
                continue
            real_output = real_output.view(-1)  # 确保 real_output 的形状与 real_labels 一致
            real_loss = criterion_d(real_output, real_labels)
            
            with torch.no_grad():
                features = wavenet(mel_spectrogram)
                _, quantized_features, _, _, _ = gfsq(features)
                decoded_features = decoder(quantized_features)
            fake_output = discriminator(decoded_features.detach())
            fake_output = fake_output.view(-1)  # 确保 fake_output 的形状与 fake_labels 一致
            fake_loss = criterion_d(fake_output, fake_labels)
            
            d_loss = real_loss + fake_loss
        
        scaler.scale(d_loss).backward()
        scaler.step(optimizer_d)
        scaler.update()
        
        # 记录判别器的损失
        writer.add_scalar('Discriminator Loss/real', real_loss.item(), d_step)
        writer.add_scalar('Discriminator Loss/fake', fake_loss.item(), d_step)
        writer.add_scalar('Discriminator Loss/total', d_loss.item(), d_step)
        
        if (d_step + 1) % 100 == 0 or (d_step + 1) == len(train_loader):
            logger.info(f"Epoch [{epoch+1}/{num_epochs}], Discriminator Step [{d_step+1}], Real Loss: {real_loss.item()}, Fake Loss: {fake_loss.item()}, Total Loss: {d_loss.item()}")
        
        d_step += 1
    
    # 训练生成器
    wavenet.train()
    gfsq.train()
    decoder.train()
    discriminator.eval()
    
    for mel_spectrogram in train_loader:
        if mel_spectrogram is None:
            continue  # Skip this batch
        
        mel_spectrogram = mel_spectrogram.to(device)
        
        optimizer.zero_grad()
        with autocast():
            features = wavenet(mel_spectrogram)
            _, quantized_features, _, _, _ = gfsq(features)
            decoded_features = decoder(quantized_features)
            
            g_loss = criterion(decoded_features, mel_spectrogram)
            g_output = discriminator(decoded_features)
            g_output = g_output.view(-1)  # 确保 g_output 的形状与 real_labels 一致
            g_adv_loss = criterion_d(g_output, real_labels)
            
            total_g_loss = g_loss + g_adv_loss
        
        scaler.scale(total_g_loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        # 记录生成器的损失
        writer.add_scalar('Generator Loss/total', total_g_loss.item(), g_step)
        writer.add_scalar('Generator Loss/reconstruction', g_loss.item(), g_step)
        writer.add_scalar('Generator Loss/adversarial', g_adv_loss.item(), g_step)
        
        if (g_step + 1) % 100 == 0 or (g_step + 1) == len(train_loader):
            logger.info(f"Epoch [{epoch+1}/{num_epochs}], Generator Step [{g_step+1}], Total Loss: {total_g_loss.item()}, Reconstruction Loss: {g_loss.item()}, Adversarial Loss: {g_adv_loss.item()}")
        
        g_step += 1
    
    scheduler.step()
    
    # 验证模型
    val_loss_mse, val_loss_l1 = evaluate(wavenet, gfsq, decoder, val_loader, device)
    logger.info(f"Epoch [{epoch+1}/{num_epochs}], MSE Loss: {val_loss_mse}, L1 Loss: {val_loss_l1}")
    writer.add_scalar('validation_mse_loss', val_loss_mse, epoch)
    writer.add_scalar('validation_l1_loss', val_loss_l1, epoch)

logger.info("训练完成")
writer.close()


# # 训练循环
# num_epochs = 10000  # 定义训练的总轮数
# i = 0
# for epoch in range(start_epoch, num_epochs):
#     wavenet.train()  # 设置WaveNet模型为训练模式
#     gfsq.train()  # 设置GFSQ模型为训练模式
#     decoder.train()  # 设置DVAEDecoder模型为训练模式
#     
#     for _, mel_spectrogram in enumerate(train_loader):
#         mel_spectrogram = mel_spectrogram.to(device)  # 将mel谱图数据移动到指定设备（GPU或CPU）
#         optimizer.zero_grad()  # 清空梯度
#         
#         # 前向传播
#         with autocast():
#             features = wavenet(mel_spectrogram)  # 通过WaveNet提取特征
#             _, quantized_features, perplexity, _, quantized_indices = gfsq(features)# 通过GFSQ量化特征
# 
#             # 将特征沿着 dim=1 维度分成两部分，得到两个形状为 (1, 512, 121) 的张量。
#             temp = torch.chunk(quantized_features, 2, dim=1) # flatten trick :)
#             #将这两个张量堆叠在一起，得到形状为 (1, 512, 121, 2) 的张量
#             temp = torch.stack(temp, -1)
#             # 重新调整特征形状，得到 vq_feats 形状为 (1, 512, 242)
#             quantized_features = temp.reshape(*temp.shape[:2], -1)
#             # 将特征 vq_feats 转置,转置后的特征 vq_feats 形状为 (1, 242, 512)
#             # quantized_features = vq_feats.transpose(1, 2)
#             
#             decoded_features = decoder(quantized_features)  # 通过DVAEDecoder解码特征
#             # 解码的时间步会翻倍，所以也需要将原来的Mel频谱图翻倍处理
#             mel_spectrogram = double_time_steps(mel_spectrogram)
# 
#             # 训练判别器
#             discriminator.zero_grad()
#             real_data = mel_spectrogram.unsqueeze(1).to(device)  # 真实数据
#             fake_data = decoder(quantized_features).detach()      # 生成器生成的数据
#             real_output = discriminator(real_data)
#             fake_output = discriminator(fake_data)
#             d_loss = discriminator_loss_fn(real_output, fake_output)
#             d_loss.backward()
#             optimizer_d.step()
#             
#             # 训练生成器
#             optimizer.zero_grad()
#             fake_data = decoder(quantized_features)
#             fake_output = discriminator(fake_data)
#             g_loss = generator_loss_fn(fake_output)
#             g_loss.backward()
#             optimizer.step()
# 
#             # 计算损失
#             loss = criterion(decoded_features, mel_spectrogram)  # 计算解码后的特征与原始mel谱图之间的均方误差损失
#             # loss = mixed_loss(decoded_features, mel_spectrogram)
#             # (loss / accumulation_steps).backward()  # 反向传播并进行梯度累积
#         scaler.scale(loss).backward()
#         
#         if (i + 1) % accumulation_steps == 0:
#             # optimizer.step()  # 每 accumulation_steps 步更新一次模型参数
#             scaler.step(optimizer)
#             scaler.update()
#             optimizer.zero_grad()
# 
#         # 打印每100 steps的信息
#         if (i + 1) % 100 == 0:
#             logger.info(f"Epoch [{epoch+1}/{num_epochs}], Step [{i+1}], Loss: {loss.item()}, Perplexity: {perplexity.mean().item()}")
#             writer.add_scalar('training_perplexity', perplexity.mean().item(), epoch * len(train_loader) + i)
# 
#         # 每500 steps保存一次模型
#         if (i + 1) % 1000 == 0 or (i+1) == len(train_loader):
#             checkpoint_path = f'checkpoint_epoch_{epoch+1}_step_{i+1}.pth'
#             torch.save({
#                 'epoch': epoch,
#                 'wavenet_state_dict': wavenet.state_dict(),
#                 'gfsq_state_dict': gfsq.state_dict(), 
#                 'decoder_state_dict': decoder.state_dict(),
#                 'optimizer_state_dict': optimizer.state_dict(),
#                 'scaler_state_dict': scaler.state_dict(),  # 保存 GradScaler 状态
#             }, checkpoint_path)
#             logger.info(f"Model saved to {checkpoint_path}")
#         i += 1   # 更新迭代计数器
# 
#         # 记录损失值
#         writer.add_scalar('Loss/Discriminator', d_loss.item(), global_step=i)
#         writer.add_scalar('Loss/Generator', g_loss.item(), global_step=batch_idx)
# 
#     scheduler.step()  # 每个epoch结束后更新学习率
# 
#     # 验证模型
#     wavenet.eval()  # 设置WaveNet模型为评估模式
#     gfsq.eval()  # 设置GFSQ模型为评估模式
#     decoder.eval()  # 设置DVAEDecoder模型为评估模式
#     val_loss_mse = 0  # 初始化验证MSE损失
#     val_loss_l1 = 0  # 初始化验证L1损失
#     with torch.no_grad():  # 禁用梯度计算
#         for batch_index, mel_spectrogram in enumerate(val_loader):
#             mel_spectrogram = mel_spectrogram.to(device)  # 将mel谱图数据移动到指定设备
#             with autocast():
#                 features = wavenet(mel_spectrogram)  # 通过WaveNet提取特征
#                 _, quantized_features, perplexity, _, quantized_indices = gfsq(features) # 通过GFSQ量化特征
#     
#                 # 将特征沿着 dim=1 维度分成两部分，得到两个形状为 (1, 512, 121) 的张量。
#                 temp = torch.chunk(quantized_features, 2, dim=1) # flatten trick :)
#                 #将这两个张量堆叠在一起，得到形状为 (1, 512, 121, 2) 的张量
#                 temp = torch.stack(temp, -1)
#                 # 重新调整特征形状，得到 vq_feats 形状为 (1, 512, 242)
#                 quantized_features = temp.reshape(*temp.shape[:2], -1)
#                 # 将特征 vq_feats 转置,转置后的特征 vq_feats 形状为 (1, 242, 512)
#                 # quantized_features = vq_feats.transpose(1, 2)
#                 
#                 decoded_features = decoder(quantized_features)  # 通过DVAEDecoder解码特征
#                 # 解码的时间步会翻倍，所以也需要将原来的Mel频谱图翻倍处理
#                 mel_spectrogram = double_time_steps(mel_spectrogram)
#                 
#                 # 计算MSE损失
#                 loss_mse = F.mse_loss(decoded_features, mel_spectrogram)  # 计算解码后的特征与原始mel谱图之间的均方误差损失
#                 val_loss_mse += loss_mse.item()  # 累加验证MSE损失
# 
#                 # 计算L1损失
#                 loss_l1 = F.l1_loss(decoded_features, mel_spectrogram)  # 计算解码后的特征与原始mel谱图之间的L1损失
#                 val_loss_l1 += loss_l1.item()  # 累加验证L1损失
# 
#     val_loss_mse /= len(val_loader)  # 计算平均验证MSE损失
#     val_loss_l1 /= len(val_loader)  # 计算平均验证L1损失
# 
#     logger.info(f"Epoch [{epoch+1}/{num_epochs}], MSE Loss: {val_loss_mse}, L1 Loss: {val_loss_l1}")
#     writer.add_scalar('validation_mse_loss', val_loss_mse, epoch)  # 记录验证MSE损失到TensorBoard
#     writer.add_scalar('validation_l1_loss', val_loss_l1, epoch)  # 记录验证L1损失到TensorBoard
# 
# logger.info("训练完成")  # 训练完成后打印日志
# writer.close()  # 关闭TensorBoard日志记录器
