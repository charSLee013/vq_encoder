import torch
from torch import nn
import math

from .feature_extractors import MelSpectrogramFeatures
from .fsq import DownsampleFiniteScalarQuantize
from .wavenet import WaveNet
from .spectrogram import LogMelSpectrogram
from .utils import sequence_mask

class VQEncoder(nn.Module):
    def __init__(self, 
                 # 采样率是每秒钟对连续信号进行采样的次数，单位是赫兹 (Hz),
                 # 采样率决定了音频信号的时间分辨率。常见的采样率有 44.1 kHz（CD 音质）和 48 kHz（专业音频）
                 sample_rate=44100, 
                 # 进行快速傅里叶变换 (FFT) 时使用的点数
                 # n_fft 决定了频谱图的频率分辨率。较大的 n_fft 会提供更高的频率分辨率，但计算量也更大
                 n_fft=2048, 
                 # 相邻帧之间的间隔，单位是采样点数
                 # hop_length 决定了频谱图的时间分辨率。较小的 hop_length 会提供更高的时间分辨率，但频谱图的帧数也会增加
                 hop_length=512, 
                 # n_mels 决定了梅尔频谱图的频率分辨率。梅尔频谱图使用对数刻度来模拟人耳的听觉感知
                 n_mels=128, 
                 # 每次 FFT 计算时使用的窗口长度，单位是采样点数,通常，win_length 与 n_fft 相同
                 win_length=2048,
                 # 特征输出维度
                 output_channels = 1024,
                 residual_channels = 768,
                 residual_layers = 20,
                 dilation_cycle =4,
                 ):
        super().__init__()
        # 这是一个深度卷积神经网络，用于提取音频信号的高级特征。
        # WaveNet 是一种非常有效的音频生成模型，它能够捕捉音频信号的时间结构和频谱特征。
        self.encoder = WaveNet(
            input_channels=n_mels,
            output_channels=output_channels,
            residual_channels=residual_channels,
            residual_layers=residual_layers,
            dilation_cycle=dilation_cycle,
        )
        #  这是一个向量量化器，它将 WaveNet 提取的特征映射到离散的码本中。
        # 向量量化是一种将连续信号转换为离散表示的技术，它有助于模型学习到音色的不变性，因为相同的音色应该映射到相同的码字
        # self.quantizer = DownsampleFiniteScalarQuantize(
        #     input_dim=768, n_codebooks=1, n_groups=2, levels=[8, 5, 5, 5]
        # )
        
        # # 这是一个特征提取器，用于将音频信号转换为梅尔谱图。梅尔谱图是一种常用的音频特征表示，它能够捕捉音频的频谱特性。
        # self.spec = LogMelSpectrogram(
        #     sample_rate=sample_rate,
        #     n_fft=n_fft,
        #     win_length=win_length,
        #     hop_length=hop_length,
        #     n_mels=n_mels,
        #     f_min=0.0,
        #     f_max=8000.0,
        # )
        self.spec = MelSpectrogramFeatures(sample_rate=sample_rate, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)
        self.sample_rate = sample_rate
        self.hop_length = hop_length
        self.n_fft = n_fft
        

    @torch.no_grad()
    def forward(self, audios, audio_lengths, sr=None):
        """在前向传播中，VQEncoder 首先将输入音频转换为梅尔谱图，然后使用 WaveNet 提取特征。接着，这些特征被送入向量量化器进行离散化。离散化后的特征表示是音色的固定表示，因为它们是从一个预定义的码本中选择的，这个码本是在训练过程中学习到的，它包含了与音色相关的信息。

        通过这种方式，VQEncoder 能够将音频信号编码成一种固定音色的表示，这种表示对于生成具有特定音色的语音非常有用。在模型的其他部分，如生成器中，这些离散的音色表示可以用来合成具有相同音色的语音。
        """
        # 将输入的音频信号 audios 转换为梅尔频谱图,但是在mps设备上不兼容，需要临时转回cpu上运行
        if audios.device.type == 'mps':
            mel_spec = self.spec(audios.cpu()).to(audios.device.type)
        else:
            mel_spec = self.spec(audios)
        if sr is not None:
            # 确保音频长度与模型的采样率一致，以便后续处理
            audio_lengths = audio_lengths * self.sample_rate // sr
        #  计算梅尔频谱图的长度 mel_lengths，即每个音频对应的梅尔频谱图的时间步数
        mel_lengths = audio_lengths // self.hop_length
        mel_lengths = mel_lengths.to(mel_spec.device.type)
        # 创建一个掩码 mel_masks，用于标识每个梅尔频谱图的有效部分,处理不同长度的音频时，需要掩码来忽略填充部分
        # 形状为 [batch_size, mel_spec_length]
        mel_masks = (torch.arange(mel_spec.shape[2], device=mel_spec.device) < mel_lengths[:, None])
        # 这一步将布尔掩码 mel_masks 转换为浮点数，并增加一个维度，使其形状变为 [batch_size, 1, mel_spec_length]。
        # 这样做的目的是为了方便与梅尔频谱图进行逐元素相乘操作。
        mel_masks_float_conv = mel_masks[:, None, :].float()
        # 将梅尔频谱图 mel_spec 与掩码 mel_masks_float_conv 相乘，得到有效部分的梅尔频谱图 mels
        mels = mel_spec * mel_masks_float_conv
        # 将有效的梅尔频谱图 mels 输入到编码器 self.encoder 中，得到编码特征 encoded_features，并再次应用掩码。
        # 提取音频的高层次特征表示，同时忽略填充部分
        encoded_features = self.encoder(mels) * mel_masks_float_conv
        # 将编码特征 encoded_features 输入到量化器 self.quantizer 中，得到量化后的特征 encoded_features，并再次应用掩码
        # encoded_features = self.quantizer(encoded_features).z * mel_masks_float_conv
        return encoded_features,mels

    @torch.no_grad()
    def encode(self, audios, audio_lengths, sr=None):
        audios = audios.float()

        # 将输入的音频信号 audios 转换为梅尔频谱图 mels,适合用于后续的特征提取和编码
        mels = self.spec(audios, sample_rate=sr)
        # 计算梅尔频谱图的长度 mel_lengths，即每个音频对应的梅尔频谱图的时间步数
        mel_lengths = audio_lengths // self.spec.hop_length
        # 创建一个掩码 mel_masks，用于标识每个梅尔频谱图的有效部分,处理不同长度的音频时，需要掩码来忽略填充部分
        mel_masks = sequence_mask(mel_lengths, mels.shape[2])
        # 将布尔掩码 mel_masks 转换为浮点数，并增加一个维度，使其形状变为 [batch_size, 1, mel_spec_length]
        mel_masks_float_conv = mel_masks[:, None, :].float()
        # 将梅尔频谱图 mels 与掩码 mel_masks_float_conv 相乘，得到有效部分的梅尔频谱图 mels
        mels = mels * mel_masks_float_conv

        # Encode
        # 将有效的梅尔频谱图 mels 输入到编码器 self.encoder 中，得到编码特征 encoded_features，并再次应用掩码
        encoded_features = self.encoder(mels) * mel_masks_float_conv
        # 量化器可能会对特征进行下采样，因此需要计算下采样后的特征长度
        feature_lengths = mel_lengths // math.prod(self.quantizer.downsample_factor)

        # 返回量化后的特征和特征长度
        # 将编码特征 encoded_features 输入到量化器 self.quantizer 中，返回量化后的特征和特征长度
        return self.quantizer.encode(encoded_features), feature_lengths
