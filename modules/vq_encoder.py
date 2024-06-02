import torch
from torch import nn
import math
from .fsq import DownsampleFiniteScalarQuantize
from .wavenet import WaveNet
from .spectrogram import LogMelSpectrogram
from .utils import sequence_mask

class VQEncoder(nn.Module):
    def __init__(self, sample_rate=44100, n_fft=2048, hop_length=512, n_mels=128, win_length=2048):
        super().__init__()
        self.encoder = WaveNet(
            input_channels=n_mels,
            residual_channels=768,
            residual_layers=20,
            dilation_cycle=4,
        )
        self.quantizer = DownsampleFiniteScalarQuantize(
            input_dim=768, n_codebooks=1, n_groups=2, levels=[8, 5, 5, 5]
        )
        # 
        self.spec = LogMelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
            n_mels=n_mels,
            f_min=0.0,
            f_max=8000.0,
        )

    @torch.no_grad()
    def forward(self, audios, audio_lengths, sr=None):
        mel_spec = self.spec(audios, sample_rate=sr)
        if sr is not None:
            audio_lengths = audio_lengths * 44100 // sr
        mel_lengths = audio_lengths // self.spec.hop_length
        mel_masks = (torch.arange(mel_spec.shape[2], device=mel_spec.device) < mel_lengths[:, None])
        mel_masks_float_conv = mel_masks[:, None, :].float()
        mels = mel_spec * mel_masks_float_conv
        encoded_features = self.encoder(mels) * mel_masks_float_conv
        encoded_features = self.quantizer(encoded_features).z * mel_masks_float_conv
        return encoded_features

    @torch.no_grad()
    def encode(self, audios, audio_lengths, sr=None):
        audios = audios.float()

        # 将音频转换为梅尔频谱图
        mels = self.spec(audios, sample_rate=sr)
        mel_lengths = audio_lengths // self.spec.hop_length
        # 生成掩码以处理不同长度的音频
        mel_masks = sequence_mask(mel_lengths, mels.shape[2])
        mel_masks_float_conv = mel_masks[:, None, :].float()
        mels = mels * mel_masks_float_conv

        # Encode
        encoded_features = self.encoder(mels) * mel_masks_float_conv
        feature_lengths = mel_lengths // math.prod(self.quantizer.downsample_factor)

        # 返回量化后的特征和特征长度
        return self.quantizer.encode(encoded_features), feature_lengths
