import torch
import torchaudio.functional as F
from torch import Tensor, nn
from torchaudio.transforms import MelScale


class LinearSpectrogram(nn.Module):
    def __init__(
        self,
        n_fft=2048,         # 快速傅里叶变换的大小，决定频谱的分辨率。
        win_length=2048,    # 窗口长度，通常与 n_fft 相同
        hop_length=512,     # 每次滑动的步长，决定时间分辨率
        center=False,       # 是否在信号中心进行填充
        mode="pow2_sqrt",   # 计算模式，这里使用 "pow2_sqrt" 模式
    ):
        super().__init__()

        self.n_fft = n_fft
        self.win_length = win_length
        self.hop_length = hop_length
        self.center = center
        self.mode = mode

        # 注册一个汉宁窗（Hann window），用于加窗处理
        self.register_buffer("window", torch.hann_window(win_length), persistent=False)

    def forward(self, y: Tensor) -> Tensor:
        """计算线性谱图

        Args:
            y (Tensor): 输入的音频信号

        Returns:
            Tensor: 返回计算后的线性谱图
        """
        # 如果输入是三维张量，则压缩维度
        if y.ndim == 3:
            y = y.squeeze(1)

        # 对输入信号进行填充，以适应窗口长度和步长
        y = torch.nn.functional.pad(
            y.unsqueeze(1),
            (
                (self.win_length - self.hop_length) // 2,
                (self.win_length - self.hop_length + 1) // 2,
            ),
            mode="reflect",
        ).squeeze(1)
        
        # 使用 torch.stft 进行短时傅里叶变换（STFT），生成复数频谱
        spec = torch.stft(
            y,
            self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window,
            center=self.center,
            pad_mode="reflect",
            normalized=False,
            onesided=True,
            return_complex=True,
        )

        # 根据模式计算频谱的幅度
        spec = torch.view_as_real(spec)

        if self.mode == "pow2_sqrt":
            spec = torch.sqrt(spec.pow(2).sum(-1) + 1e-6)

        return spec


class LogMelSpectrogram(nn.Module):
    def __init__(
        self,
        sample_rate=44100,      # 采样率
        n_fft=2048,             # FFT 的大小
        win_length=2048,        # 窗口长度
        hop_length=512,         # 步长
        n_mels=128,             # 梅尔频谱的通道数
        center=False,           # 是否在信号中心进行填充
        f_min=0.0,              # 最小频率
        f_max=None,             # 最高频率，默认为采样率的一半
    ):
        super().__init__()

        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.win_length = win_length
        self.hop_length = hop_length
        self.center = center
        self.n_mels = n_mels
        self.f_min = f_min
        self.f_max = f_max or float(sample_rate // 2)

        self.spectrogram = LinearSpectrogram(n_fft, win_length, hop_length, center)

        fb = F.melscale_fbanks(
            n_freqs=self.n_fft // 2 + 1,
            f_min=self.f_min,
            f_max=self.f_max,
            n_mels=self.n_mels,
            sample_rate=self.sample_rate,
            norm="slaney",
            mel_scale="slaney",
        )
        # 计算梅尔滤波器组，并注册为缓冲区
        self.register_buffer(
            "fb",
            fb,
            persistent=False,
        )

    # 对线性谱图进行压缩
    def compress(self, x: Tensor) -> Tensor:
        return torch.log(torch.clamp(x, min=1e-5))

    # 对线性谱图进行解压缩
    def decompress(self, x: Tensor) -> Tensor:
        return torch.exp(x)

    # 应用梅尔尺度函数
    def apply_mel_scale(self, x: Tensor) -> Tensor:
        """将线性谱图转换为梅尔谱图
        """
        return torch.matmul(x.transpose(-1, -2), self.fb).transpose(-1, -2)

    def forward(
        self, x: Tensor, return_linear: bool = False, sample_rate: int = None
    ) -> Tensor:
        """_summary_

        Args:
            x (Tensor): 输入的音频信号，类型为 Tensor
            return_linear (bool, optional): 是否返回线性谱图 Defaults to False.
            sample_rate (int, optional): 输入信号的采样率. Defaults to None.

        Returns:
            Tensor: _description_
        """
        # 如果提供了采样率且与类的采样率不同，则对输入信号进行重采样
        if sample_rate is not None and sample_rate != self.sample_rate:
            x = F.resample(x, orig_freq=sample_rate, new_freq=self.sample_rate)

        # 计算线性谱图
        linear = self.spectrogram(x)
        # 应用梅尔尺度转换
        x = self.apply_mel_scale(linear)
        # 对梅尔谱图进行对数压缩
        x = self.compress(x)

        # 根据 return_linear 参数决定是否返回线性谱图
        if return_linear:
            return x, self.compress(linear)

        return x
