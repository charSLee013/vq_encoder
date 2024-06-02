from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from vector_quantize_pytorch import GroupedResidualFSQ

from .firefly import ConvNeXtBlock


# 用于存储量化结果，包括量化后的特征 z、量化编码 codes 和潜在特征 latents
@dataclass
class FSQResult:
    z: torch.Tensor
    codes: torch.Tensor
    latents: torch.Tensor


# 用于对输入特征进行下采样、量化和上采样
class DownsampleFiniteScalarQuantize(nn.Module):
    def __init__(
        self,
        input_dim: int = 512,       # 输入特征的维度
        n_codebooks: int = 9,       # 量化器的码本数量
        n_groups: int = 1,          # 量化器的组数
        levels: tuple[int] = (8, 5, 5, 5),  # Approximate 2**10 # 量化器的层级
        downsample_factor: tuple[int] = (2, 2), # 下采样因子
        downsample_dims: tuple[int] | None = None,  # 下采样维度
    ):
        super().__init__()

        if downsample_dims is None:
            downsample_dims = [input_dim for _ in range(len(downsample_factor))]

        all_dims = (input_dim,) + tuple(downsample_dims)

        # 用于残差量化
        self.residual_fsq = GroupedResidualFSQ(
            dim=all_dims[-1],
            levels=levels,
            num_quantizers=n_codebooks,
            groups=n_groups,
        )

        self.downsample_factor = downsample_factor
        self.downsample_dims = downsample_dims

        self.downsample = nn.Sequential(
            *[
                nn.Sequential(
                    nn.Conv1d(
                        all_dims[idx],
                        all_dims[idx + 1],
                        kernel_size=factor,
                        stride=factor,
                    ),
                    ConvNeXtBlock(dim=all_dims[idx + 1]),
                )
                for idx, factor in enumerate(downsample_factor)
            ]
        )

        self.upsample = nn.Sequential(
            *[
                nn.Sequential(
                    nn.ConvTranspose1d(
                        all_dims[idx + 1],
                        all_dims[idx],
                        kernel_size=factor,
                        stride=factor,
                    ),
                    ConvNeXtBlock(dim=all_dims[idx]),
                )
                for idx, factor in reversed(list(enumerate(downsample_factor)))
            ]
        )

        self.apply(self._init_weights)

    def _init_weights(self, m):
        """初始化卷积层和线性层的权重和偏置
        """
        if isinstance(m, (nn.Conv1d, nn.Linear)):
            nn.init.trunc_normal_(m.weight, std=0.02)
            nn.init.constant_(m.bias, 0)

    def forward(self, z) -> FSQResult:
        """对输入特征进行下采样
        使用残差量化器进行量化
        对量化后的特征进行上采样
        调整上采样后的特征以匹配原始形状
        """
        original_shape = z.shape
        z = self.downsample(z)
        quantized, indices = self.residual_fsq(z.mT)
        result = FSQResult(
            z=quantized.mT,
            codes=indices.mT,
            latents=z,
        )
        result.z = self.upsample(result.z)

        # Pad or crop z to match original shape
        diff = original_shape[-1] - result.z.shape[-1]
        left = diff // 2
        right = diff - left

        if diff > 0:
            result.z = F.pad(result.z, (left, right))
        elif diff < 0:
            result.z = result.z[..., left:-right]

        return result

    def encode(self, z):
        """对输入特征进行下采样
        使用残差量化器进行量化，并返回量化编码
        """
        z = self.downsample(z)
        _, indices = self.residual_fsq(z.mT)
        indices = rearrange(indices, "g b l r -> b (g r) l")
        return indices

    def decode(self, indices: torch.Tensor):
        """使用量化编码解码出特征
        对解码后的特征进行上采样
        """
        indices = rearrange(indices, "b (g r) l -> g b l r", g=self.residual_fsq.groups)
        z_q = self.residual_fsq.get_output_from_indices(indices)
        z_q = self.upsample(z_q.mT)
        return z_q

    # def from_latents(self, latents: torch.Tensor):
    #     z_q, z_p, codes = super().from_latents(latents)
    #     z_q = self.upsample(z_q)
    #     return z_q, z_p, codes


if __name__ == "__main__":
    rvq = DownsampleFiniteScalarQuantize(
        n_codebooks=1,
        downsample_factor=(2, 2),
    )
    x = torch.randn(16, 512, 80)

    result = rvq(x)
    print(rvq)
    print(result.latents.shape, result.codes.shape, result.z.shape)

    # y = rvq.from_codes(result.codes)
    # print(y[0].shape)

    # y = rvq.from_latents(result.latents)
    # print(y[0].shape)

"""
下采样是指减少数据的采样率或分辨率。在TTS系统中，下采样通常用于减少特征的时间分辨率或空间分辨率，以便更高效地处理数据。

举例：假设你有一个音频信号，每秒钟有44100个采样点（即采样率为44.1kHz）。如果你将采样率降低到22050个采样点每秒（即22.05kHz），这就是下采样。这样做的好处是减少了数据量，使得处理和存储更加高效。

在代码中的体现：
```python
self.downsample = nn.Sequential(
    *[
        nn.Sequential(
            nn.Conv1d(
                all_dims[idx],
                all_dims[idx + 1],
                kernel_size=factor,
                stride=factor,
            ),
            ConvNeXtBlock(dim=all_dims[idx + 1]),
        )
        for idx, factor in enumerate(downsample_factor)
    ]
)
```
这里的 nn.Conv1d 和 stride=factor 就是实现下采样的关键部分，通过卷积操作和步幅（stride）来减少特征的分辨率

上采样是指增加数据的采样率或分辨率。在TTS系统中，上采样通常用于恢复下采样前的分辨率，以便生成高质量的输出。

举例：如果你有一个下采样后的音频信号，每秒钟有22050个采样点（即采样率为22.05kHz），你可以通过上采样将其恢复到每秒钟44100个采样点（即44.1kHz）。这样做的目的是恢复原始信号的细节。

在代码中的体现：
```python
self.upsample = nn.Sequential(
    *[
        nn.Sequential(
            nn.ConvTranspose1d(
                all_dims[idx + 1],
                all_dims[idx],
                kernel_size=factor,
                stride=factor,
            ),
            ConvNeXtBlock(dim=all_dims[idx]),
        )
        for idx, factor in reversed(list(enumerate(downsample_factor)))
    ]
)
```
这里的 nn.ConvTranspose1d 就是实现上采样的关键部分，通过转置卷积操作来增加特征的分辨率。


残差量化是一种量化技术，用于将连续的特征表示转换为离散的表示，同时尽量减少量化误差。在TTS系统中，残差量化可以帮助模型更高效地表示和生成音频信号。

基本概念：

量化：将连续的数值转换为离散的数值。例如，将一个范围在0到1之间的浮点数转换为0、0.25、0.5、0.75、1这五个离散值。
残差：量化过程中产生的误差。残差量化通过多次量化和编码残差来减少这种误差。
举例：假设你有一个值0.68，你想将其量化为0.5或0.75。直接量化会产生误差（残差）。残差量化通过多次迭代量化和编码残差来减少这种误差，使得最终的量化结果更接近原始值。

在代码中的体现：
```
self.residual_fsq = GroupedResidualFSQ(
    dim=all_dims[-1],
    levels=levels,
    num_quantizers=n_codebooks,
    groups=n_groups,
)
```
这里的 GroupedResidualFSQ 就是实现残差量化的关键部分，通过多级量化和编码残差来减少量化误差。
"""