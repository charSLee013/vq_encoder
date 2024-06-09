import math
from typing import List
from einops import rearrange
from vector_quantize_pytorch import GroupedResidualFSQ

import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvNeXtBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        intermediate_dim: int,
        kernel, dilation,
        layer_scale_init_value: float = 1e-6,
    ):
        # ConvNeXt Block copied from Vocos.
        super().__init__()
        self.dwconv = nn.Conv1d(dim, dim, 
                                kernel_size=kernel, padding=dilation*(kernel//2), 
                                dilation=dilation, groups=dim
                            )  # depthwise conv
        
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, intermediate_dim)  # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(intermediate_dim, dim)
        self.gamma = (
            nn.Parameter(layer_scale_init_value * torch.ones(dim), requires_grad=True)
            if layer_scale_init_value > 0
            else None
        )

    def forward(self, x: torch.Tensor, cond = None) -> torch.Tensor:
        residual = x
        x = self.dwconv(x)
        x = x.transpose(1, 2)  # (B, C, T) -> (B, T, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.transpose(1, 2)  # (B, T, C) -> (B, C, T)

        x = residual + x
        return x
    


class GFSQ(nn.Module):
    """GFSQ（Grouped Residual Finite State Quantization）模块的主要作用是对输入特征进行量化处理。量化是将连续的数值特征映射到离散的符号空间，这在压缩和特征表示中非常有用。GFSQ 模块通过分组残差量化技术，将输入特征分成多个组，每组特征通过多个量化器进行量化，从而实现高效的特征表示和压缩。
    """

    def __init__(self, 
            dim, # 输入维度
            levels:List[int], # 量化的粒度，即每个特征映射到多少个离散级别。
            G,  # 输入特征分为的组数
            R, # 每组量化器的数量
            eps=1e-5,   # 较小的 epsilon 值以防止被零除 
            transpose = True    # 是否转置输入的维度
        ):
        super(GFSQ, self).__init__()
        self.quantizer = GroupedResidualFSQ(
            dim=dim,
            levels=levels,
            num_quantizers=R,  # 量化等级数量
            groups=G,       # GFSQ的分组数量
        )
        # 索引总数的计算公式为len(levels)
        self.n_ind = math.prod(levels)
        self.eps = eps
        self.transpose = transpose
        self.G = G
        self.R = R
        
    def _embed(self, x):
        """将输入特征通过量化器进行量化，并返回量化后的特征
        """
        if self.transpose:
            # 将输入特征的维度从 (B, T, C) 转置为 (B, C, T)
            x = x.transpose(1,2)
        # 将输入特征重排列为 (g, b, t, r) 的形状，其中 g 是组数，b 是批次大小，t 是时间步，r 是每组的量化器数
        x = rearrange(
            x, "b t (g r) -> g b t r", g = self.G, r = self.R,
        )  
        feat = self.quantizer.get_output_from_indices(x)
        return feat.transpose(1,2) if self.transpose else feat
        
    def forward(self, x,):
        # 将输入特征的维度从 (B, T, C) 转置为 (B, C, T)，以适应模型的处理需求
        if self.transpose:
            x = x.transpose(1,2)
        # 使用quantizer对输入特征进行量化处理，得到量化后的特征和量化索引
        feat, ind = self.quantizer(x)
        # 这里的g是组数，b是批次大小，t是时间步，r是每组的量化器数。
        # 这种重排是为了将分组的信息合并，便于后续处理
        ind = rearrange(
            ind, "g b t r ->b t (g r)",
        )  
        # 对量化索引进行One-hot编码，这是为了计算每个量化级别的使用频率，进而计算模型的多样性（即复杂度
        embed_onehot = F.one_hot(ind.long(), self.n_ind).to(x.dtype)
        # 计算One-hot编码的均值，然后用这个均值来计算每个量化级别的使用频率
        e_mean = torch.mean(embed_onehot, dim=[0,1])
        # 对平均独热编码进行归一化，确保每行的和为1。self.eps是一个小的常数，用于防止除以零
        e_mean = e_mean / (e_mean.sum(dim=1) + self.eps).unsqueeze(1)
        # 使用这些频率来计算复杂度（perplexity），这是一个衡量量化效果多样性的指标，复杂度越高
        # 表示模型的量化效果越丰富，能更好地捕捉输入数据的特征
        perplexity = torch.exp(-torch.sum(e_mean * torch.log(e_mean + self.eps), dim=1))
        
        return (
            torch.zeros(perplexity.shape, dtype=x.dtype, device=x.device),  # 零张量，占位符或用于计算某些指标
            feat.transpose(1,2) if self.transpose else feat,                # 量化特征张量,形状为[batch_size, channels, time_steps]
            perplexity, # 复杂度值,用于衡量量化器的使用情况
            None,       # 占位
            ind.transpose(1,2) if self.transpose else ind,  # 这是量化后的索引张量,被转置回原始形状 [batch_size, groups, time_steps]
        )
        
class DVAEDecoder(nn.Module):
    def __init__(self, 
                 idim,  # 输入维度
                 odim,  # 输出维度
                 n_layer = 12, # 解码层数
                 bn_dim = 64,  # 初始卷积序列中瓶颈层的尺寸。
                 hidden = 256, # 隐藏层的数量
                 kernel = 7,   # 解码器块中卷积层的内核大小。
                 dilation = 2, # 解码器块中卷积层的膨胀率。
                 up = False    # 是否开启升采样
                ):
        super().__init__()
        self.up = up
        # 输入的潜在表示首先由初始卷积层（conv_in）处理。
        # 这些层旨在将潜在表示扩展到适合进一步处理的更详细的特征空间。
        self.conv_in = nn.Sequential(
            nn.Conv1d(idim, bn_dim, 3, 1, 1), nn.GELU(),
            nn.Conv1d(bn_dim, hidden, 3, 1, 1)
        )
        # 然后，扩展的特征通过多个层（decoder_block）传递，每个层都可能添加更多细节并细化特征。
        # 这些层对于捕获音频数据中的复杂模式至关重要，其中包括时间动态和频率特征。
        self.decoder_block = nn.ModuleList([
            ConvNeXtBlock(hidden, hidden* 4, kernel, dilation,)
            for _ in range(n_layer)])
        # 最后一个解码器块的输出通过卷积层（conv_out）转换为最终音频信号。
        # 该层将处理后的特征映射回原始音频空间（或与其非常相似的空间）。
        self.conv_out = nn.Conv1d(hidden, odim, kernel_size=3,stride=1,padding=1,dilation=1 ,bias=False)

    def forward(self, input, conditioning=None):
        # B, T, C
        # # 首先对输入进行转置以匹配预期的维度（批次、通道、序列长度）。
        # x = input.transpose(1, 2)
        # 它通过 conv_in 进行初始处理。 
        x = self.conv_in(x)
        # 使用额外的调节数据按顺序处理 conv_in 的输出。
        for f in self.decoder_block:
            x = f(x, conditioning)
        # 最后，最后一个解码器块的输出通过 conv_out 传递
        x = self.conv_out(x)
        # # 并将结果转置回原始维度。
        # return x.transpose(1, 2)
        return x


class DVAE(nn.Module):
    def __init__(
        self, decoder_config, vq_config, dim=512
    ):
        super().__init__()
        # 返回一个张量，该张量填充了平均值为 0、方差为 1 的正态分布（也称为标准正态分布）的随机数。
        self.register_buffer('coef', torch.randn(1, 100, 1))
        # self.register_buffer('coef', torch.load('coef.pt'))

        self.decoder = DVAEDecoder(**decoder_config)
        self.out_conv = nn.Conv1d(dim, 100, 3, 1, 1, bias=False)
        if vq_config is not None:
            self.vq_layer = GFSQ(**vq_config)
        else:
            self.vq_layer = None

    def forward(self, inp):
        """
        假设inp的形状为torch.Size([1, 4, 121])
        """

        if self.vq_layer is not None:
            # vq_feats的形状为torch.Size([1, 1024, 121])
            # 使用 GFSQ 的 _embed 函数来量化输入特征。然后将量化的特征（vq_feats）用于进一步处理
            vq_feats = self.vq_layer._embed(inp)
        else:
            vq_feats = inp.detach().clone()
        
        # 通过调整量化特征的维度来准备解码
        # 将特征沿着 dim=1 维度分成两部分，得到两个形状为 (1, 512, 121) 的张量。
        temp = torch.chunk(vq_feats, 2, dim=1) # flatten trick :)
        #将这两个张量堆叠在一起，得到形状为 (1, 512, 121, 2) 的张量
        temp = torch.stack(temp, -1)
        # 重新调整特征形状，得到 vq_feats 形状为 (1, 512, 242)
        vq_feats = temp.reshape(*temp.shape[:2], -1)
        # 将特征 vq_feats 转置,转置后的特征 vq_feats 形状为 (1, 242, 512)
        vq_feats = vq_feats.transpose(1, 2)
        
        # 根据量化和处理过的特征重建音频信号,解码后的输出 dec_out 形状为 (1, 242, 512)
        # DVAEDecoder 利用准备好的特征，通过卷积和残差块对其进行处理，以生成接近原始音频信号的解码输出。
        dec_out = self.decoder(input=vq_feats)
        
        # 解码后的输出被转置并通过附加的卷积层 (out_conv) 以调整其维度。
        # 然后将结果按系数张量 (self.coef) 缩放以产生最终输出，即音频信号的梅尔频谱图。
        # 先进行转置后的 dec_out 形状为 (1, 512, 242)
        # 然后再通过卷积层后，dec_out 形状为 (1, 100, 242)
        dec_out = self.out_conv(dec_out.transpose(1, 2))
        # Mel 频谱图 mel 的形状为 (1, 100, 242)
        mel = dec_out * self.coef

        return mel
