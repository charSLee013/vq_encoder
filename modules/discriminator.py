import random
import torch
from torch import nn
from torch.nn.utils.parametrizations import weight_norm


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()

        blocks = []
        convs = [
            (1, 64, (3, 9), 1, (1, 4)),
            (64, 128, (3, 9), (1, 2), (1, 4)),
            (128, 256, (3, 9), (1, 2), (1, 4)),
            (256, 512, (3, 9), (1, 2), (1, 4)),
            (512, 1024, (3, 3), 1, (1, 1)),
            (1024, 1, (3, 3), 1, (1, 1)),
        ]

        for idx, (in_channels, out_channels, kernel_size, stride, padding) in enumerate(
            convs
        ):
            blocks.append(
                weight_norm(
                    nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
                )
            )

            if idx != len(convs) - 1:
                blocks.append(nn.SiLU(inplace=True))

        self.blocks = nn.Sequential(*blocks)

    def forward(self, x):
        return self.blocks(x[:, None])[:, 0]

"""AdaptiveAvgPool2d 的工作原理：
自适应：意味着该层会根据输入的实际尺寸动态调整池化窗口的大小，以达到指定的输出尺寸。与之相对的是固定大小的池化层（如AvgPool2d），后者要求预先设定固定的窗口尺寸。

平均池化：在每个通道中，对选定的窗口内的元素取平均值作为输出该窗口位置的结果。这有助于减少空间维度上的信息量，同时保留每个区域的大致信息，实现一定程度的下采样和空间维度的压缩，有助于减小计算量和参数量，也可以增加模型的泛化能力。

### 示例理解：

假设您的卷积层输出了一个形状为 `(batch_size, channels, height, width)` 的特征图，比如 `(8, 512, 9, 9)`，代表了8个样本，每个样本有512个通道，特征图的高和宽分别为9像素。经过 `nn.AdaptiveAvgPool2d((1, 1))` 后，每个样本的每个通道都会被压缩成一个单一的值，输出形状将变为 `(8, 512, 1, 1)`。随后，通过视图调整（`.view(x.size(0), -1)`)，进一步压平为 `(8, 512)` 形状的向量，可以直接输入到全连接层中进行分类或回归任务。

### 为何重要：

- **灵活性**：能够处理不同尺寸的输入，适用于例如图像识别任务中图片尺寸不统一的场景。
- **简化结构**：简化模型设计，不需要手动计算每次池化的步长和窗口大小。
- **特征聚合**：有效整合局部特征，减少计算负担的同时保留关键信息，为后续全连接层的分类或回归任务提供准备。
"""

class DynamicAudioDiscriminator(nn.Module):
    def __init__(self, num_mels=100):
        super(DynamicAudioDiscriminator, self).__init__()
        
        self.conv_blocks = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=(5, 1), stride=(2, 1), padding=(2, 0)),  # 考虑num_mels调整卷积核大小
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, kernel_size=(5, 1), stride=(2, 1), padding=(2, 0)),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 256, kernel_size=(5, 1), stride=(2, 1), padding=(2, 0)),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            nn.Conv2d(256, 512, kernel_size=(5, 1), stride=(1, 1), padding=(2, 0))
        )

        self.global_pooling = nn.AdaptiveAvgPool2d((1, 1))  # 全局平均池化到一个点
        self.fc = nn.Linear(512, 1)  
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        # 确保输入符合预期的通道数（单通道）
        x = x.unsqueeze(1)
        
        # 动态通过卷积层
        x = self.conv_blocks(x)
        
        # 使用全局平均池化适应不同尺寸的输入
        x = self.global_pooling(x)
        
        # 将池化后的张量展平以适应全连接层
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        output = self.sigmoid(x)
        return output.squeeze()
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def run_demo(device):
    # 定义模型并转移到设备
    discriminator = DynamicAudioDiscriminator(num_mels=100).to(device)

    # 准备测试数据并转移到设备
    batch_size = 8
    time_steps = random.randint(100,1024)
    random_data = torch.randn(batch_size, 100, time_steps, device=device)

    # 将模型设置为评估模式
    discriminator.eval()

    # 执行前向传播
    with torch.no_grad():
        predictions = discriminator(random_data)

    # 输出预测结果
    print("Predictions shape:", predictions.shape)
    print("Example of predictions:", predictions)
    

if __name__ == "__main__":
    # model = Discriminator()
    # print(sum(p.numel() for p in model.parameters()) / 1_000_000)
    # x = torch.randn(8, 100, 1024)
    # y = model(x)
    # print(y.shape)
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    run_demo(device)