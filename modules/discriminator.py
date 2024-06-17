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
        self.global_pooling = nn.AdaptiveAvgPool2d((1, 1))  # 添加全局自适应平均池化
        self.fc = nn.Linear(1, 1)  # 添加全连接层
        self.sigmoid = nn.Sigmoid()  # 添加Sigmoid激活函数

    def forward(self, x):
        x = self.blocks(x[:, None])
        x = self.global_pooling(x)  # 应用全局自适应平均池化
        x = x.view(x.size(0), -1)  # 将池化后的张量展平
        x = self.fc(x)  # 通过全连接层
        output = self.sigmoid(x)  # 应用Sigmoid激活函数输出概率
        return output.squeeze()

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
            nn.Conv2d(1, 64, kernel_size=(5, num_mels), stride=(2, 1), padding=(2, 1)),  # 考虑num_mels调整卷积核大小
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

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.LeakyReLU(0.2)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, 1, padding)
        self.bn2 = nn.BatchNorm2d(out_channels)
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        residual = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x += self.shortcut(residual)  # 残差连接
        x = self.relu(x)  # 再次应用激活函数
        return x

class DynamicAudioDiscriminatorWithResidual(nn.Module):
    def __init__(self, num_mels=100):
        super(DynamicAudioDiscriminatorWithResidual, self).__init__()
        
        self.conv1 = nn.Conv2d(1, 64, kernel_size=(3, 9), stride=1, padding=(1, 4))
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.LeakyReLU(0.2)
        
        self.res_blocks = nn.Sequential(
            ResidualBlock(64, 128, (5, 1), (2, 1), (2, 0)),
            ResidualBlock(128, 256, (5, 1), (2, 1), (2, 0)),
            ResidualBlock(256, 512, (5, 1), (2, 1), (2, 0)),
            # ResidualBlock(512, 1024, (5, 1), (2, 1), (2, 0)),
        )
        # 根据最后一个残差块的输出通道数动态设置全连接层
        last_res_block_out_channels = self.res_blocks[-1].conv2.out_channels  # 获取最后一个残差块的输出通道数
        self.fc = nn.Linear(last_res_block_out_channels, 1)  # 设置全连接层
        self.global_pooling = nn.AdaptiveAvgPool2d((1, 1))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        
        x = self.res_blocks(x)
        
        x = self.global_pooling(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        # output = self.sigmoid(x)
        # return output.squeeze()
        return x
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def run_demo(model:torch.nn.Module,device):
    # 定义模型并转移到设备
    discriminator = model.to(device)

    # 准备测试数据并转移到设备
    batch_size = 8
    time_steps = 99
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
    # print(y)
    # device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    device = 'cpu'
    model = Discriminator()
    run_demo(model,device)
    model = DynamicAudioDiscriminator()
    run_demo(model,device)
    model = DynamicAudioDiscriminatorWithResidual()
    run_demo(model,device)