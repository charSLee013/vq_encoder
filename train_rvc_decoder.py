#!/usr/bin/env python
# coding: utf-8

# ## 将通过RVC换声后的音频数据用来训练Decoder解码器以实现音色固定的效果

# In[1]:


import torch
from torch.nn.functional import pad

def collate_fn(batch):
    """
    Custom collate function with hidden transposed and shapes maintained throughout.
    
    Args:
        batch (List[Tuple[Tensor, Tensor]]): List of tuples, each containing a pair of hidden and log_mel_spec tensors.
        
    Returns:
        Dict[str, Tensor]: A dictionary with keys 'hidden' and 'log_mel_spec', 
        each associated with a padded tensor, with 'hidden' remaining in its transposed shape.
    """
    # Separate inputs and transpose hidden
    hidden_list_transposed, log_mel_spec_list = zip(*[(h.transpose(0, 1), l) for h, l in batch])
    
    # Calculate the maximum lengths based on the transposed shapes
    max_len_hidden = max([h.size(1) for h in hidden_list_transposed])  # Now looking at the first dimension after transpose
    max_len_log_mel_spec = max_len_hidden * 2
    
    # Pad hidden and log_mel_spec sequences
    hidden_padded = torch.stack([pad(h, (0, max_len_hidden - h.size(1)), value=0) for h in hidden_list_transposed])
    log_mel_spec_padded = torch.stack([pad(l, (0, max_len_log_mel_spec - l.size(1)), value=0) for l in log_mel_spec_list])
    
    # Return a dictionary with padded tensors, hidden remains in its transposed state
    return {
        'hidden': hidden_padded,  # Already in the desired transposed shape
        'log_mel_spec': log_mel_spec_padded
    }


# In[2]:


import torch
from torch.utils.data import Dataset
import numpy as np
import glob
from torch.utils.data import DataLoader, random_split
import os
device = torch.device('cuda:0' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    
if 'mps' in str(device):
    os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] ='1'
    # 在jupyter notebook 里面还需要额外引入一个魔法命令
    try:
        # 只在jupyter notebook中运行
        from IPython import get_ipython
        if get_ipython() is not None:
            get_ipython().run_line_magic('set_env', 'PYTORCH_ENABLE_MPS_FALLBACK=1')
    except:
        pass

class MelSpecDataset(Dataset):
    def __init__(self, data_dir):
        super(MelSpecDataset, self).__init__()
        self.data_dir = data_dir
        self.npz_files = glob.glob(f"{data_dir}/*.npz")

    def __len__(self):
        return len(self.npz_files)

    def __getitem__(self, idx):
        npz_file = self.npz_files[idx]
        data = np.load(npz_file)
        
        # 获取输入特征和目标特征
        hidden = data["hidden"]
        log_mel_spec = data["log_mel_spec"]
        
        # 将 NumPy 数组转换为 PyTorch 张量
        hidden = torch.from_numpy(hidden).float()
        log_mel_spec = torch.from_numpy(log_mel_spec).float()
        
        # 返回输入和目标张量
        return hidden, log_mel_spec


# In[3]:


from modules.dvae import DVAEDecoder
import pytorch_lightning as pl
from torch.optim import Adam
from torch.nn import MSELoss, L1Loss
import torch
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from pytorch_lightning.loggers import TensorBoardLogger
from modules.discriminator import DynamicAudioDiscriminatorWithResidual

class LightningDVAEDecoder(pl.LightningModule):
    def __init__(self, idim, odim, n_layer=12, bn_dim=64, hidden=256, kernel=7, dilation=2, up=False):
        super().__init__()
        self.save_hyperparameters()  # 这一行会自动保存模型的参数为超参数
        self.model = DVAEDecoder(
            idim, odim, n_layer, bn_dim, hidden, kernel, dilation, up)
        self.loss_fn = MSELoss()  # 假设我们使用均方误差作为损失函数
        self.l1_loss_fn = L1Loss()  # 使用 L1 损失作为辅助损失函数

    def forward(self, vq_feats):
        # 通过调整量化特征的维度来准备解码
        # 将特征沿着 dim=1 维度分成两部分，得到两个形状为 (1, 512, 121) 的张量。
        temp = torch.chunk(vq_feats, 2, dim=1)  # flatten trick :)
        # 将这两个张量堆叠在一起，得到形状为 (1, 512, 121, 2) 的张量
        temp = torch.stack(temp, -1)
        # 重新调整特征形状，得到 vq_feats 形状为 (1, 512, 242)
        vq_feats = temp.reshape(*temp.shape[:2], -1)

        return self.model(vq_feats)

    def training_step(self, batch, batch_idx):
        # 注意：这里直接使用collate_fn处理后的数据结构
        hidden, target = batch['hidden'], batch['log_mel_spec']
        output = self(hidden)

        loss = self.loss_fn(output, target)
        self.log('train_loss', loss, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        hidden, target = batch['hidden'], batch['log_mel_spec']
        output = self(hidden)

        # 计算 MSE 损失
        mse_loss = self.loss_fn(output, target)
        self.log('val_mse_loss', mse_loss, prog_bar=True, logger=True)

        # 计算 L1 损失
        l1_loss = self.l1_loss_fn(output, target)
        self.log('val_l1_loss', l1_loss, prog_bar=True, logger=True)

        # 返回一个字典，包含所有记录的损失值
        return {'val_mse_loss': mse_loss, 'val_l1_loss': l1_loss}

    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=1e-3)  # 你可以根据需求调整学习率和其他参数
        # 余弦退火
        scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=128, T_mult=2, eta_min=1e-6)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1,
                "monitor": "val_loss",
                "strict": True
            }
        }
#         scheduler = CosineAnnealingLR(optimizer, T_max=100, eta_min=1e-6) 

#         # 返回optimizer和scheduler的字典给PyTorch Lightning
#         return {
#             "optimizer": optimizer,
#             "lr_scheduler": {
#                 "scheduler": scheduler,
#                 "interval": "epoch",
#                 "frequency": 1,  # 由于CosineAnnealingLR通常是每个epoch更新，所以frequency为1
#                 "monitor": "val_loss",  # 仍然监控验证损失来决定是否调整学习率
#                 "strict": True
#             }
#         }


# In[ ]:


# Faster, but less precise
torch.set_float32_matmul_precision("high")

# 使用自定义的 Dataset 类
dataset = MelSpecDataset("./train_rvc")

# Accessing a sample
sample_hidden, sample_mel_spec = dataset[0]
print(f"Sample hidden shape: {sample_hidden.shape}")
print(f"Sample mel spec shape: {sample_mel_spec.shape}")

# 设定训练集和验证集的比例
train_val_ratio = 0.90
val_size = int(len(dataset) * (1 - train_val_ratio))
train_size = len(dataset) - val_size

# 使用 random_split 分割数据集
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

if 'cuda' in str(device):
    batch_size = 36
else:
    batch_size = 4

# 创建 DataLoader 以批量加载数据
train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn,num_workers=4)
val_loader = torch.utils.data.DataLoader(
    val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn,num_workers=4)

# 打印数量
print(f"Train dataset size: {len(train_dataset)}")
print(f"Validation dataset size: {len(val_dataset)}")

# 实例化你的 Lightning 模型
IDIM = 384
ODIM = 100
model_params = {
    'idim':IDIM, 
    'odim':ODIM, 
    'bn_dim':128,
    'hidden':256,
    'n_layer':4,
}
model = LightningDVAEDecoder(**model_params)

# 自定义检查点保存的目录
checkpoint_dir = './checkpoints'
latest_checkpoint = sorted(glob.glob(os.path.join(
    checkpoint_dir, '*.ckpt')), key=os.path.getmtime, reverse=True)

# 检查是否有可用的检查点文件
if latest_checkpoint:
    last_checkpoint_path = latest_checkpoint[0]
    print(
        f"Resuming training from the latest checkpoint: {last_checkpoint_path}")
else:
    print("No checkpoint found. Starting training from scratch.")
    last_checkpoint_path = None


# 添加 ModelCheckpoint 回调，设置保存条件和频率
checkpoint_callback = ModelCheckpoint(
    monitor='step',  # 监控的指标，用于决定是否保存模型
    mode='max',          # 监控指标的模式，这里是最小化验证损失
    save_top_k=10,       # 保持最新的 10 个模型文件
    # every_n_epochs=1,    # 每个 epoch 保存一次模型
    filename='model-{epoch:02d}-{val_loss:.4f}',  # 模型文件名的格式
    auto_insert_metric_name=False,  # 不自动在文件名中插入监控的指标名,
    every_n_train_steps=2000,  # save checkpoints every 2000 steps
    dirpath=checkpoint_dir,
)

# 添加 EarlyStopping 回调，设置早停条件
early_stop_callback = pl.callbacks.EarlyStopping(
    monitor='val_mse_loss',
    min_delta=0.1,  # 最小变化量，小于这个值的变化不会触发早停
    patience=10,       # 在满足条件后，继续训练的 epoch 数
    verbose=True,      # 是否在控制台输出早停信息
    mode='min'         # 监控指标的模式，这里是最小化验证损失
)

# 设置TensorBoard日志存储的根目录
tb_logger = TensorBoardLogger(save_dir="./tensorboard/",log_graph=False,default_hp_metric=True,name=None,version=None)

# 初始化 PyTorch Lightning Trainer
max_epochs = 1000
callbacks = [checkpoint_callback] # 忽略早停法
trainer = pl.Trainer(max_epochs=max_epochs, 
                     accelerator="auto", 
                     devices="auto",
                     # callbacks=[checkpoint_callback, early_stop_callback],
                     callbacks=callbacks,
                        precision="16-mixed",  # 这一行开启了混合精度训练
                     logger=tb_logger,  # 这里指定了TensorBoard日志记录器
                    )

# 正式开始训练之前打印参数
print(f"model_params: {model_params}")

# 开始训练
last_checkpoint_path = None # 暂时不从检查点恢复
trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader,ckpt_path = last_checkpoint_path)

