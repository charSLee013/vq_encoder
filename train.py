import argparse
import logging
import os
import torch
from torch import nn, optim
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, random_split  # 添加这行

from dataset import AudioDataset, collate_fn, get_audio_files, load_config
from modules.vq_encoder import VQEncoder
# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def train_model(config, resample, batch_size):
    # 加载数据集
    train_data_path = config['data']['train_data_path']
    target_sample_rate = config['train']['sample_rate']
    if not os.path.exists(train_data_path):
        logger.error(f"训练数据路径不存在: {train_data_path}")
        return

    audio_files, _ = get_audio_files(train_data_path, target_sample_rate, resample, batch_size)
    if not audio_files:
        logger.error("未找到符合条件的音频文件")
        return
    logger.info(f"找到{len(audio_files)}个音频文件")

    dataset = AudioDataset(audio_files=audio_files, sample_rate=target_sample_rate)
    
    # 划分训练集和验证集
    val_split = config['data']['val_split']
    train_size = int((1 - val_split) * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    logger.info(f"划分训练集和验证集，训练集大小：{train_size}，验证集大小：{val_size}")
    
    train_loader = DataLoader(train_dataset, batch_size=config['train']['batch_size'], shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=config['train']['batch_size'], shuffle=False, collate_fn=collate_fn)

    # 找寻最佳的设备
    device = 'cpu'
    if torch.cuda.is_available():
        device = 'cuda:0'

    # 初始化模型、损失函数和优化器
    logger.info(f'Using device: {device}')
    model = VQEncoder(
        sample_rate=config['train']['sample_rate'],
        n_fft=config['train']['n_fft'],
        hop_length=config['train']['hop_length'],
        n_mels=config['train']['n_mels'],
        win_length=config['train']['win_length']
    ).to(device)
    
    criterion = nn.MSELoss()
    
    optimizer_name = config['train']['optimizer']
    if optimizer_name == "Adam":
        optimizer = optim.Adam(model.parameters(), lr=config['train']['lr'], betas=config['train']['betas'])
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")
    
    scaler = GradScaler()

    # 训练循环
    num_epochs = config['train']['epochs']
    for epoch in range(num_epochs):
        model.train()
        for audios, lengths in train_loader:
            audios = audios.to(device)
            lengths = lengths.to(device)
            
            encoded_features = model(audios, lengths, sr=target_sample_rate)
            loss = criterion(encoded_features, audios)  # Example loss computation
            
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        logger.info(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

        # 验证循环
        model.eval()
        with torch.no_grad():
            val_loss = 0
            for audios, lengths in val_loader:
                audios = audios.to(device)
                lengths = lengths.to(device)
                outputs = model(audios, lengths)
                val_loss += criterion(outputs, audios).item()
            val_loss /= len(val_loader)
            logger.info(f'Validation Loss: {val_loss:.4f}')

    # 保存模型
    torch.save(model.state_dict(), 'vq_encoder.pth')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train VQEncoder model")
    parser.add_argument('--config', type=str, required=True,default="./train_config.yaml" ,help="Path to the config file")
    parser.add_argument('--resample', action='store_true', help="Resample audio files to match the target sample rate")
    parser.add_argument('--batch_size', type=int, default=os.cpu_count(), help="Number of threads for processing audio files")
    args = parser.parse_args()

    if not os.path.exists(args.config):
        raise FileNotFoundError(f"Config file not found at {args.config}")
    config = load_config(args.config)

    train_model(config, args.resample, args.batch_size)
