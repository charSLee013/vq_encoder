from concurrent.futures import ThreadPoolExecutor
import os
import torch
import torchaudio
import yaml
import argparse
import logging
import subprocess
from torch.utils.data import Dataset

import os
import subprocess
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed


class AudioDataset(Dataset):
    def __init__(self, audio_files, sample_rate=44100, transform=None):
        self.audio_files = audio_files
        self.sample_rate = sample_rate
        self.transform = transform

    def __len__(self):
        return len(self.audio_files)

    def __getitem__(self, idx):
        audio_path = self.audio_files[idx]
        audio, sr = torchaudio.load(audio_path)
        if sr != self.sample_rate:
            audio = torchaudio.transforms.Resample(sr, self.sample_rate)(audio)
        if self.transform:
            audio = self.transform(audio)
        return audio, audio.shape[1]  # Return audio and its length

def collate_fn(batch):
    audios, lengths = zip(*batch)
    max_length = max(lengths)
    padded_audios = torch.zeros(len(audios), 1, max_length)
    for i, audio in enumerate(audios):
        padded_audios[i, 0, :audio.shape[1]] = audio
    return padded_audios, torch.tensor(lengths)


def load_config(config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config


def resample_audio(audio_path, target_sample_rate):
    temp_path = audio_path + ".tmp.wav"
    subprocess.run(['ffmpeg', '-i', audio_path, '-ar',
                   str(target_sample_rate), temp_path], check=True)
    os.replace(temp_path, audio_path)


def get_audio_files(data_path, target_sample_rate, resample=False, batch_size=None):
    audio_files = []
    labels = []
    with ThreadPoolExecutor(max_workers=batch_size) as executor:
        futures = []
        for root, _, files in os.walk(data_path):
            for file in files:
                if file.endswith('.wav'):
                    audio_path = os.path.join(root, file)
                    label_path = os.path.splitext(audio_path)[0] + '.lab'
                    if os.path.exists(label_path):
                        futures.append(executor.submit(
                            process_audio_file, audio_path, label_path, target_sample_rate, resample))

        for future in as_completed(futures):
            result = future.result()
            if result:
                audio_files.append(result[0])
                labels.append(result[1])

    return audio_files, labels


def process_audio_file(audio_path, label_path, target_sample_rate, resample):
    # 获取音频文件的采样率
    result = subprocess.run(
        ['ffmpeg', '-i', audio_path, '-ar',
            str(target_sample_rate), '-f', 'null', '-'],
        stderr=subprocess.PIPE,
        stdout=None,
        universal_newlines=True
    )

    # 检查音频文件的时长
    duration_line = [line for line in result.stderr.split(
        '\n') if 'Duration' in line]
    if duration_line:
        duration_str = duration_line[0].split(',')[0].split('Duration: ')[1]
        h, m, s = map(float, duration_str.split(':'))
        duration = h * 3600 + m * 60 + s
        if 2 <= duration <= 30:
            # 检查采样率是否匹配
            sample_rate_line = [line for line in result.stderr.split(
                '\n') if 'Stream' in line and 'Audio' in line]
            if sample_rate_line:
                current_sample_rate = int(sample_rate_line[0].split(',')[
                                          1].split('Hz')[0].strip())
                if current_sample_rate == target_sample_rate:
                    return audio_path, label_path
                else:
                    if resample:
                        resample_audio(audio_path, target_sample_rate)
                        return audio_path, label_path
                    else:
                        logging.warning(
                            f"File {audio_path} has a different sample rate: {current_sample_rate} Hz.")
                        return audio_path, label_path
    return None
