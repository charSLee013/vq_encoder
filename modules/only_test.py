import torch
import librosa
import soundfile as sf
from vits_decoder import VITSDecoder

# 加载模型
model_path = "/private/tmp/vits_decoder_v1.1.ckpt"
model = VITSDecoder.load_from_checkpoint(model_path)
model.eval()  # 设置为评估模式

# 加载参考音频
reference_audio_path = "/private/tmp/芙宁娜/vo_dialog_DYJEQ001_furina_01.wav"
reference_audio, sr = librosa.load(reference_audio_path)

# 转换为梅尔频谱图
mel_transform = model.mel_transform
reference_audio_tensor = torch.tensor(reference_audio).unsqueeze(0).unsqueeze(0)
reference_mel = mel_transform(reference_audio_tensor)

# 获取音色特征向量
reference_mel_lengths = torch.tensor([reference_mel.shape[-1]])
ge = model.encode_ref(reference_mel, reference_mel_lengths)

# 下载模型
from modelscope import snapshot_download
model_dir = snapshot_download('mirror013/ChatTTS')

# 使用TTS系统生成目标文本的梅尔频谱图
text = "你好TTS"
import ChatTTS
chat = ChatTTS.Chat()
chat.load_models(
    source="local",
    local_path=model_dir,
    device='cpu',
    compile=False,
)
mel = chat.infer(
    text='你好，我是Chat T T S。',
    use_decoder=True,
    return_mel_spec=True,
)[0]
mel_tensor = torch.tensor(mel).unsqueeze(0)
mel_lengths = torch.tensor([mel_tensor.shape[-1]])

# 生成固定音色的语音
with torch.no_grad():
    output_audio = model.infer_posterior(mel_tensor, mel_lengths, ge=ge)

# 保存生成的音频
sf.write("output.wav", output_audio.squeeze().cpu().numpy(), sr)
