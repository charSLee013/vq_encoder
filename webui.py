import json
import os
import argparse
import sys

import torch
import gradio as gr
import numpy as np
from modelscope import snapshot_download

import ChatTTS
import modelscope
from modelscope.hub.api import HubApi
from modules.dvae import DVAEDecoder

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu' # dont support mps Q_Q
if sys.platform == "darwin":
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

SEED = 1397
SDK_TOKEN = os.environ.get('MODELSCOPE_TOKEN',None)
if SDK_TOKEN is None:
    raise Exception("实验阶段，暂不公开.")

chat = None
special_decoder:DVAEDecoder = None


def filter_punctuation(text):
    allowed_punctuations = {".", ",", "!", "?", "，", "。", "！", "？"," "}
    new_text = ""
    for char in text:
        if char.isalnum() or char in allowed_punctuations:
            new_text += char
    return new_text

def preprocess_features(features_tensor):
    """
    Preprocess the features tensor to prepare it for decoding.
    """
    features_tensor = features_tensor.transpose(1, 2)
    temp = torch.chunk(features_tensor, 2, dim=1)
    temp = torch.stack(temp, -1)
    vq_feats = temp.reshape(*temp.shape[:2], -1)
    return vq_feats

@torch.no_grad()
def generate_audio(text, refine_text_flag):
    # 清除文本里面不符合标准的标点符号
    text = filter_punctuation(text)
    
    if len(text) >= 300:
        raise Exception("文本长度超过300字，请重新输入")
    
    # load from local file if exists
    if os.path.exists('spk_emb.npy'):
        spk_emb = torch.load('spk_emb.npy',map_location='cpu')
        print("use local speaker embedding")
    else:
        spk_emb = chat.sample_random_speaker()
        print("use random speaker embedding")
    torch.manual_seed(SEED)
    params_infer_code = {
        'spk_emb': spk_emb, 
        'temperature': 0.3,
        'top_P': 0.7,
        'top_K': 20,
        }
    params_refine_text = {'prompt': '[oral_6][laugh_2][break_6]'}
    
    torch.manual_seed(SEED)
    if refine_text_flag:
        text = chat.infer(text, 
                          skip_refine_text=False,
                          refine_text_only=True,
                          params_refine_text=params_refine_text,
                          params_infer_code=params_infer_code
                          )
    
    torch.manual_seed(SEED)
    result = chat.infer_debug(text, 
                     params_infer_code=params_infer_code
                     )
    # 对中间层进行特定解码
    hidden = result['hiddens'][0]
    hidden = hidden.to(DEVICE)
    # 转置
    hidden = preprocess_features(hidden[None])
    mel_special_build = special_decoder(hidden)
    # 再通过vocoas 解码
    transit_audio = chat.pretrain_models['vocos'].decode(mel_special_build)
    
    audio_data = np.array(result['wav'][0]).flatten()
    transit_audio_data = transit_audio.numpy().flatten()
    sample_rate = 24000
    text_data = text[0] if isinstance(text, list) else text

    return [(sample_rate, audio_data), (sample_rate, transit_audio_data), text_data]

def main():

    with gr.Blocks() as demo:
        gr.Markdown("# ChatTTS Webui - 三月七 v1.0")
        gr.Markdown("## 说明")
        gr.Markdown("该页面仅作为展示在不外置其他工具的情况下，实现音色固定.")
        gr.Markdown("ChatTTS Model: [2noise/ChatTTS](https://github.com/2noise/ChatTTS)")

        default_text = "四川美食确实以辣闻名，但也有不辣的选择。比如甜水面、赖汤圆、蛋烘糕、叶儿粑等，这些小吃口味温和，甜而不腻，也很受欢迎。"        
        text_input = gr.Textbox(label="Input Text", lines=4, placeholder="请输入文字（约200字内），暂时不处理数字和标点符号读音问题", value=default_text)

        with gr.Row():
            refine_text_checkbox = gr.Checkbox(label="参考文本处理（建议关闭）", value=False)

        generate_button = gr.Button("Generate")
        
        text_output = gr.Textbox(label="Output Text", interactive=False)
        audio_output = gr.Audio(label="原声 Audio")
        transit_audio_output = gr.Audio(label="换声 Audio")
        
        generate_button.click(generate_audio, 
                              inputs=[text_input, refine_text_checkbox], 
                              outputs=[audio_output, transit_audio_output,text_output])

        gr.Examples(
            examples=[
                ["四川美食确实以辣闻名，但也有不辣的选择。比如甜水面、赖汤圆、蛋烘糕、叶儿粑等，这些小吃口味温和，甜而不腻，也很受欢迎。", True],
                ["What is [uv_break]your favorite english food?[laugh][lbreak]", False],
                ["你说的对，但是《原神》是由米哈游自主研发的一款全新开放世界冒险游戏。游戏发生在一个被称作「提瓦特」的幻想世界，在这里，被神选中的人将被授予「神之眼」，导引元素之力。你将扮演一位名为「旅行者」的神秘角色，在自由的旅行中邂逅性格各异、能力独特的同伴们，和他们一起击败强敌，找回失散的亲人——同时，逐步发掘「原神」的真相", True],
            ],
            inputs=[text_input, refine_text_checkbox],
        )
    
    parser = argparse.ArgumentParser(description='ChatTTS demo Launch')
    parser.add_argument('--server_name', type=str, default='127.0.0.1', help='Server name')
    parser.add_argument('--server_port', type=int, default=38888, help='Server port')
    args = parser.parse_args()

    print("loading ChatTTS model and special decoder model ...")
    global chat,special_decoder
    model_dir = snapshot_download('mirror013/ChatTTS')
    chat = ChatTTS.Chat()

    chat.load_models(
        source="local",
        local_path=model_dir,
        compile=False,
        device=DEVICE,
    )
    #验证SDK token
    api = HubApi()
    api.login(SDK_TOKEN)

    checkpoin_dir = snapshot_download('mirror013/March7-decoder')
    with open(os.path.join(checkpoin_dir, 'model-55-0.0000_config.json')) as f:
        config = json.loads(f.read())
    # 加载特制解码器
    special_decoder = DVAEDecoder(**config)
    checkpoint = torch.load(os.path.join(checkpoin_dir, 'model-55-0.0000.ckpt'), map_location='cpu',mmap=True)
    
    state_dict = checkpoint['state_dict']
    
    # 处理state_dict的key，去除前缀'model.'，如果存在的话
    new_state_dict = {k.replace("model.", ""): v for k, v in state_dict.items()}
    
    # 加载修正后的state_dict
    special_decoder.load_state_dict(new_state_dict)

    demo.launch(server_name=args.server_name, server_port=args.server_port,show_error=True,debug=True)
if __name__ == '__main__':
    main()