import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import logging
from tqdm import tqdm
from einops import rearrange
from transformers.cache_utils import Cache

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.parametrize as P
from torch.nn.utils.parametrizations import weight_norm
from transformers import LlamaModel, LlamaConfig
    
    
class LlamaMLP(nn.Module):
    def __init__(self, hidden_size, intermediate_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = F.silu

    def forward(self, x):
        down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        return down_proj
    
    
class GPT_warpper(nn.Module):
    def __init__(
        self, 
        gpt_config, 
        num_audio_tokens,
        num_text_tokens,
        num_vq=4,
        **kwargs,
        ):
        super().__init__()

        self.logger = logging.getLogger(__name__)
        self.gpt = self.build_model(gpt_config)
        self.model_dim = self.gpt.config.hidden_size 

        # 保存音频嵌入层的数量 
        self.num_vq = num_vq
        # 创建一个包含 num_vq 个嵌入层的模块列表，每个嵌入层将**音频**令牌映射到模型的隐藏层维度。
        self.emb_code = nn.ModuleList([nn.Embedding(num_audio_tokens, self.model_dim) for i in range(self.num_vq)])
        # 创建一个嵌入层，将**文本**令牌映射到模型的隐藏层维度。
        self.emb_text = nn.Embedding(num_text_tokens, self.model_dim)
        
        # 创建一个线性层，用于将模型的输出映射回文本令牌空间，并应用权重归一化。
        self.head_text = weight_norm(nn.Linear(self.model_dim, num_text_tokens, bias=False), name='weight')
        # 创建一个包含 num_vq 个线性层的模块列表，每个线性层用于将模型的输出映射回音频令牌空间，并应用权重归一化。
        self.head_code = nn.ModuleList([weight_norm(nn.Linear(self.model_dim, num_audio_tokens, bias=False), name='weight') for i in range(self.num_vq)])

    def build_model(self, config):
        
        configuration = LlamaConfig(**config)
        model = LlamaModel(configuration)
        del model.embed_tokens
        
        return model
    
    def get_emb(self, input_ids, text_mask, **kwargs):

        emb_text = self.emb_text(input_ids[text_mask][:, 0])
        
        emb_code = [self.emb_code[i](input_ids[~text_mask][:, i]) for i in range(self.num_vq)]
        emb_code = torch.stack(emb_code, 2).sum(2)
        
        emb = torch.zeros((input_ids.shape[:-1])+(emb_text.shape[-1],), device=emb_text.device, dtype=emb_text.dtype)
        emb[text_mask] = emb_text
        emb[~text_mask] = emb_code.to(emb.dtype)
        
        return emb
    
    def prepare_inputs_for_generation(
        self, input_ids, # 输入序列的 ID。
        past_key_values=None, # 过去的键值对，用于缓存先前计算的注意力权重。
        attention_mask=None, # 注意力掩码，用于指示模型应该关注哪些位置。
        inputs_embeds=None, # 输入的嵌入表示。
        cache_position=None, #缓存位置。
        **kwargs
    ):
        """这个函数的主要目的是为生成过程准备输入数据。它处理输入序列的 ID、过去的键值对、注意力掩码、输入嵌入表示和缓存位置等信息，以确保模型在生成过程中能够高效地处理输入数据
        """
        # With static cache, the `past_key_values` is None
        # TODO joao: standardize interface for the different Cache classes and remove of this if
        has_static_cache = False
        # 在某些情况下，past_key_values 可能为 None，这通常发生在生成过程的第一步。
        # 在这种情况下，模型需要从自注意力层中获取缓存的键值对，以便在后续步骤中使用。
        # 通过缓存先前计算的注意力键值对，模型可以避免重复计算，从而显著提高生成效率。
        if past_key_values is None:
            # 尝试从模型的自注意力层中获取缓存的键值对
            # 解释：在 GPT 这样的自回归模型，每一步生成新的 token 时，模型需要重新计算所有先前生成的 token 的注意力权重。
            # 这会导致计算量随着生成长度的增加而线性增长。
            # 为了优化这一过程，模型会缓存先前计算的注意力键值对，这样在生成新的 token 时，只需要计算新增的部分，而不必重复计算所有先前的 token。
            # 
            past_key_values = getattr(self.gpt.layers[0].self_attn, "past_key_value", None)
            has_static_cache = past_key_values is not None

        past_length = 0
        if past_key_values is not None:
            if isinstance(past_key_values, Cache):
                # 如果cache_position为空则调用 past_key_values.get_seq_length() 获取缓存的序列长度。
                past_length = cache_position[0] if cache_position is not None else past_key_values.get_seq_length()
                # 确定缓存的最大长度，以便在后续计算 cache_length 时使用
                max_cache_length = (
                    torch.tensor(past_key_values.get_max_length(), device=input_ids.device)
                    if past_key_values.get_max_length() is not None
                    else None
                )
                # 
                cache_length = past_length if max_cache_length is None else torch.min(max_cache_length, past_length)
            # TODO joao: remove this `else` after `generate` prioritizes `Cache` objects
            else:
                # 如果 past_key_values 不是 Cache 对象，则假设它是一个包含键值对的列表
                # 通过 past_key_values[0][0].shape[2] 获取缓存的序列长度，并将其赋值给 past_length 和 cache_length
                cache_length = past_length = past_key_values[0][0].shape[2]
                max_cache_length = None

            # Keep only the unprocessed tokens:
            # 1 - If the length of the attention_mask exceeds the length of input_ids, then we are in a setting where
            # some of the inputs are exclusively passed as part of the cache (e.g. when passing input_embeds as
            # input)
            # 如果 attention_mask 不为空且其长度大于 input_ids 的长度，则表示有些输入是作为缓存的一部分传递的（例如，当传递 input_embeds 作为输入时）
            if attention_mask is not None and attention_mask.shape[1] > input_ids.shape[1]:
                # 在这种情况下，更新 input_ids 以仅保留未处理的 token。具体来说，保留从 input_ids 的末尾开始，长度为 attention_mask.shape[1] - past_length 的部分
                input_ids = input_ids[:, -(attention_mask.shape[1] - past_length) :]
            # 2 - If the past_length is smaller than input_ids', then input_ids holds all input tokens. We can discard
            # input_ids based on the past_length.
            # 如果 past_length 小于 input_ids 的长度，则表示 input_ids 包含所有输入 token
            elif past_length < input_ids.shape[1]:
                # 在这种情况下，更新 input_ids 以仅保留未处理的 token。具体来说，保留从 input_ids 的 past_length 位置开始的部分
                input_ids = input_ids[:, past_length:]
            # 3 - Otherwise (past_length >= input_ids.shape[1]), let's assume input_ids only has unprocessed tokens.

            # 如果 past_length 大于或等于 input_ids 的长度，则假设 input_ids 仅包含未处理的 token。在这种情况下，不需要对 input_ids 进行任何修改。
            # If we are about to go beyond the maximum cache length, we need to crop the input attention mask.
            # 如果 max_cache_length 不为空，且 attention_mask 不为空，并且 cache_length 加上 input_ids 的长度超过 max_cache_length，则需要裁剪 attention_mask
            if (
                max_cache_length is not None
                and attention_mask is not None
                and cache_length + input_ids.shape[1] > max_cache_length
            ):
                # 更新 attention_mask 以仅保留最后 max_cache_length 长度的部分。
                # 这确保了 attention_mask 的长度不会超过最大缓存长度
                attention_mask = attention_mask[:, -max_cache_length:]

        position_ids = kwargs.get("position_ids", None)
        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -input_ids.shape[1] :]

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            # 如果 inputs_embeds 不为空且 past_key_values 为空，则表示这是生成过程的第一步，此时只使用 inputs_embeds
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            # The `contiguous()` here is necessary to have a static stride during decoding. torchdynamo otherwise
            # recompiles graphs as the stride of the inputs is a guard. Ref: https://github.com/huggingface/transformers/pull/29114
            # TODO: use `next_tokens` directly instead.
            # 否则，使用 input_ids 并调用 contiguous() 方法确保其在解码过程中具有静态步幅。这是为了避免 torchdynamo 重新编译图形
            model_inputs = {"input_ids": input_ids.contiguous()}

        # 确定 input_length，即输入序列的长度。
        # 在某些情况下，模型可能会使用 position_ids 来明确指定每个 token 的位置
        # 在其他情况下，模型可能只使用 input_ids，而不需要显式的 position_ids
        input_length = position_ids.shape[-1] if position_ids is not None else input_ids.shape[-1]
        if cache_position is None:
            cache_position = torch.arange(past_length, past_length + input_length, device=input_ids.device)
        else:
            cache_position = cache_position[-input_length:]

        if has_static_cache:
            past_key_values = None

        model_inputs.update(
            {
                "position_ids": position_ids,
                "cache_position": cache_position,
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
            }
        )
        return model_inputs
    
    def generate(
        self, 
        emb,    # 初始的嵌入表示
        inputs_ids,     # 输入的 token IDs
        temperature,    #  控制生成的随机性
        eos_token,      # 结束 token 的 ID
        attention_mask = None, # 注意力掩码
        max_new_token = 2048,  # 最大生成 token 数量
        min_new_token = 0,     # 最小生成 token 数量
        LogitsWarpers = [],    #  用于修改 logits 的函数列表
        LogitsProcessors = [], # 用于处理 logits 的函数列表
        infer_text=False,      # 是否生成文本
        return_attn=False,     # 是否返回注意力权重
        return_hidden=False,   # 是否返回隐藏状态
    ):
        
        with torch.no_grad():   
        
            attentions = [] # 注意力权重
            hiddens = []   # 隐藏状态
            
            # 用于记录输入序列的初始长度。在生成新 token 时，这个值可以帮助确定新生成的 token 应该添加到输入序列的哪个位置。
            # inputs_ids 的第二个维度的大小，即输入序列的长度。inputs_ids 是一个形状为 (batch_size, sequence_length, num_vq) 的张量，因此 start_idx 保存了输入序列的位置
            start_idx = inputs_ids.shape[1]
            # 这行代码创建了一个形状为 (batch_size,) 的全零张量，数据类型为 torch.long，并且与 inputs_ids 位于同一个设备上（例如 CPU 或 GPU）
            # end_idx 用于记录每个样本的结束位置。初始化为全零意味着在生成开始时，所有样本的结束位置都未确定
            end_idx = torch.zeros(inputs_ids.shape[0], device=inputs_ids.device, dtype=torch.long)
            #  创建一个形状为(batch_size,) 的布尔张量，其中batch_size 就是输入序列的批次大小(说白了就是多少句话)。初始化为全False，表示所有样本尚未完成生成
            finish = torch.zeros(inputs_ids.shape[0], device=inputs_ids.device).bool()
            
            # 这一步通过在 temperature 张量的第一个维度添加一个新的维度，将 temperature 从形状 (n,) 变为 (1, n)
            temperature = temperature[None]
            # 这一步将 temperature 张量扩展到形状 (batch_size, n)，其中 batch_size 是 inputs_ids 的第一个维度的大小。-1 表示保持该维度的大小不变
            temperature = temperature.expand(inputs_ids.shape[0], -1)
            # "b n -> (b n) 1"：这一步将 temperature 张量从形状 (batch_size, n) 重排为形状 ((batch_size * n), 1)。具体来说，它将第一个维度 batch_size 和第二个维度 n 合并成一个维度，并在最后添加一个新的维度。
            temperature = rearrange(temperature, "b n -> (b n) 1")

            # 用于存储注意力掩码
            attention_mask_cache = torch.ones(  # 创建一个全为 1 的张量
                (inputs_ids.shape[0], # 表示批次大小（batch size）。
                 inputs_ids.shape[1]+max_new_token,),  # 表示输入序列的长度加上最大新生成的 token 数，以便容纳输入序列和生成的token
                dtype=torch.bool, 
                device=inputs_ids.device)
            if attention_mask is not None:
                # 在生成模型中，注意力掩码用于指示模型在处理输入序列时应该关注哪些位置。
                # attention_mask 通常是一个布尔型张量，其中值为 1 的位置表示模型应该关注的 token，值为 0 的位置表示模型应该忽略的 token
                # attention_mask.shape[1] 表示 attention_mask 的列数，即输入序列的长度
                # 通过这一步操作，attention_mask_cache 的前 attention_mask.shape[1] 列被初始化为 attention_mask 的值。
                # 这确保了在生成新 token 之前，attention_mask_cache 已经包含了**输入序列**的注意力掩码信息
                # 这样就可以确保我们预先输入文本的为有效序列，避免引入无效的注意力信息
                attention_mask_cache[:, :attention_mask.shape[1]] = attention_mask
            
            for i in tqdm(range(max_new_token)):
                # 在生成过程中，每次迭代都会调用这段代码，以确保模型在每一步都能正确处理输入数据。
                # 通过传递 past_key_values 和更新的注意力掩码，模型可以高效地生成新的 token，而不必重复计算所有先前的 token 的注意力权重。
                model_input = self.prepare_inputs_for_generation(
                    inputs_ids,     # 这是输入的 token IDs。它表示当前生成序列的 token ID 列表。
                    outputs.past_key_values if i!=0 else None, # 如果不是第一次迭代则获得在前一次生成过程中计算并缓存的注意力键值对
                    attention_mask_cache[:, :inputs_ids.shape[1]], # 从 attention_mask_cache 中截取前 inputs_ids.shape[1] 列，确保注意力掩码的长度与当前输入序列的长度一致
                    use_cache=True)
            
                if i == 0:
                    # 在第一次迭代时，使用初始嵌入 `emb`，
                    model_input['inputs_embeds'] = emb
                else:
                    # 之后根据 `infer_text` 决定使用文本嵌入还是音频嵌入。
                    if infer_text:
                        model_input['inputs_embeds'] = self.emb_text(model_input['input_ids'][:,:,0])
                    else:
                        # code_emb = [self.emb_code[i](model_input['input_ids'][:,:,i]) for i in range(self.num_vq)]
                        # model_input['inputs_embeds'] = torch.stack(code_emb, 3).sum(3)
                        
                        code_emb = []   # 用于存储每个向量量化层的嵌入表示
                        # 通过遍历多个向量量化层（VQ 层），为每个层的 token IDs 生成嵌入表示。这种方法可以捕捉输入序列中不同层次的信息
                        for i in range(self.num_vq):
                            # model_input['input_ids'] 的形状为 (batch_size, sequence_length, num_vq)
                            input_ids_i = model_input['input_ids'][:, :, i] # 获取第 i 个向量量化层的 token IDs，形状为 (batch_size, sequence_length)
                            
                            # self.emb_code[i] 是第 i 个嵌入层，它将 input_ids_i 转换为嵌入表示
                            emb_i = self.emb_code[i](input_ids_i)   # emb_i 的形状为 (batch_size, sequence_length, embedding_dim)
                            # 将每个向量量化层的嵌入表示存储在 code_emb 列表中
                            code_emb.append(emb_i)

                        # 堆叠后的形状为 (batch_size, sequence_length, embedding_dim, num_vq)
                        stacked_emb = torch.stack(code_emb, 3)
                        
                        # 对堆叠后的张量在第 3 维度上进行求和
                        final_emb = stacked_emb.sum(3)
                        
                        # 将最终的嵌入表示添加到 model_input 字典中
                        model_input['inputs_embeds'] = final_emb
                        
                        # 解释：将多个嵌入表示沿新的维度堆叠起来，并在该维度上求和，生成一个综合的嵌入表示。这种方法可以将不同层次的信息融合在一起，提供更丰富的输入表示
                        # 在一些高级应用中，可能会使用多层嵌入表示。例如：
                        # 在多头注意力机制中，不同的头可以看作是不同的层，每个头捕捉输入序列的不同方面的信息
                        # 在多任务学习中，不同的任务可能需要不同的嵌入表示，通过多层嵌入表示可以为每个任务生成特定的嵌入表示
                        # 在一些生成模型中，向量量化技术用于将连续的嵌入表示离散化，以便更好地捕捉输入序列的结构信息
                
                # 由于我们已经生成了嵌入表示并将其存储在 model_input['inputs_embeds'] 中，因此不再需要 input_ids
                model_input['input_ids'] = None
                # 调用 GPT 模型的 forward 方法进行前向传播。
                outputs = self.gpt.forward(**model_input, output_attentions=return_attn)
                # 将注意力权重存储到 attentions 列表中，以便后续操作
                attentions.append(outputs.attentions)
                # 获取模型的输出隐藏状态，形状为 (batch_size, sequence_length, hidden_size)
                # 这些隐藏状态包含了模型对输入序列的表示
                # 存储隐藏状态可以用于下游任务，如分类、生成等
                hidden_states = outputs[0] # 🐻
                if return_hidden:
                    hiddens.append(hidden_states[:, -1])

                with P.cached():
                    if infer_text:  
                        logits = self.head_text(hidden_states) 
                    else:
                        # 使用 self.head_code 的多个头（self.num_vq）处理隐藏状态，并将结果堆叠在一起，生成代码的 logits
                        logits = torch.stack([self.head_code[i](hidden_states) for i in range(self.num_vq)], 3)
        
                # 获取每个序列的最后一个 token 的 logits，形状为 (batch_size, num_classes)
                # 将 logits 转换为浮点数，以确保后续计算的精度
                logits = logits[:, -1].float()

                if not infer_text:
                    # logits 重新排列为形状 (batch_size * num_vq, num_classes)
                    logits = rearrange(logits, "b c n -> (b n) c")
                    #  重新排列为形状 (batch_size * num_vq, num_classes)
                    logits_token = rearrange(inputs_ids[:, start_idx:], "b c n -> (b n) c")
                else:
                    logits_token = inputs_ids[:, start_idx:, 0]
                
                # 将 logits 除以温度参数 temperature，以控制模型输出的平滑度
                # 较高的温度会使分布更加平滑，较低的温度会使分布更加尖锐。
                logits = logits / temperature
                
                # 重复惩罚器，避免重复输出同一个字词
                for logitsProcessors in LogitsProcessors:
                    logits = logitsProcessors(logits_token, logits)
                
                # # top-k,top-p,temperate 等多个参数应用
                # 1. top-k：只保留概率最高的 top-k 个候选词，其他候选词的概率被设置为 0。这样可以限制模型生成的结果只包含概率最高的 top-k 个候选词。
                # 2. top-p：只保留概率之和不大于 top-p 的候选词，其他候选词的概率被设置为 0。这样可以限制模型生成的结果只包含概率之和不大于 top-p 的候选词。
                # 3. temperate：控制模型输出的平滑度，temperature 越大输出越多样性，temperature 越小输出越相似。
                for logitsWarpers in LogitsWarpers:
                    logits = logitsWarpers(logits_token, logits)
                
                # 如果当前生成的 token 数量小于 min_new_token，则将 eos_token 的 logits 设置为负无穷大。
                # 这样可以防止模型在生成足够的新 token 之前提前结束。
                if i < min_new_token:
                    logits[:, eos_token] = -torch.inf
                
                # 应用 softmax 函数，得到每个 token 的概率分布
                scores = F.softmax(logits, dim=-1)
            
                # 使用多项式分布从分数中采样下一个 token 的索引
                # num_samples=1 表示每次采样一个 token
                idx_next = torch.multinomial(scores, num_samples=1)
                
                if not infer_text:
                    # idx_next 重新排列为形状 (batch_size, num_vq)
                    idx_next = rearrange(idx_next, "(b n) 1 -> b n", n=self.num_vq)
                    # 标记更新为是否有任何序列生成了 eos_token
                    finish = finish | (idx_next == eos_token).any(1)
                    
                    # inputs_ids 是当前生成的序列张量，形状为 (batch_size, sequence_length),它包含了模型已经生成的所有 token 的索引
                    # idx_next 是从概率分布中采样得到的下一个 token 的索引，形状为 (batch_size, 1),它表示每个序列在当前步骤生成的下一个 token
                    # unsqueeze 是 PyTorch 中的一个函数，用于在指定位置插入一个新的维度
                    # idx_next.unsqueeze(1) 将 idx_next 的形状从 (batch_size, 1) 变为 (batch_size, 1, 1)
                    # torch.cat 将这两个张量进行拼接，得到新的张量形状为 (batch_size, sequence_length + 1),表示在当前生成的序列末尾添加了新生成的 token。
                    inputs_ids = torch.cat([inputs_ids, idx_next.unsqueeze(1)], 1)
                else:
                    finish = finish | (idx_next == eos_token).any(1)
                    inputs_ids = torch.cat([inputs_ids, idx_next.unsqueeze(-1).expand(-1, -1, self.num_vq)], 1)

                # 更新 `end_idx` 记录每个样本的结束位置
                end_idx = end_idx + (~finish).int()

                # 如果所有样本都完成生成，退出循环
                if finish.all():
                    break
            
            #  根据 `end_idx` 截取生成的 `inputs_ids`
            # inputs_ids = [inputs_ids[idx, start_idx: start_idx+i] for idx, i in enumerate(end_idx.int())]
            # inputs_ids = [i[:, 0] for i in inputs_ids] if infer_text else inputs_ids
            # 在推理过程中，这段代码的作用是根据每个序列的结束位置对输入序列进行切片，
            # 以便模型能够处理变长的输入序列。这在生成任务中非常重要，因为生成的序列长度可能会有所不同
            inputs_ids_list = []
            for idx, i in enumerate(end_idx.int()):
                sliced_input = inputs_ids[idx, start_idx: start_idx+i]
                inputs_ids_list.append(sliced_input)
            inputs_ids = inputs_ids_list
            # 展开后的代码
            if infer_text:
                inputs_ids_list = []
                for i in inputs_ids:
                    sliced_input = i[:, 0]
                    inputs_ids_list.append(sliced_input)
                inputs_ids = inputs_ids_list
            
            # 如果 `return_hidden` 为真，返回隐藏状态
            if return_hidden:
                # 将 hiddens 列表中的张量沿维度 1 进行堆叠，生成一个新的张量。
                # 使用列表推导式对 hiddens 进行切片操作，类似于处理 inputs_ids 的第一步
                hiddens = torch.stack(hiddens, 1)
                # hiddens = [hiddens[idx, :i] for idx, i in enumerate(end_idx.int())]
                hiddens_list = []
                for idx, i in enumerate(end_idx.int()):
                    sliced_hidden = hiddens[idx, :i]
                    hiddens_list.append(sliced_hidden)
                hiddens = hiddens_list
                    
            if not finish.all():
                self.logger.warn(f'Incomplete result. hit max_new_token: {max_new_token}')    
                   
            return {
                'ids': inputs_ids, 
                'attentions': attentions,
                'hiddens':hiddens,
            }