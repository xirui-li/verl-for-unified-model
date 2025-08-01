import os
import PIL.Image
import torch
import numpy as np
from typing import Union, List
from transformers import AutoModelForCausalLM
from janus.models import MultiModalityCausalLM, VLChatProcessor

def get_eos_mask(response_id: torch.Tensor, eos_token: Union[int, List[int]] = 2, dtype=torch.int64):
    '''
    end of sentence token can be int or list: 1 or [1, 2]
    e.g. eos_token=1
    response_id: [0, 0, 2, 42, 3, 5, 1, 0, 0]
    eos_mask:     [1, 1, 1, 1,  1, 1, 1, 0, 0]
    '''
    if isinstance(eos_token, int):
        eos_token = [eos_token]

    eos_mask = torch.zeros_like(response_id, dtype=torch.bool)
    for token in eos_token:
        eos_mask |= response_id.eq(token)

    eos_mask = eos_mask.long()
    eos_mask = (torch.cumsum(eos_mask, dim=1) - eos_mask).bool()
    eos_mask = torch.logical_not(eos_mask).to(dtype)
    return eos_mask

# specify the path to the model
model_path = 'Franklin0/ReasonGen-R1'
vl_chat_processor: VLChatProcessor = VLChatProcessor.from_pretrained(model_path, trust_remote_code=True)
tokenizer = vl_chat_processor.tokenizer

vl_gpt: MultiModalityCausalLM = AutoModelForCausalLM.from_pretrained(
    model_path, trust_remote_code=True
)
vl_gpt = vl_gpt.to(torch.bfloat16).cuda().eval()

parallel_size = 16
max_new_tokens = 1024

output_path = "generated_samples"
os.makedirs(output_path, exist_ok=True)


template = "A photo of {}. Output a richly detailed prompt: "

prompt = "a train left of a broccoli"
conversation = [
    {
        "role": "<|User|>",
        "content": template.format(prompt),
    },
    {"role": "<|Assistant|>", "content": ""},
]

sft_format = vl_chat_processor.apply_sft_template_for_multi_turn_prompts(
    conversations=conversation,
    sft_format=vl_chat_processor.sft_format,
    system_prompt="",
)

prompt = sft_format 
input_ids = vl_chat_processor.tokenizer.encode(prompt)
input_ids = torch.LongTensor(input_ids).cuda()
attention_mask = torch.ones((len(input_ids)), dtype=torch.bool).cuda()
do_sample = True
eos_token_id = vl_chat_processor.tokenizer.eos_token_id
pad_token_id = vl_chat_processor.tokenizer.pad_token_id
image_start_token_id = vl_chat_processor.image_start_id
input_ids = input_ids.unsqueeze(0).repeat(parallel_size, 1)
attention_mask = attention_mask.unsqueeze(0).repeat(parallel_size, 1)
generation_config = {'cfg_weight': 5.0}
output = vl_gpt.text_img_generate(
    input_ids,
    attention_mask, 
    do_sample, 
    max_new_tokens, 
    eos_token_id, 
    pad_token_id, 
    image_start_token_id,
    generation_config,
)
sequence = output.sequences
attn_mask = get_eos_mask(sequence, eos_token_id)
res_len = attn_mask.sum(dim=-1)
print("res_len", res_len)
tokens = output.text_tokens
for i in range(len(tokens)):
    cot = vl_chat_processor.tokenizer.decode(tokens[i], skip_special_tokens=False)
    print()
    print()
    print(cot)
    with open(f"{output_path}/cot_{i}.txt", "w") as f:
        f.write(cot)
    print(f"Text saved to '{output_path}/cot_{i}.txt'")
    save_path = os.path.join(f'{output_path}', "img_cot_{}.png".format(i))
    PIL.Image.fromarray(output.gen_img[i]).save(save_path)
print(f"Images saved to '{output_path}' folder")