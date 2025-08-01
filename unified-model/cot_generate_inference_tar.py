import os
from PIL import Image
import re
import torch
import numpy as np
from typing import Union, List
from transformers import AutoModelForCausalLM
from tar.models import MultiModalityCausalLM, VLChatProcessor
from tar.models.config import T2IConfig

T2I_config = T2IConfig()
T2I_config = T2IConfig(device='cuda' if torch.cuda.is_available() else 'cpu')

# specify the path to the model
model_path = 'csuhan/Tar-1.5B'
vl_chat_processor: VLChatProcessor = VLChatProcessor.from_pretrained(model_path, trust_remote_code=True)
tokenizer = vl_chat_processor.tokenizer

vl_gpt: MultiModalityCausalLM = AutoModelForCausalLM.from_pretrained(
    model_path, trust_remote_code=True
)

vl_gpt = vl_gpt.to(torch.bfloat16).cuda().eval()

parallel_size = 16
max_new_tokens = 1024

template = "A photo of {}. Output a richly detailed prompt: "

prompt = "a train left of a broccoli"

conversation = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": prompt}
]

input_text = vl_chat_processor.tokenizer.apply_chat_template(
            conversation,
            tokenize=False,
            add_generation_prompt=True)

input_text += f"<im_start><S{T2I_config.scale}>"

inputs = vl_chat_processor.tokenizer(input_text, return_tensors="pt")
input_ids = inputs.input_ids.to(T2I_config.device)
attention_mask = inputs.attention_mask.to(T2I_config.device)

gen_ids = vl_gpt.generate(
    input_ids,
    attention_mask=attention_mask,
    max_new_tokens=T2I_config.seq_len,
    do_sample=True,
    temperature=T2I_config.temperature,
    top_p=T2I_config.top_p,
    top_k=T2I_config.top_k)

# Process generated tokens
gen_text = vl_chat_processor.tokenizer.batch_decode(gen_ids)[0]
image = vl_chat_processor.decode_text_to_image(generated_text=gen_text)
