import torch
from dataclasses import dataclass

@dataclass
class T2IConfig:
    model_path: str = "csuhan/Tar-1.5B"
    # visual tokenizer config
    ar_path: str = 'ar_dtok_lp_256px.pth'
    encoder_path: str = 'ta_tok.pth'
    decoder_path: str = 'vq_ds16_t2i.pt'
    
    device: str = "cuda:0"
    dtype: torch.dtype = torch.bfloat16
    # generation parameters
    scale: int = 0  # choose from [0, 1, 2]
    seq_len: int = 729  # choose from [729, 169, 81]
    temperature: float = 1.0
    top_p: float = 0.95
    top_k: int = 1200
    cfg_scale: float = 4.0

@dataclass
class I2TConfig:
    model_path: str = "csuhan/Tar-1.5B"
    ta_tok_path: str = "ta_tok.pth"
    device: str = "cuda:0"
    max_new_tokens: int = 256

