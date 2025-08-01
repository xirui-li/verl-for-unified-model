# tar_chat_processor.py

from dataclasses import dataclass
from typing import List, Union, Dict
from PIL import Image
import torch
import re
from transformers import PreTrainedTokenizer, ProcessorMixin, AutoTokenizer
from huggingface_hub import hf_hub_download

from .image_processing_vlm import VLMImageProcessor
from ..tok.mm_autoencoder import MMAutoEncoder
from .config import T2IConfig


@dataclass
class TarProcessorOutput:
    input_ids: torch.Tensor
    pixel_values: Union[torch.Tensor, None]
    prompt_text: str

class VLChatProcessor(ProcessorMixin):
    image_processor_class = "TarImageProcessor"
    tokenizer_class = ("LlamaTokenizer", "LlamaTokenizerFast")

    attributes = ["image_processor", "tokenizer"]

    def __init__(
        self,
        image_processor: VLMImageProcessor,
        tokenizer: PreTrainedTokenizer,
        sft_format: str = "tar",
        system_prompt: str = "You are a helpful vision-language assistant.",
    ):
        self.image_processor = image_processor
        self.tokenizer = tokenizer
        self.sft_format = sft_format
        self.system_prompt = system_prompt
        self.config = T2IConfig()

    @classmethod
    def from_pretrained(cls, model_path, **kwargs):
        tokenizer = AutoTokenizer.from_pretrained(model_path, **kwargs)
        
        # Load your visual tokenizer config from model repo or path # hard coded for now
        ar_path = hf_hub_download("csuhan/TA-Tok", "ar_dtok_lp_1024px.pth")
        encoder_path = hf_hub_download("csuhan/TA-Tok", "ta_tok.pth")
        decoder_path = hf_hub_download("peizesun/llamagen_t2i", "vq_ds16_t2i.pt")

        T2I_config = T2IConfig(
            model_path=model_path,
            ar_path=ar_path,
            encoder_path=encoder_path,
            decoder_path=decoder_path,
        )
                # Initialize visual tokenizer
        config = dict(
            ar_path=T2I_config.ar_path,
            encoder_path=T2I_config.encoder_path,
            decoder_path=T2I_config.decoder_path,
            encoder_args={'input_type': 'rec'},
            decoder_args={},
        )

        image_processor = MMAutoEncoder(**config).eval().to(dtype=T2I_config.dtype, device=T2I_config.device) # can modify device here
        image_processor.ar_model.cls_token_num = T2I_config.seq_len
        image_processor.encoder.pool_scale = T2I_config.scale + 1

        return cls(tokenizer=tokenizer, image_processor=image_processor)

    def process_one(
        self,
        prompt: str = None,
        conversations: List[Dict[str, str]] = None,
        images: List[Image.Image] = None,
    ) -> TarProcessorOutput:
        """Processes a single input for TAR."""
        assert prompt or conversations, "Need either `prompt` or `conversations`"

        if prompt is None:
            prompt = self.apply_sft_template_for_multi_turn_prompts(conversations)

        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").squeeze(0)

        pixel_values = None
        if images is not None:
            pixel_values = self.image_processor(images, return_tensors="pt")["pixel_values"]

        return TarProcessorOutput(
            input_ids=input_ids,
            pixel_values=pixel_values,
            prompt_text=prompt,
        )

    def __call__(
        self,
        *,
        prompt: str = None,
        conversations: List[Dict[str, str]] = None,
        images: List[Image.Image] = None,
    ):
        return self.process_one(prompt=prompt, conversations=conversations, images=images)
    
    def decode_text_to_image(self, generated_text: str) -> Image.Image:
        """
        Convert generated text with <Ixxx> tokens to image via visual tokenizer.
        """
        gen_code = [int(x) for x in re.findall(r'<I(\d+)>', generated_text)]
        gen_code = gen_code[:self.config.seq_len] + [0] * max(0, self.config.seq_len - len(gen_code))
        gen_code = torch.tensor(gen_code).unsqueeze(0).to(self.config.device)

        image_tensor = self.image_processor.decode_from_encoder_indices(
            gen_code,
            {'cfg_scale': self.config.cfg_scale}
        )

        image = Image.fromarray(image_tensor[0].cpu().numpy())
        return image

