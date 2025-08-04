# tar_chat_processor.py

from dataclasses import dataclass
from typing import List, Union, Dict, Optional
from PIL import Image
import torch
import re
from transformers import PreTrainedTokenizer, ProcessorMixin, AutoTokenizer
from huggingface_hub import hf_hub_download

from .image_processing_vlm import VLMImageProcessor
from ..tok.mm_autoencoder import MMAutoEncoder
from .config import T2IConfig

from dataclasses import dataclass, asdict, field

@dataclass
class TarProcessorOutput:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    prompt_text: str
    pixel_values: Optional[torch.Tensor] = field(default=None)


    def __getitem__(self, key):
        return getattr(self, key)

    def __contains__(self, key):
        return hasattr(self, key)

    def pop(self, key, default=None):
        if hasattr(self, key):
            val = getattr(self, key)
            setattr(self, key, None)
            return val
        if default is not None:
            return default
        raise KeyError(f"{key} not found in TarProcessorOutput")

    def __iter__(self):
        return iter(asdict(self))  # Allows iteration over keys

    def items(self):
        return asdict(self).items()

    def keys(self):
        return asdict(self).keys()

    def values(self):
        return asdict(self).values()
        

class VLChatProcessor(ProcessorMixin):
    image_processor_class = "TarImageProcessor"
    tokenizer_class = ("LlamaTokenizer", "LlamaTokenizerFast")

    attributes = ["image_processor", "tokenizer"]

    def __init__(
        self,
        image_processor: VLMImageProcessor,
        tokenizer: PreTrainedTokenizer,
        sft_format: str = "tar",
        system_prompt: str = "You are a helpful assistant.",
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

        config = dict(
            ar_path=T2I_config.ar_path,
            encoder_path=T2I_config.encoder_path,
            decoder_path=T2I_config.decoder_path,
            encoder_args={'input_type': 'rec'},
            decoder_args={},
        )

        # === Safe GPU allocation ===
        if torch.cuda.is_available():
            device = 'cuda'
        else:
            device = 'cpu'
            print("[Warning] CUDA not available in Ray worker. Using CPU for image processor. Expect slower rollouts.")

        try:
            image_processor = MMAutoEncoder(**config).eval().to(device=device)
        except RuntimeError as e:
            print(f"[Error] Failed to move MMAutoEncoder to {device}: {e}")
            print("[Fallback] Forcing CPU...")
            image_processor = MMAutoEncoder(**config).eval().to(device='cpu')

        # === Freeze parameters ===
        for param in image_processor.parameters():
            param.requires_grad = False

        image_processor.ar_model.cls_token_num = T2I_config.seq_len
        image_processor.encoder.pool_scale = T2I_config.scale + 1

        return cls(tokenizer=tokenizer, image_processor=image_processor)

    def process_one(
        self,
        prompt: str = None,
        conversations: List[Dict[str, str]] = None,
        images: List[Image.Image] = None,
        videos: List = None  # placeholder
    ) -> TarProcessorOutput:
        assert prompt or conversations, "Need either `prompt` or `conversations`"

        if prompt is None:
            prompt = self.apply_sft_template_for_multi_turn_prompts(conversations)

        prompt += f"<im_start><S{self.config.scale}>"

        # print(self.tokenizer.encode(prompt, return_tensors="pt"))
        inputs = self.tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]

        pixel_values = None
        if images is not None:
            pixel_values = self.image_processor(images, return_tensors="pt")["pixel_values"]

        # Placeholder: future support for videos
        # You can log, validate, or prepare preprocessor for later
        if videos is not None:
            print("[Debug] Received videos (ignored for now):", type(videos), len(videos) if hasattr(videos, '__len__') else "N/A")


        kwargs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "prompt_text": prompt,
        }

        if images is not None and len(images) > 0:
            pixel_values = self.image_processor(images, return_tensors="pt")["pixel_values"]
            kwargs["pixel_values"] = pixel_values

        return TarProcessorOutput(**kwargs)

    def __call__(self, text=None, images=None, videos=None, conversations=None, prompt=None, **kwargs):
        """
        Unified processor call to support both Hugging Face datasets and custom usage.
        `text` is expected to be a list or str. Only the first element will be used if list.
        """
        # Extract prompt from `text` if provided
        if prompt is None and text is not None:
            if isinstance(text, list):
                prompt = text[0]
            else:
                prompt = text

        # Check for actual visual inputs (non-empty images/videos)
        has_image = images is not None and isinstance(images, list) and len(images) > 0
        has_video = videos is not None and isinstance(videos, list) and len(videos) > 0

        # Only forward images/videos if present
        if has_image or has_video:
            return self.process_one(prompt=prompt, conversations=conversations, images=images, videos=videos)
        else:
            return self.process_one(prompt=prompt, conversations=conversations)

    
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

    def apply_chat_template(self, conversation: List[Dict[str, str]], tokenize=False, add_generation_prompt=True):
        """
        Applies chat template with system prompt injected if not already present.
        """
        if conversation[0].get("role", "") != "system":
            conversation = [{"role": "system", "content": self.system_prompt}] + conversation

        return self.tokenizer.apply_chat_template(
            conversation,
            tokenize=tokenize,
            add_generation_prompt=add_generation_prompt
        )

