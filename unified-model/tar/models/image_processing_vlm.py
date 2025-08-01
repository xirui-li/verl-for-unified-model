# tar_image_processor.py

from typing import List, Tuple, Union
from PIL import Image
import numpy as np
import torch
import torchvision.transforms.functional as F
from transformers.image_processing_utils import BaseImageProcessor, BatchFeature
from transformers.image_utils import to_numpy_array

ImageType = Union[np.ndarray, torch.Tensor, Image.Image]

IMAGENET_MEAN = (0.48145466, 0.4578275, 0.40821073)
IMAGENET_STD = (0.26862954, 0.26130258, 0.27577711)

def expand2square(pil_img, background_color):
    width, height = pil_img.size
    if width == height:
        return pil_img
    elif width > height:
        result = Image.new(pil_img.mode, (width, width), background_color)
        result.paste(pil_img, (0, (width - height) // 2))
        return result
    else:
        result = Image.new(pil_img.mode, (height, height), background_color)
        result.paste(pil_img, ((height - width) // 2, 0))
        return result

class VLMImageProcessor(BaseImageProcessor):
    model_input_names = ["pixel_values"]

    def __init__(
        self,
        image_size: int = 256,
        image_mean: Union[Tuple[float, float, float], List[float]] = IMAGENET_MEAN,
        image_std: Union[Tuple[float, float, float], List[float]] = IMAGENET_STD,
        rescale_factor: float = 1.0 / 255.0,
        do_normalize: bool = True,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.image_size = image_size
        self.image_mean = image_mean
        self.image_std = image_std
        self.rescale_factor = rescale_factor
        self.do_normalize = do_normalize
        self.background_color = tuple(int(x * 255) for x in image_mean)

    def resize(self, pil_img: Image.Image) -> np.ndarray:
        pil_img = pil_img.convert("RGB")
        pil_img = pil_img.resize((self.image_size, self.image_size), resample=Image.BICUBIC)
        pil_img = expand2square(pil_img, self.background_color)
        x = to_numpy_array(pil_img)  # [H, W, 3]
        x = np.transpose(x, (2, 0, 1))  # [3, H, W]
        return x

    def preprocess(self, images, return_tensors: str = "pt", **kwargs) -> BatchFeature:
        images = [self.resize(image) for image in images]

        # rescale
        images = [
            self.rescale(
                image=image,
                scale=self.rescale_factor,
                input_data_format="channels_first",
            ) for image in images
        ]

        # normalize
        if self.do_normalize:
            images = [
                self.normalize(
                    image=image,
                    mean=self.image_mean,
                    std=self.image_std,
                    input_data_format="channels_first",
                ) for image in images
            ]

        data = {"pixel_values": images}
        return BatchFeature(data=data, tensor_type=return_tensors)

    @property
    def default_shape(self):
        return [3, self.image_size, self.image_size]
