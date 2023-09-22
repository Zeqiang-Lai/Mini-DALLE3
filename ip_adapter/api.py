import torch
from diffusers import StableDiffusionXLPipeline
from PIL import Image

from . import IPAdapterXL


class IPAdapterWrapper:
    def __init__(self) -> None:
        base_model_path = "stabilityai/stable-diffusion-xl-base-1.0"
        image_encoder_path = "checkpoints/sdxl_models/image_encoder"
        ip_ckpt = "checkpoints/sdxl_models/ip-adapter_sdxl.bin"
        device = "cuda"

        # load SDXL pipeline
        self.pipe = StableDiffusionXLPipeline.from_pretrained(
            base_model_path,
            torch_dtype=torch.float16,
            add_watermarker=False,
            local_files_only=True,
        )
        self.pipe.to(device)

        # load ip-adapter
        self.ip_model = IPAdapterXL(self.pipe, image_encoder_path, ip_ckpt, device)

        self.a_prompt = 'best quality, extremely detailed'
        self.n_prompt = 'longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, ' \
                        'fewer digits, cropped, worst quality, low quality'
                        
    def generate(self, prompt):
        prompt = prompt
        images = self.pipe(prompt).images
        return images[0]

    
    def variation(self, image, num_samples=1):
        images = self.ip_model.generate(pil_image=image,
                                        num_samples=num_samples,
                                        num_inference_steps=30)
        return images[0]

    def edit(self, image, prompt, num_samples=1):
        # multimodal prompts
        prompt = prompt + ', ' + self.a_prompt
        images = self.ip_model.generate(pil_image=image, num_samples=num_samples, num_inference_steps=30,
                                        prompt=prompt, scale=0.4, negative_prompt=self.n_prompt)
        return images[0]
