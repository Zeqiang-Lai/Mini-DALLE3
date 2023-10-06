import logging
import re
import dataclasses
import PIL
import math

import openai
import torch
from diffusers import DiffusionPipeline, StableDiffusionXLPipeline

from .t2i.ip_adapter import IPAdapterXL

logger = logging.getLogger(__file__)


def user(content):
    return {'role': 'user', 'content': content}


def ai(content):
    return {'role': 'assistant', 'content': content}


def chat(messages):
    result = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=messages,
        temperature=0,
    )
    response = result['choices'][0]['message']['content']
    logger.info(response)
    return response


def extract_pattern(message, pattern):
    matches = re.findall(pattern, message, re.DOTALL)
    for match in matches:
        return match.strip()
    return None


def remove_pattern(message, pattern):
    return re.sub(pattern, '', message, flags=re.DOTALL)


def image_grid(imgs):
    n = len(imgs)
    rows = int(math.sqrt(n))
    cols = math.ceil(n/rows)

    w, h = imgs[0].size
    grid = PIL.Image.new('RGB', size=(cols*w, rows*h))
    grid_w, grid_h = grid.size

    for i, img in enumerate(imgs):
        grid.paste(img, box=(i % cols*w, i//cols*h))
    return grid


class APIPool:
    def __init__(
        self,
        base_model_path="stabilityai/stable-diffusion-xl-base-1.0",
        image_encoder_path="checkpoints/sdxl_models/image_encoder",
        ip_ckpt="checkpoints/sdxl_models/ip-adapter_sdxl.bin",
        device="cuda"
    ) -> None:
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

        self.t2i = DiffusionPipeline.from_pretrained(
            base_model_path,
            torch_dtype=torch.float16,
            add_watermarker=False,
            local_files_only=True,
        )
        self.t2i.to(device)

        self.a_prompt = 'best quality, extremely detailed'
        self.n_prompt = 'longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, ' \
                        'fewer digits, cropped, worst quality, low quality'

    def text_to_image(self, prompt, num_samples=1):
        prompt = prompt + ', ' + self.a_prompt
        images = self.t2i(prompt, negative_prompt=self.n_prompt,
                          num_images_per_prompt=num_samples, num_inference_steps=30).images
        return image_grid(images)

    def variation(self, image, num_samples=1):
        images = self.ip_model.generate(pil_image=image,
                                        num_samples=num_samples,
                                        num_inference_steps=30)
        return image_grid(images)

    def edit(self, image, prompt, num_samples=1):
        # multimodal prompts
        prompt = prompt + ', ' + self.a_prompt
        images = self.ip_model.generate(pil_image=image, num_samples=num_samples, num_inference_steps=30,
                                        prompt=prompt, scale=0.3, negative_prompt=self.n_prompt)
        return image_grid(images)


api = APIPool()


@dataclasses.dataclass
class Response:
    response: str
    image: PIL.Image.Image = None
    image_prompt: str = None

class Text2Image:
    PATTERN = r'<image>(.*?)<\/image>'

    def process(self, message, history_messages, history_images):
        image_prompt = extract_pattern(message, self.PATTERN)
        if image_prompt is None:
            return None
        logger.info(f'Text2Image(prompt="{image_prompt}")')
        image = api.text_to_image(image_prompt)
        response = remove_pattern(message, self.PATTERN)
        return Response(response, image, image_prompt)


class ImageEdit:
    PATTERN = r'<edit>(.*?)<\/edit>'

    def process(self, response, history_messages, history_images):
        image_prompt = extract_pattern(response, self.PATTERN)
        if image_prompt is None:
            return None
        logger.info(f'Edit(prompt="{image_prompt}")')

        if len(history_images) > 0:
            last_image = history_images[-1]
            image = api.edit(last_image, image_prompt)
        else:
            image = api.text_to_image(image_prompt)

        response = remove_pattern(response, self.PATTERN)
        return Response(response, image, image_prompt)


DEFAULT_PROMPT = 'minidalle3/prompts/prompt-v2.txt'


class MiniDALLE3:
    def __init__(
        self,
        llm='gpt3.5',
        prompt_path=None,
    ) -> None:
        self.tools = [
            Text2Image(),
            ImageEdit(),
        ]

        if prompt_path is None:
            prompt_path = DEFAULT_PROMPT
        self.system_message = {
            'role': 'system',
            'content': open(prompt_path, 'r').read().strip()
        }

    def ask(self, history_messages, history_images):
        output = chat(history_messages)

        for tool in self.tools:
            response = tool.process(output, history_messages, history_images)
            if response is not None:
                return output, response

        return output, None
