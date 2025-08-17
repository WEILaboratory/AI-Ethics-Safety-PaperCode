import os

os.environ["CUDA_VISIBLE_DEVICES"] = "5, 6, 7, 8"

import torch

from PIL import Image
from captum.attr import IntegratedGradients
from StableDiffusionXL import StableDiffusionXLPipeline
from transformers import CLIPTextModel, CLIPTextModelWithProjection, CLIPTokenizer
from diffusers import EulerDiscreteScheduler, UNet2DConditionModel, AutoencoderKL

from diffusers.loaders import (
    StableDiffusionXLLoraLoaderMixin,
    TextualInversionLoaderMixin,
)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

model_path = '/data/junlei/project-code/diffusion/models/stable-diffusion-xl-base-1.0'

text_encoder = CLIPTextModel.from_pretrained(model_path, subfolder='text_encoder', torch_dtype=torch.float16,
                                             variant='fp16', use_safetensors=True).to("cuda")
text_encoder_2 = CLIPTextModelWithProjection.from_pretrained(model_path, subfolder='text_encoder_2',
                                                             torch_dtype=torch.float16, variant='fp16',
                                                             use_safetensors=True).to("cuda")
tokenizer = CLIPTokenizer.from_pretrained(model_path, subfolder='tokenizer', torch_dtype=torch.float16, variant='fp16',
                                          use_safetensors=True)
tokenizer_2 = CLIPTokenizer.from_pretrained(model_path, subfolder='tokenizer_2', torch_dtype=torch.float16,
                                            variant='fp16', use_safetensors=True)
unet = UNet2DConditionModel.from_pretrained(model_path, subfolder='unet', torch_dtype=torch.float16, variant='fp16',
                                            use_safetensors=True).to("cuda")


class EncoderImageBias(StableDiffusionXLPipeline):
    def __init__(self):
        super(EncoderImageBias, self).__init__()
        self.image_embedding = None
        self.bias_embedding = None
        self.image_prompt = None

    def __call__(self,
                 image,
                 bias_description,
                 prompt,
                 sample,
                 distribution):
        self.generated_image = self.encode_image(image)
        self.bias_embedding = self.encode_prompt(bias_description)
        self.image_prompt = self.encode_prompt(prompt)


def calculate_attribution(input_embedding, embedding, batch_size, num_batch):
    baseline = torch.zeros_like(input_embedding)

    # embeds_test = torch.randn(2, 4, 128, 128)

    # input_embed = input_embedding.view(-1, )

def get_prompts(file_path):
    prompts = []
    with open(file_path, 'r') as f:
        prompts = f.readlines()
        if prompts:
            prompts = [prompt.strip() for prompt in prompts]
    return prompts


def main():
    prompt_file = "data/prompt.csv"
    prompts = get_prompts(prompt_file)

    generator = torch.manual_seed(29)
    num_inference_steps = 75
    n_steps = 40
    high_noise_frac = 0.8
    height = 512
    width = 512

    generate_picture_num = 1
    account_num = 31
    print(prompts)

    for prompt in prompts:
        
        account_num += 1


        pipe = StableDiffusionXLPipeline.from_pretrained(model_path, torch_dtype=torch.float16, use_safetensors=True,
                                                        variant="fp16")
        pipe.to(device)

        refiner = StableDiffusionXLPipeline.from_pretrained(
            model_path,
            text_encoder_2=pipe.text_encoder_2,
            vae=pipe.vae,
            torch_dtype=torch.float16,
            use_safetensors=True,
            variant="fp16",
        )
        refiner.to(device)

        n_steps = 40

        image, attribution, delta = pipe(
            prompt=prompt,
            num_inference_steps=n_steps,
            denoising_end=high_noise_frac,
            output_type="latents",
        )

        images, attribution, delta = refiner(
            prompt=prompt,
            num_images_steps=n_steps,
            denoising_start=high_noise_frac,
            image=image[0],
        )
        image = images[0]


        print(image, type(image), attribution, type(attribution), delta)
        
        output_image_name = "output/output" + str(account_num) + ".png"
        image[0].save(output_image_name)


if __name__ == '__main__':
    main()
