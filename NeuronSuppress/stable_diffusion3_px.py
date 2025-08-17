import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import torch
from diffusers import StableDiffusion3Pipeline
from tqdm import tqdm
from accelerate import PartialState

device = 'cuda' if torch.cuda.is_available() else 'cpu'


model_path = '/data/junlei/project-code/diffusion/models/stable-diffusion-3.5-large/'

pipe = StableDiffusion3Pipeline.from_pretrained(model_path, torch_dtype=torch.bfloat16)
pipe.to(device)


def get_prompts(file_path):
    prompts = []
    with open(file_path, 'r') as f:
        prompts = f.readlines()
        if prompts:
            prompts = [prompt.strip() for prompt in prompts]
    return prompts

prompt_file = "data/prompt.csv"
prompts = get_prompts(prompt_file)

account_num = 87
for prompt in tqdm(prompts):
    
    account_num += 1
    if not os.path.exists("output/output_sd3_" + str(account_num)):
        os.makedirs("output/output_sd3_" + str(account_num))
    for nums in range(51, 100):
        image = pipe(
            prompt,
            num_inference_steps=28,
            guidance_scale=3.5,
        ).images[0]

        output_image_name = "output/output_sd3_" + str(account_num) + "/" + str(nums) + ".png"
        image.save(output_image_name)
