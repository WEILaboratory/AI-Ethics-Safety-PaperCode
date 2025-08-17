# from typing import Optional, Tuple, Union
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from captum.attr import IntegratedGradients
# from diffusers.utils import deprecate, is_torch_version
# from diffusers import UNet2DConditionModel

# device = 'cuda' if torch.cuda.is_available() else 'cpu'

# # model_path = '/data/junlei/project-code/diffusion/models/stable-diffusion-xl-base-1.0'
    
# # unet = UNet2DConditionModel.from_pretrained(model_path, subfolder='unet', torch_dtype=torch.float16, variant='fp16', use_safetensors=True).to("cuda")
# # unet.eval()

# # def forward_func(input, condition):
# #     return unet(input, condition).sample


# # input = torch.randn(1, 4, 64, 64)
# # condition = torch.randn(1, 77, 768)
# # baseline = torch.zeros_like(input)


# # # defining and applying integrated gradients on ToyModel and the
# # ig = IntegratedGradients(forward_func)
# # attributions, approximation_error = ig.attribute(input,
# #                                                  baselines=baseline,
# #                                                  additional_forward_args=(condition,),
# #                                                  method='gausslegendre',
# #                                                  return_convergence_delta=True)

# # print(attributions, approximation_error)

# class ToySoftmaxModel(nn.Module):

#     def __init__(self, num_in, num_hidden, num_out):
#         super().__init__()
#         self.num_in = num_in
#         self.num_hidden = num_hidden
#         self.num_out = num_out
#         self.lin1 = nn.Linear(num_in, num_hidden)
#         self.lin2 = nn.Linear(num_hidden, num_hidden)
#         self.lin3 = nn.Linear(num_hidden, num_out)
#         self.softmax = nn.Softmax(dim=1)

#     def forward(self, input):
#         lin1 = F.relu(self.lin1(input))
#         lin2 = F.relu(self.lin2(lin1))
#         lin3 = self.lin3(lin2)
#         return self.softmax(lin3)
    
# num_in = 40
# input = torch.arange(0.0, num_in * 1.0, requires_grad=True).unsqueeze(0)

# # 10-class classification model
# model = ToySoftmaxModel(num_in, 20, 10)

# # attribution score will be computed with respect to target class
# target_class_index = 5

# # applying integrated gradients on the SoftmaxModel and input data point
# ig = IntegratedGradients(model)
# attributions, approximation_error = ig.attribute(input, target=target_class_index,
#                                     return_convergence_delta=True)

# # The input and returned corresponding attribution have the
# # same shape and dimensionality.


import os
os.environ["CUDA_VISIBLE_DEVICES"] = "5, 6, 7"
import torch

import numpy as np
import torch.nn as nn
from diffusers import DiffusionPipeline, StableDiffusionXLPipeline
from tqdm.auto import tqdm
from captum.attr import IntegratedGradients
from transformers import CLIPTextModel, CLIPTextModelWithProjection, CLIPTokenizer
from diffusers import EulerDiscreteScheduler, UNet2DConditionModel, AutoencoderKL

device = 'cuda' if torch.cuda.is_available() else 'cpu'

device = os.environ.get("CUDA_VISIBLE_DEVICES", "cuda")
print(device)

# model_path = '/data/junlei/project-code/diffusion/models/stable-diffusion-xl-base-1.0'

# text_encoder = CLIPTextModel.from_pretrained(model_path, subfolder='text_encoder', torch_dtype=torch.float16, variant='fp16', use_safetensors=True).to("cuda")
# text_encoder_2 = CLIPTextModelWithProjection.from_pretrained(model_path, subfolder='text_encoder_2', torch_dtype=torch.float16, variant='fp16', use_safetensors=True).to("cuda")
# tokenizer = CLIPTokenizer.from_pretrained(model_path, subfolder='tokenizer', torch_dtype=torch.float16, variant='fp16', use_safetensors=True)
# tokenizer_2 = CLIPTokenizer.from_pretrained(model_path, subfolder='tokenizer_2', torch_dtype=torch.float16, variant='fp16', use_safetensors=True)
# unet = UNet2DConditionModel.from_pretrained(model_path, subfolder='unet', torch_dtype=torch.float16, variant='fp16', use_safetensors=True).to("cuda")
# vae = AutoencoderKL.from_pretrained(model_path, subfolder='vae', torch_dtype=torch.float16, variant='fp16', use_safetensors=True).to("cuda")
# scheduler = EulerDiscreteScheduler.from_pretrained(model_path, subfolder='scheduler', torch_dtype=torch.float16, variant='fp16', use_safetensors=True)

# # unet.state_dict()
# # unet, scheduler


# # text_encoder, tokenizer


# for name, module in unet.named_modules():
#     if isinstance(module, nn.Linear) and any(x in name for x in ['to_q', 'to_k', 'to_v']):
#         print(">>>", name, module.__class__.__name__)
#         model = unet
#         module_keys = name.split('.')
#         for key in module_keys:
#             if key.isdigit():
#                 model = model[int(key)]
#             else:
#                 model  = getattr(model, key)

#         print(model.state_dict())