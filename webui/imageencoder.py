import os
import sys
import argparse

import cv2
import numpy as np

import torch
import torchvision

sys.path.insert(0, "../")

import stylegan2
from stylegan2 import utils

from pspencoder import CreateEncoder
import torch.nn.functional as F

#torch.set_grad_enabled(False)

def encode_images(device, G, encoder, dlatent_avg, images, truncation_psi, num_steps):
    lpips_model = stylegan2.external_models.lpips.LPIPS_VGG16(pixel_min=-1, pixel_max=1)

    proj = stylegan2.project.Projector(
        G=G,
        dlatent_avg_samples=10000,
        dlatent_avg_label=None,
        dlatent_device=device,
        dlatent_batch_size=1024,
        lpips_model=lpips_model,
        lpips_size=256
    )

    dlatent_param = encoder(F.interpolate(images, (256,256), mode='bicubic'))
    dlatent_param = dlatent_param + dlatent_avg.repeat(dlatent_param.shape[0], 1, 1)

    batch_size = 1

    proj.start(
        target=images,
        dlatent_param=dlatent_param,
        num_steps=num_steps,
        initial_learning_rate=0.1,
        initial_noise_factor=0.05,
        lr_rampdown_length=0.25,
        lr_rampup_length=0.05,
        noise_ramp_length=0.75,
        regularize_noise_weight=1e5,
        verbose=True,
        verbose_prefix='Projecting image(s) {}/{}'.format(0 * batch_size + len(images), len(images))
        )

    for j in range(num_steps):
        proj.step()

    dlatents = proj.get_dlatent()
    return dlatents

def encode_real_images(device, G, encoder, dlatent_avg, faceimg, truncation_psi, num_steps):
    images = np.expand_dims(faceimg.transpose((2,0,1)), 0)
    images = torch.Tensor(images).to(device)
    images = torchvision.transforms.Normalize(mean=0.5,std=0.5)(images/255.)
    dlatents = encode_images(device, G, encoder, dlatent_avg, images, truncation_psi, num_steps)
    return dlatents.detach().cpu().numpy()

def encoder_init(device):
    G = stylegan2.models.load('../mymodels/Gs_ffhq.pth')
    G = utils.unwrap_module(G).to(device)
    G.eval()

    encoder = CreateEncoder()
    state_dict = torch.load('../mymodels/psp_encoder.pth', map_location=lambda storage, loc: storage)
    encoder.load_state_dict(state_dict['encoder'])
    encoder.to(device)
    encoder.eval()

    dlatent_avg = torch.nn.Parameter(state_dict['dlatent_avg'].clone()).to(device).detach()

    return encoder, G, dlatent_avg
