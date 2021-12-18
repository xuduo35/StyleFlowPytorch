import sys

from .utils.utils_SH import *

# other modules
import os
import numpy as np

from torch.autograd import Variable
from torchvision.utils import make_grid
import torch
import time
import cv2

# load model
from .model.defineHourglass_1024_gray_skip_matchFeature import *


modelFolder = os.path.join(os.path.dirname(__file__), 'trained_model')
lightFolder = os.path.join(os.path.dirname(__file__), 'example_light')

shading = [ \
    np.loadtxt(os.path.join(lightFolder, 'rotate_light_{:02d}.txt'.format(i))) \
    for i in range(7) \
    ]
shading = np.asarray(shading)
shading = shading[0:9] * 0.7

def dpr_init(device):
    model_512 = HourglassNet(16)
    model = HourglassNet_1024(model_512, 16)
    model.load_state_dict(torch.load(os.path.join(modelFolder, 'trained_model_1024_03.t7')))
    model.to(device)
    model.eval()

    return model

def get_lightvec(device, model, img):
    Lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

    inputL = Lab[:,:,0]
    inputL = inputL.astype(np.float32)/255.0
    inputL = inputL.transpose((0,1))
    inputL = inputL[None,None,...]
    inputL = Variable(torch.from_numpy(inputL).to(device))

    for i in range(7):
        sh = shading[6,:].copy()
        sh = np.reshape(sh, (1,9,1,1)).astype(np.float32)
        sh = Variable(torch.from_numpy(sh).to(device))
        outputImg, outLight, outputSH, outputSH_ori = model(inputL, sh, 0)
        return outputSH
