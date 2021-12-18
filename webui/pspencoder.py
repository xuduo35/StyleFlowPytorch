import math
from encoders import psp_encoders
import torch

class EncoderOption:
    def __init__(self, n_styles, input_nc):
        self.n_styles = n_styles
        self.input_nc = input_nc

def CreateEncoder(imgsize=1024, input_nc=3, encoder_type='GradualStyleEncoder'):
    n_styles = int(math.log(imgsize, 2)) * 2 - 2

    opts = EncoderOption(n_styles, input_nc)

    if encoder_type == 'GradualStyleEncoder':
        encoder = psp_encoders.GradualStyleEncoder(50, 'ir_se', opts)
    elif encoder_type == 'BackboneEncoderUsingLastLayerIntoW':
        encoder = psp_encoders.BackboneEncoderUsingLastLayerIntoW(50, 'ir_se', opts)
    elif encoder_type == 'BackboneEncoderUsingLastLayerIntoWPlus':
        encoder = psp_encoders.BackboneEncoderUsingLastLayerIntoWPlus(50, 'ir_se', opts)
    else:
        raise Exception('{} is not a valid encoders'.format(encoder_type))
    return encoder
