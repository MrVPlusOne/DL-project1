import numpy as np
import PIL

def encode(img):
    """
    Your code here
    img: a 256x256 PIL Image
    """
    
    raise NotImplementedError('encode')
    
def decode(x):
    """
    Your code here
    x: a numpy array, <= 4096 Byte
    """
    
    raise NotImplementedError('decode')
    
# ====================================== #
#          Optional for PyTorch          #
# ====================================== #
# from .models import Encoder, Decoder
# encode = Encoder()
# decode = Decoder()
# encode.eval()
# decode.eval()

# # Load model weights here