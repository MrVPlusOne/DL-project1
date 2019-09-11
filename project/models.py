import torch
from torch import Tensor
import torch.nn as nn
from typing import *
import numpy as np
from PIL import Image

"""
Optional: Your code here
"""

channels = [3, 16, 8, 8]
outChannels = 8
kernel_size = 3
stride = 2
padding = 0

imageSize = 256
maxOutputBytes = 8192
maxOutputFloats = int(maxOutputBytes / 4)

def testOutputSize():
    enc: Encoder = Encoder()
    x = torch.randn([1, 3, imageSize, imageSize], dtype=torch.float)
    r = enc.forward(x)
    print("output shape {} has {} elements. Required limit = {}".format(r.shape, r.numel(), maxOutputFloats))
    dec: Decoder = Decoder()
    r2 = dec.forward(r)
    print("reconstructed shape: {}".format(r2.shape))

def encodeLayers() -> nn.Sequential:
    layers = [
        layer
        for i in range(len(channels) - 1)
        for layer in [
            nn.Conv2d(channels[i], channels[i + 1], kernel_size, stride, padding),
            nn.ReLU(True)]]
    layers.append(nn.Conv2d(channels[-1], outChannels, kernel_size, stride, padding))
    return nn.Sequential(*layers)


def decodeLayers() -> nn.Sequential:
    cns = channels.copy()
    cns.reverse()
    layers = [nn.ConvTranspose2d(outChannels, cns[0], kernel_size, stride, padding=padding)]

    def outPadding(i: int) -> int:
        if i == len(cns) - 2:
            return 1
        else:
            return 0

    layers.extend([
        layer
        for i in range(len(cns) - 1)
        for layer in [
            nn.ReLU(True),
            nn.ConvTranspose2d(cns[i], cns[i + 1], kernel_size, stride, padding=padding, output_padding=outPadding(i)),
        ]])
    return nn.Sequential(*layers)


class Encoder(torch.nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.model = encodeLayers()

    def forward(self, img) -> Tensor:
        return self.model.forward(img)

    def encodeImage(self, img, device, transform) -> Tensor:
        t = torch.tensor(np.array(img), dtype=torch.float, device=device)
        t = t.permute(2, 0, 1)[None, :, :, :]
        t = transform(t)
        return self.forward(t)


class Decoder(torch.nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.model = decodeLayers()

    def forward(self, img) -> Tensor:
        return self.model.forward(img)

    def decodeImage(self, t: Tensor, invTransform):
        t = self.forward(t)
        t = invTransform(t)
        array = t.squeeze(0).permute(1,2,0).detach().numpy()
        return Image.fromarray(array)

class EncoderDecoder(torch.nn.Module):
    def __init__(self):
        super(EncoderDecoder, self).__init__()
        self.encoder: Encoder = Encoder()
        self.decoder: Decoder = Decoder()

    def forward(self, img) -> Tensor:
        return self.decoder(self.encoder(img))

    def encodeDecodeImage(self, img, device, transform, invTransform):
        enc: Encoder = self.encoder
        t = enc.encodeImage(img, device, transform)
        dec: Decoder = self.decoder
        return dec.decodeImage(t, invTransform)


# testOutputSize()
