import torch
from torch import Tensor
import torch.nn as nn
from typing import *

"""
Optional: Your code here
"""

channels = [3, 16, 8, 8]
outChannels = 8
kernel_size = 3
stride = 2

imageSize = 256
maxOutputBytes = 8192
maxOutputFloats = int(maxOutputBytes / 4)


def testOutputSize():
    enc: Encoder = Encoder()
    x = torch.randn([1, 3, imageSize, imageSize], dtype=torch.float)
    r = enc.forward(x)
    print("output shape {} has {} elements. Required limit = {}".format(r.shape, r.numel(), maxOutputFloats))


def encodeLayers() -> nn.Sequential:
    layers = [
        layer
        for i in range(len(channels) - 1)
        for layer in [
            nn.Conv2d(channels[i], channels[i + 1], kernel_size, stride),
            nn.ReLU(True)]]
    layers.append(nn.Conv2d(channels[-1], outChannels, kernel_size, stride))
    return nn.Sequential(*layers)


def decodeLayers() -> nn.Sequential:
    cns = channels.copy()
    cns.reverse()
    layers = [nn.ConvTranspose2d(outChannels, cns[0], kernel_size, stride)]
    layers.extend([
        layer
        for i in range(len(cns) - 1)
        for layer in [
            nn.ReLU(True),
            nn.ConvTranspose2d(cns[i], cns[i + 1], kernel_size, stride),
        ]])
    return nn.Sequential(*layers)


class Encoder(torch.nn.Module):
    def __init__(self):
        """
        Your code here
        """
        super(Encoder, self).__init__()
        self.model = encodeLayers()

    def forward(self, img) -> Tensor:
        """
        Your code here
        """
        return self.model.forward(img)


class Decoder(torch.nn.Module):
    def __init__(self):
        """
        Your code here
        """
        super(Decoder, self).__init__()
        self.model = decodeLayers()

    def forward(self, img):
        """
        Your code here
        """
        return self.model.forward(img)
