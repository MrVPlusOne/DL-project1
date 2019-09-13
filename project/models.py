import torch
from torch import Tensor
import torch.nn as nn
from typing import *
import numpy as np
from PIL import Image
import torch.nn.functional as F
import torchvision.transforms as transforms

channels = [3, 24, 16, 12]
outChannels = 8
kernelSize = 5
stride = 2
padding = 2

imageSize = 256
maxOutputBytes = 8192
maxOutputFloats = int(maxOutputBytes / 4)

transform = transforms.Compose([
    transforms.Resize(imageSize),
    transforms.RandomCrop(imageSize),
    transforms.ToTensor()
])

invTransform = transforms.Compose([
    transforms.ToPILImage()
])

T = TypeVar('T')
device = torch.device("cpu")


def toDevice(m: T) -> T:
    return m.to(device, torch.float)


def testOutputSize():
    enc: Encoder = Encoder()
    x = torch.randn([1, 3, imageSize, imageSize], dtype=torch.float)
    r = enc(x)
    print("output shape {} has {} elements. Required limit = {}".format(r.shape, r.numel(), maxOutputFloats))
    dec: Decoder = Decoder()
    r2 = dec(r)
    print("reconstructed shape: {}".format(r2.shape))


def doubleRelu(x, slop=0.01):
    return F.relu(x) - F.relu(x - 1.0) - F.relu(-x) * slop + F.relu(x - 1.0) * slop


def encodeLayers() -> nn.Sequential:
    layers = [
        layer
        for i in range(len(channels) - 1)
        for layer in [
            nn.Conv2d(channels[i], channels[i + 1], kernelSize, stride, padding),
            nn.ReLU(True)]]
    layers.append(nn.Conv2d(channels[-1], outChannels, kernelSize, stride, padding))
    return nn.Sequential(*layers)


def decodeLayers() -> nn.Sequential:
    cns = channels.copy()
    cns.reverse()
    layers = [nn.ConvTranspose2d(outChannels, cns[0], kernelSize, stride, padding=padding, output_padding=1)]

    layers.extend([
        layer
        for i in range(len(cns) - 1)
        for layer in [
            nn.ReLU(True),
            nn.ConvTranspose2d(cns[i], cns[i + 1], kernelSize, stride, padding=padding, output_padding=1),
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
        return doubleRelu(self.model(img))


class EncoderDecoder(torch.nn.Module):
    def __init__(self):
        super(EncoderDecoder, self).__init__()
        self.encoder: Encoder = Encoder()
        self.decoder: Decoder = Decoder()

    def forward(self, img) -> Tensor:
        return self.decoder(self.encoder(img))


# testOutputSize()
