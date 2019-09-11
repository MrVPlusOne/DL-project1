import torch
import numpy as np
from .models import EncoderDecoder, transform, invTransform, toDevice

model: EncoderDecoder = EncoderDecoder()
model.load_state_dict(torch.load('savedModel/state_dict.pth'))
model.eval()


def encode(img) -> np.ndarray:
    with torch.no_grad():
        img = toDevice(transform(img)[None, :, :, :])
        img: torch.Tensor = model.encoder(img)
        return img.detach().cpu().numpy()


def decode(img: np.ndarray):
    with torch.no_grad():
        t = toDevice(torch.tensor(img, dtype=torch.float))
        t = model.decoder(t).squeeze(0).detach().cpu()
        img = invTransform(t)
        return img
