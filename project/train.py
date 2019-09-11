import torch
import torchvision as tv
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from project.models import EncoderDecoder
from typing import TypeVar
import torch.nn as nn
from itertools import chain, islice
from torch import Tensor
import numpy as np

# %% load data

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=(0.4914, 0.4822, 0.4466),
        std=(0.247, 0.243, 0.261))])

invTransform = transforms.Compose([
    transforms.Normalize(mean=[0., 0., 0.],
                         std=[1 / 0.247, 1 / 0.243, 1 / 0.261]),
    transforms.Normalize(mean=[-0.4914, -0.4822, -0.4466],
                         std=[1., 1., 1.]),
    transforms.ToPILImage()
])


def getDataSet(root: str, train: bool) -> DataLoader:
    dataSet = tv.datasets.CIFAR10(root, train=train, transform=transform,
                                  download=True)
    loader = DataLoader(dataSet, batch_size=64,
                        shuffle=True, num_workers=4)
    return loader


root = "../data/CIFAR/"
trainSet: DataLoader = getDataSet(root, True)
testSet: DataLoader = getDataSet(root, False)

# %% initialize model
T = TypeVar('T')
device = torch.device("cpu")


def toDevice(m: T) -> T:
    return m.to(device, torch.float)


model: EncoderDecoder = EncoderDecoder()
lossModel = nn.L1Loss()
allParams = chain(model.parameters())
optimizer = torch.optim.Adam(allParams, lr=1e-4, weight_decay=1e-5)

# %% test on sample images
from PIL import Image
from pathlib import Path


def testOnSample(fromDir: Path, toDir: Path):
    for img_path in fromDir.glob('*.jpg'):
        img = Image.open(img_path)
        img = transform(img)[None, :, :, :]
        img = model(img).squeeze(0)
        img = invTransform(img)
        img.save("{}/{}".format(toDir, img_path.name), "JPEG")


# %% training loop
import datetime
from torch.utils.tensorboard import SummaryWriter


def trainOnBatch(img: Tensor) -> np.array:
    img: Tensor = toDevice(img)
    out = model(img)
    loss = lossModel(out, img)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.detach().numpy()


def testOnBatch(img: Tensor) -> np.array:
    img = toDevice(img)
    with torch.no_grad():
        out = model(img)
        loss = lossModel(out, img)
        return loss.detach().numpy()


writer = SummaryWriter(log_dir="../runs")

trainBatches = 3
testBatches = min(3, len(testSet))

startTime = datetime.datetime.now().ctime()


def trainingLoop():
    step = 0
    for epoch in range(3):
        print("===epoch {}===".format(epoch))
        print('-' * trainBatches)
        for img, _ in islice(trainSet, trainBatches):
            lossValue = trainOnBatch(img)

            writer.add_scalar('Loss/train', lossValue, step)
            print(".", end='')
            step += 1
        print()

        print("start testing")
        losses = []
        print('-' * testBatches)
        for img, _ in islice(testSet, testBatches):
            losses.append(testOnBatch(img))
            print(".", end='')
        print()
        avgLoss = np.mean(losses)
        writer.add_scalar('Loss/test', avgLoss, step)

        saveDir = Path("../saves/{}/epoch{}".format(startTime, epoch))
        saveDir.mkdir(parents=True)
        testOnSample(Path("../data"), saveDir)

        torch.save(model.state_dict(), '{}/state_dict.pth'.format(str(saveDir)))


trainingLoop()
