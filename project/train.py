import torch
import torchvision as tv
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import random
from pathlib import Path
from project.models import *
from typing import TypeVar
import torch.nn as nn
from itertools import chain, islice
from torch import Tensor
import numpy as np


# %% load data
def getDataSet() -> (DataLoader, DataLoader):
    root = "data/ImageNet/"
    dataSet = tv.datasets.ImageNet(root, split='val', download=True, transform=transform)
    totalSize = len(dataSet)
    allIndices = list(range(0, totalSize))
    random.shuffle(allIndices)
    testSize = int(totalSize / 10)
    [train, test] = [DataLoader(dataSet, batch_size=64,
                                sampler=SubsetRandomSampler(indices), num_workers=4)
                     for indices in [allIndices[testSize:], allIndices[:testSize]]]
    return [train, test]


trainSet: DataLoader
testSet: DataLoader
trainSet, testSet = getDataSet()

# %% initialize model
model: EncoderDecoder = EncoderDecoder()
lossModel = nn.L1Loss()
allParams = chain(model.parameters())
optimizer = torch.optim.Adam(allParams, lr=1e-4, weight_decay=1e-5)


def loadModelFromFile(file: Path):
    model.load_state_dict(torch.load(file))


loadModelFromFile(Path("saves/Thu Sep 12 13:11:33 2019/epoch11/state_dict.pth"))
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


def showAtSamePlace(content):
    import sys
    sys.stdout.write(content + "   \r")
    sys.stdout.flush()


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


writer = SummaryWriter(comment="original_scale", flush_secs=30)

trainBatches = len(trainSet)
testBatches = len(testSet)
print("train/dev size: {}/{}".format(trainBatches, testBatches))

startTime = datetime.datetime.now().ctime()


def trainingLoop():
    step = 0
    for epoch in range(12, 30):
        print("===epoch {}===".format(epoch))
        progress = 0
        for img, _ in islice(trainSet, trainBatches):
            lossValue = trainOnBatch(img)

            writer.add_scalar('Loss/train', lossValue, step)
            progress += 1
            showAtSamePlace("progress: {}/{}".format(progress, trainBatches))
            step += 1
        print()

        print("start testing")
        losses = []
        progress = 0
        for img, _ in islice(testSet, testBatches):
            losses.append(testOnBatch(img))
            progress += 1
            showAtSamePlace("progress: {}/{}".format(progress, testBatches))
        print()
        avgLoss = np.mean(losses)
        writer.add_scalar('Loss/test', avgLoss, step)

        saveDir = Path("saves/{}/epoch{}".format(startTime, epoch))
        saveDir.mkdir(parents=True)
        testOnSample(Path("data"), saveDir)

        torch.save(model.state_dict(), '{}/state_dict.pth'.format(str(saveDir)))


trainingLoop()
