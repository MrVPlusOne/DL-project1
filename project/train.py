import torch
import torchvision as tv
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from project.models import Encoder, Decoder


# load data

def getDataSet(root: str, train: bool) -> DataLoader:
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.ToTensor(),
        transforms.Normalize(
            (0.4914, 0.4822, 0.4466), (0.247, 0.243, 0.261))])
    dataSet = tv.datasets.CIFAR10(root, train=train, transform=transform,
                                  download=True)
    loader = DataLoader(dataSet, batch_size=32,
                        shuffle=True, num_workers=4)
    return loader


root = "../data/CIFAR/"
trainSet: DataLoader = getDataSet(root, True)
testSet: DataLoader = getDataSet(root, False)

# initialize model
encoder: Encoder = Encoder()
decoder: Decoder = Decoder()

img0, _ = trainSet.dataset[0]

encoder.forward(img0.reshape(1, 3, 256, 256)).shape
