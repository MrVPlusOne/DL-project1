import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path

dir = Path("saves/Thu Sep 12 21:19:30 2019")
plt.figure(dpi=500)
allDirs = list(dir.glob('epoch*'))


def dirEpoch(dir) -> int:
    numberPart = str(dir.name)[len("epoch"):]
    return int(numberPart)


allDirs.sort(key=dirEpoch)

row = 0
for epochDir in allDirs:
    epochName = str(epochDir)
    col = 0
    for imgFile in epochDir.glob("*.jpg"):
        img = Image.open(imgFile)
        sub: any = plt.subplot(2, len(allDirs), 1 + row + col * len(allDirs))
        sub.axis('off')
        plt.imshow(img)
        col += 1
    row += 1

plt.show()
