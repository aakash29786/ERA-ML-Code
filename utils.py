import albumentations as A
from albumentations.pytorch import ToTensorV2
from torchvision import datasets, transforms
import torch
import matplotlib.pyplot as plt
import numpy as np
import torchvision

means = [0.4914, 0.4822, 0.4465]
stds = [0.2470, 0.2435, 0.2616]

train_transforms = A.Compose(
    [
        A.Normalize(mean=means, std=stds, always_apply=True),
        A.PadIfNeeded(min_height=40, min_width=40, always_apply=True),
        A.RandomCrop(height=32, width=32, always_apply=True),
        A.HorizontalFlip(),
        A.CoarseDropout(max_holes=1, max_height=8, max_width=8, min_holes=1, min_height=8, min_width=8, fill_value=means),
        ToTensorV2(),
    ]
)

test_transforms = A.Compose(
    [
        A.Normalize(mean=means, std=stds, always_apply=True),
        ToTensorV2(),
    ]
)

class Cifar10SearchDataset(datasets.CIFAR10):


  def __init__(self, root="~/data", train=True, download=True, transform=None):

      super().__init__(root=root, train=train, download=download, transform=transform)

  def __getitem__(self, index):

      image, label = self.data[index], self.targets[index]

      if self.transform is not None:

          transformed = self.transform(image=image)

          image = transformed["image"]

      return image, label

train = Cifar10SearchDataset(root='./data', train=True,
                                        download=True, transform=train_transforms)
test = Cifar10SearchDataset(root='./data', train=False,
                                       download=True, transform=test_transforms)

def mydataloader():

  SEED = 1

  # CUDA?
  cuda = torch.cuda.is_available()
  print("CUDA Available?", cuda)

  # For reproducibility
  torch.manual_seed(SEED)

  if cuda:
      torch.cuda.manual_seed(SEED)

  # dataloader arguments - something you'll fetch these from cmdprmt
  dataloader_args = dict(shuffle=True, batch_size=512, num_workers=0, pin_memory=True) if cuda else dict(shuffle=True, batch_size=64)

  # train dataloader
  train_loader = torch.utils.data.DataLoader(train, **dataloader_args)

  # test dataloader
  test_loader = torch.utils.data.DataLoader(test, **dataloader_args)    

  return train_loader, test_loader

def getDataiter():
  train_loader, test_loader = mydataloader()
  dataiter = iter(train_loader)
  images, labels = next(dataiter)
  return images, labels

def imshow(img):
  img = img / 2 + 0.5     # unnormalize
  npimg = img.numpy()
  plt.imshow(np.transpose(npimg, (1, 2, 0)))
  plt.show()
