"""
STL10

The STL-10 dataset is an image recognition dataset
for developing unsupervised feature learning, deep learning,
self-taught learning algorithms.
"""

import torch
import torchvision
import torch.nn as nn
import torchvision.transforms as transforms
import numpy as np
import cv2

"""
Var
"""
batch_size = 32


"""
Load data
"""

train_datasets = torchvision.datasets.STL10("./Datasets", split='train', transform=transforms.ToTensor(),
                                            download=False)
test_datasets = torchvision.datasets.STL10("./Datasets", split='test', transform=transforms.ToTensor(),
                                             download=False)

train_dl = torch.utils.data.DataLoader(train_datasets, batch_size=batch_size, shuffle=True)
test_dl = torch.utils.data.DataLoader(test_datasets, batch_size=batch_size, shuffle=True)

for i in range(100):
    cv2.imshow('kia', np.asarray(train_datasets.data[i,:,:]))
    cv2.waitKey(1000)



print("END!")