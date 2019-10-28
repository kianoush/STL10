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
from torch.autograd import Variable
import torch.optim as optim
import numpy as np
import cv2
import matplotlib.pyplot as plt

gpu_cuda = torch.cuda.is_available()
print('Cuda', gpu_cuda)

"""
Var
"""
batch_size = 1


"""
Load data
"""

train_datasets = torchvision.datasets.STL10("./Datasets", split='train', transform=transforms.Compose([
                                                                                transforms.Grayscale(),
                                                                                transforms.ToTensor()
                                                                            ]),
                                                                                download=False)
test_datasets = torchvision.datasets.STL10("./Datasets", split='test', transform=transforms.ToTensor(),
                                             download=False)

train_dl = torch.utils.data.DataLoader(train_datasets, batch_size=batch_size, shuffle=True)
test_dl = torch.utils.data.DataLoader(test_datasets, batch_size=batch_size, shuffle=True)

print(train_datasets.data.shape)

"""
Model
"""

class SimpleCNN(nn.Module):
    def __init__(self):
        self.layer1 = nn.Sequential(nn.Conv2d(1,16,3,1,2),
                                    nn.BatchNorm2d(16),
                                    nn.ReLU(),
                                    nn.MaxPool2d(2,2))
        self.layer2 = nn.Sequential(nn.Conv2d(16,32,3,1,2),
                                    nn.BatchNorm2d(32),
                                    nn.ReLU(),
                                    nn.MaxPool2d(2,2))

        self.fc = nn.Linear(25*25*32, 10)

    def forward(self, x):
        out1 = self.layer1(x)
        out2 = self.layer2(out1)
        out2 = out2.view(out2.size(0), -1)
        y = self.fc(out2)



model = SimpleCNN()

if gpu_cuda:
    model = model.cuda()



"""
Loss
"""
criterions = nn.CrossEntropyLoss()

"""
Optim
"""
optim = optim.Adam(model.parameters(), lr=0.001)
lr_sch = torch.optim.lr_scheduler.StepLR(optim, 1, gamma=0.5)



def to_var(x, volatile=False):
    if gpu_cuda:
        x = x.cuda
    return Variable(x, volatile=volatile)

epochs_num = 10
losses = []
for epoch in epochs_num:

    model.train()
    for i, (input, target) in enumerate(train_dl):

        inputs = to_var(input)
        targets = to_var(target)

        # forward
        optim.zero_grad()
        output = model(input)

        # loss
        loss = criterions(output, target)
        losses += [loss.data.item()]

        # backward
        loss.backward()

        # update parameters
        optim.step()

        if (i+1) % 100 == 0:
            print('Train, Epoch [%2d/%2d], Step [%3d/%3d], Loss: %.4f'
                  % (epoch + 1, epochs_num, i + 1, len(train_dl), loss.data.item()))




print("END!")