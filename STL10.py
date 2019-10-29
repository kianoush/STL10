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
batch_size = 8

"""
Load data
"""

train_datasets = torchvision.datasets.STL10("./Datasets", split='train', transform=transforms.Compose([
                                                                                transforms.RandomHorizontalFlip(),
                                                                                transforms.ColorJitter(0.1, 0.1, 0.1, 0.01),
                                                                                transforms.RandomRotation(5),
                                                                                transforms.ToTensor(),
                                                                                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
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
        super(SimpleCNN, self).__init__()
        self.layer1 = nn.Sequential(nn.Conv2d(3,16,3,1,2),
                                    nn.BatchNorm2d(16),
                                    nn.ReLU(),
                                    nn.MaxPool2d(2,2))
        self.layer2 = nn.Sequential(nn.Conv2d(16,32,3,1,2),
                                    nn.BatchNorm2d(32),
                                    nn.ReLU(),
                                    nn.MaxPool2d(2,2))
        self.layer3 = nn.Sequential(nn.Conv2d(32,32,3,1,2),
                                    nn.BatchNorm2d(32),
                                    nn.ReLU(),
                                    nn.MaxPool2d(2,2))
        self.fc = nn.Linear(25*25*32, 10)

    def forward(self, x):
        out1 = self.layer1(x)
        out2 = self.layer2(out1)
        #out2 = out2.view(out2.size(0), -1)
        out3 = self.layer2(out2)
        out3 = out3.view(out3.size(0), -1)
        y = self.fc(out3)
        return y



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
        x = x.cuda()
    return Variable(x, volatile=volatile)


epochs_num = 10
losses = []
for epoch in range(epochs_num):

    model.train()
    for i, (inputs, targets) in enumerate(train_dl):

        inputs = to_var(inputs)
        targets = to_var(targets)

        # forward
        optim.zero_grad()
        output = model(inputs)

        # Loss
        loss = criterions(output, targets)
        losses += [loss.data.item()]

        # backward
        loss.backward()

        # update parameters
        optim.step()


        if (i+1) % 100 == 0:
            print('Train, Epoch [%2d/%2d], Step [%3d/%3d], Loss: %.4f'
                  % (epoch + 1, epochs_num, i + 1, len(train_dl), loss.data.item()))


"""
train data
"""

model.eval()
corrects = 0
for j, (inputs, targets) in enumerate(train_dl):

    inputs = to_var(inputs)
    targets = to_var(targets)
    output = model(inputs)
    predict = torch.argmax(output, 1)
    corrects += torch.sum(predict==targets)

print(100 * corrects.data.item()/(len(test_datasets)))


print('End!')


"""
Test data
"""

model.eval()
corrects = 0
for j, (inputs, targets) in enumerate(test_dl):

    inputs = to_var(inputs)
    targets = to_var(targets)
    output = model(inputs)
    predict = torch.argmax(output, 1)
    corrects += torch.sum(predict==targets)

print(100 * corrects.data.item()/(len(test_datasets)))


print('End!')
