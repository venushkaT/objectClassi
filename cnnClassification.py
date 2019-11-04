import numpy as np
# OS
import os
import argparse


import torch
import torchvision
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt
from autoEncorder import Autoencoder
from tensorboardX import SummaryWriter
import torchvision.utils as vutils
from DataSetClass import *
from torch.utils.data import Dataset, DataLoader


def to_img(x):
    x = 0.5 * (x + 1)
    x = x.clamp(0, 1)
    x = x.view(x.size(0), 3, 32, 32)
    return x


num_epochs = 2
batch_size = 128
learning_rate = 1e-3

visualisation = {}


def get_torch_vars(x):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x)

def imshow(img):
    npimg = img.cpu().numpy()
    plt.axis('off')
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        self.conv_layer = nn.Sequential(

            # Conv Layer block 1
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Conv Layer block 2
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(p=0.05),

            # Conv Layer block 3
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.fc_layer = nn.Sequential(
            nn.Dropout(p=0.1),
            nn.Linear(1024, 512),  # 4096, 1024, 512
            nn.ReLU(inplace=True),
            nn.Linear(512, 256),  # 512
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.1),
            nn.Linear(256, 10)
        )

    def forward(self, x):
        x = self.conv_layer(x)
        # flatten
        x = x.view(x.size(0), -1)
        # fc layer
        x = self.fc_layer(x)
        return x


def main():
    parser = argparse.ArgumentParser(description="Train Autoencoder")
    parser.add_argument("--valid", action="store_true", default=False,
                        help="Perform validation only.")
    parser.add_argument("--train", action="store_true", default=False,
                        help="Perform Prediction accuracy")
    args = parser.parse_args()

    # Create model
    kwargs = {'num_workers': 8, 'pin_memory': False}
    dataloader = DataLoader(cat_dog_trainset, batch_size=64, shuffle=True, **kwargs)
    testloader = DataLoader(cat_dog_testset, batch_size=64, shuffle=False, **kwargs)

    classes = ('cat',  'dog')

    ae_model = Autoencoder().float().cuda()
    ae_model.load_state_dict(torch.load("./sim_autoencoder.pth"))

    # load AutoEncoder model
    model = CNN().cuda()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=learning_rate, weight_decay=1e-5)

    if args.valid:
        print("Loading checkpoint...")
        model = CNN().cuda()
        ae_model = Autoencoder().float().cuda()
        model.load_state_dict(torch.load("./classification.pth"))
        ae_model.load_state_dict(torch.load("./sim_autoencoder.pth"))
        correct = 0
        total = 0
        with torch.no_grad():
            for data in testloader:
                img, labels = data
                img = img.view(img.size(0), -1)
                img = get_torch_vars(img)
                labels = get_torch_vars(labels)
                encode, decode = ae_model(img.float())
                encode = encode.view(encode.size(0), 3, 16, 16)
                output = model(encode)
                _, predicted = torch.max(output.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        print('Accuracy of the network on the 10000 test images: %d %%' % (
                100 * correct / total))
        exit(0)



        exit(0)


    if args.train:
        for epoch in range(num_epochs):
            running_loss = 0.0
            for i, data in enumerate(dataloader, 0):
                img, labels = data
                img = get_torch_vars(img)
                labels = get_torch_vars(labels)

                # ============ Forward ============

                img = img.view(img.size(0), -1)
                encode, decode = ae_model(img.float())

                encode = encode.view(encode.size(0), 3, 16, 16)
                output = model(encode)

                loss = criterion(output, labels)
                # ============ Backward ============
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # ============ Logging ============
                running_loss += loss.data

                running_loss += loss.item()
                if i % 2 == 0:  # print every 2000 mini-batches
                    print('[%d, %5d] loss: %.3f' %
                          (epoch + 1, i + 1, running_loss / 2))
                    running_loss = 0.0

                    # needs tensorboard 0.4RC or later
        print('Finished Training')
        torch.save(model.state_dict(), './classification.pth')
        exit(0)




if __name__ == '__main__':
    main()