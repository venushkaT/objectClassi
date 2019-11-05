__author__ = 'Venushka Thisara'

import argparse
import torch

from torch import nn
from torch.autograd import Variable
from autoEncorder import Autoencode
from torch.utils.tensorboard import SummaryWriter

from DataSetClass import *
from torch.utils.data import Dataset, DataLoader

num_epochs = 25
learning_rate = 1e-3


def get_torch_vars(x):
    """
           convert tensor to cuda tensor
           x: torch tensor
           return: x : cuda torch tensor
       """
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x)


"""
    Convolutional neural network model 
"""

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
            nn.Linear(256, 7)
        )

    def forward(self, x):
        x = self.conv_layer(x)
        # flatten
        x = x.view(x.size(0), -1)
        # fc layer
        x = self.fc_layer(x)
        return x


def main():
    parser = argparse.ArgumentParser(description="Train Auto Encode")
    parser.add_argument("--valid", action="store_true", default=False,
                        help="Perform validation only.")
    parser.add_argument("--train", action="store_true", default=False,
                        help="Perform Prediction accuracy")

    args = parser.parse_args()

    # Create Auto Encoder
    ae_model = Autoencode().cuda()
    ae_model.load_state_dict(torch.load("./sim_autoencoder.pth"))

    # load AutoEncoder model
    model = CNN().cuda()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=learning_rate, weight_decay=1e-5)

    if args.valid:
        print("Loading checkpoint...")
        model = CNN().cuda()
        ae_model = Autoencode().cuda()
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
                encode, decode = ae_model(img)
                encode = encode.view(encode.size(0), 3, 16, 16)
                output = model(encode)
                _, predicted = torch.max(output.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        print('Accuracy of the network on the 10000 test images: %d %%' % (
                100 * correct / total))
        exit(0)

    if args.train:
        writer = SummaryWriter(log_dir="./logs")

        for epoch in range(num_epochs):
            running_loss = 0.0
            correct = 0
            total = 0
            for i, data in enumerate(dataloader, 0):
                img, labels = data
                img = get_torch_vars(img)
                labels = get_torch_vars(labels)

                # ============ Forward ============

                img = img.view(img.size(0), -1)
                encode, decode = ae_model(img)

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
                if i % 2000 == 0:  # print every 2000 mini-batches

                    writer.add_scalar("loss", running_loss, epoch)

                    _, predicted = torch.max(output.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()

                    accuracy = 100 * (correct / total)

                    writer.add_scalar("accuracy", accuracy, epoch)

                    print('[%d, %5d] loss: %.3f Accu: %.3f' %
                          (epoch + 1, i + 1, running_loss, accuracy))
                    running_loss = 0.0

        print('Finished Training')
        torch.save(model.state_dict(), './classification.pth')
        writer.close()
        exit(0)


if __name__ == '__main__':
    main()
