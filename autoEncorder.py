__author__ = 'Venushka Thisara'

import argparse
# OS
import os

import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.autograd import Variable
from DataSetClass import *
from torch.utils.data import Dataset, DataLoader


def to_img(x):
    x = 0.5 * (x + 1)
    x = x.clamp(0, 1)
    x = x.view(x.size(0), 3, 32, 32)
    return x


num_epochs = 20
batch_size = 128
learning_rate = 1e-3


def get_torch_vars(x):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x)

def imshow(img):
    npimg = img.cpu().numpy()
    plt.axis('off')
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(3 * 32 * 32, 3 * 28 * 28),
            nn.ReLU(True),
            nn.Linear(3 * 28 * 28, 3 * 24 * 24),
            nn.ReLU(True),
            nn.Linear(3 * 24 * 24, 3 * 20 * 20),
            nn.ReLU(True),
            nn.Linear(3 * 20 * 20, 3 * 16 * 16))

        self.decoder = nn.Sequential(
            nn.Linear(3 * 16 * 16, 3 * 20 * 20),
            nn.ReLU(True),
            nn.Linear(3 * 20 * 20, 3 * 24 * 24),
            nn.ReLU(True),
            nn.Linear(3 * 24 * 24, 3 * 28 * 28),
            nn.ReLU(True), nn.Linear(3 * 28 * 28, 3 * 32 * 32), nn.Tanh())

    def forward(self, x):
        encode = self.encoder(x)
        decode = self.decoder(encode)
        return encode, decode


def main():
    parser = argparse.ArgumentParser(description="Train Autoencoder")
    parser.add_argument("--valid", action="store_true", default=False,
                        help="Perform validation only.")
    parser.add_argument("--train", action="store_true", default=False,
                        help="Perform Prediction accuracy")

    parser.add_argument("--freeze", action="store_true", default=False,
                        help="freeze encoder layers")

    args = parser.parse_args()

    # Create model
    kwargs = {'num_workers': 8, 'pin_memory': False}
    dataloader = DataLoader(cat_dog_trainset, batch_size=64, shuffle=True, **kwargs)
    testloader = DataLoader(cat_dog_testset, batch_size=64, shuffle=False, **kwargs)


    model = Autoencoder().float().cuda()
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=learning_rate, weight_decay=1e-5)

    #https://gist.github.com/Miladiouss/6ba0876f0e2b65d0178be7274f61ad2f

    if args.valid:
        print("Loading checkpoint...")
        model.load_state_dict(torch.load("./sim_autoencoder.pth"))

        correct = 0
        total = 0
        with torch.no_grad():
            for data in testloader:
                img, labels = data

                img = img.view(img.size(0), -1)

                img = get_torch_vars(img)
                labels = get_torch_vars(labels)
                encode, decode = model(img.float())
                _, predicted = torch.max(encode.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        print('Accuracy of the network on the 10000 test images: %d %%' % (
                100 * correct / total))
        exit(0)

    if args.train:
        for epoch in range(num_epochs):
            running_loss = 0.0
            for i, data in enumerate(dataloader, 0):
                img, labels = data

                img = img.view(img.size(0), -1)

                img = get_torch_vars(img)
                # ============ Forward ============

                encode, decode = model(img.float())

                loss = criterion(decode.float(), img.float())
                # ============ Backward ============
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # ============ Logging ============
                running_loss += loss.data
                if i % 2 == 0:
                    print('[%d, %5d] loss: %.3f' %
                          (epoch + 1, i + 1, running_loss / 2))
                    running_loss = 0.0

        torch.save(model.state_dict(), './sim_autoencoder.pth')
        exit(0)
    if args.freeze:
        model.load_state_dict(torch.load("./sim_autoencoder.pth"))
        encodeLayer = nn.Sequential(*list(model.children())[0:1])
        for name, param in encodeLayer.named_parameters():
            param.requires_grad = False

        exit(0)


    # Print all the parameters in model
    model.load_state_dict(torch.load("./sim_autoencoder.pth"))
    for name, param in model.named_parameters():
        print('name: ', name)
        print(type(param))
        print('param.shape: ', param.shape)
        print('param.requires_grad: ', param.requires_grad)
        print('=====')


if __name__ == '__main__':
    main()