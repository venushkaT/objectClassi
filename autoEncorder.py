__author__ = 'Venushka Thisara'

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

if not os.path.exists('./mlp_img'):
    os.mkdir('./mlp_img')


def to_img(x):
    x = 0.5 * (x + 1)
    x = x.clamp(0, 1)
    x = x.view(x.size(0), 3, 32, 32)
    return x


num_epochs = 100
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
    img_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    dataset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                           download=True, transform=img_transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=8)

    # for testing purpose
    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=True, transform=img_transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                             shuffle=False, num_workers=8)

    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    model = Autoencoder().cuda()
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=learning_rate, weight_decay=1e-5)

    if args.valid:
        print("Loading checkpoint...")
        model.load_state_dict(torch.load("./sim_autoencoder.pth"))
        dataiter = iter(testloader)
        img, labels = dataiter.next()

        print('GroundTruth: ', ' '.join('%5s' % classes[labels[j]] for j in range(4)))
        imshow(torchvision.utils.make_grid(to_img(img)))

        img = img.view(img.size(0), -1)
        img = get_torch_vars(img)
        encode, decode = model(img)
        # convert into 3x32x 32
        imshow(torchvision.utils.make_grid(to_img(decode).data))
        exit(0)

    if args.train:
        for epoch in range(num_epochs):
            running_loss = 0.0
            for i, data in enumerate(dataloader, 0):
                img, labels = data
                img = img.view(img.size(0), -1)
                img = get_torch_vars(img)
                # ============ Forward ============
                encode, decode = model(img)

                loss = criterion(decode, img)
                # ============ Backward ============
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # ============ Logging ============
                running_loss += loss.data
                if i % 2000 == 1999:
                    print('[%d, %5d] loss: %.3f' %
                          (epoch + 1, i + 1, running_loss / 2000))
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