__author__ = 'Venushka Thisara'

import argparse
import matplotlib.pyplot as plt
import torch
import torchvision
from torch import nn
from torch.autograd import Variable
from DataSetClass import *
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter


def to_img(x):
    """
        x: normalise image
        return: x : un normalise image
    """
    x = 0.5 * (x + 1)
    x = x.clamp(0, 1)
    x = x.view(x.size(0), 3, 32, 32)
    return x


num_epochs = 130
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


def imshow(img):
    """
        image display
       img:  image
       return :void
    """
    npimg = img.cpu().numpy()
    plt.axis('off')
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


"""
    Auto encode model
"""


class Autoencode(nn.Module):
    def __init__(self):
        super(Autoencode, self).__init__()

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

    # Create Auto Encoder model
    model = Autoencode().cuda()
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=learning_rate, weight_decay=1e-5)

    if args.valid:
        print("Loading checkpoint...")
        model.load_state_dict(torch.load("./sim_autoencoder.pth"))
        for i, data in enumerate(dataloader, 0):
            img, labels = data
            imshow(torchvision.utils.make_grid(to_img(img)))
            img = img.view(img.size(0), -1)
            img = get_torch_vars(img)
            encode, decode = model(img)
            imshow(torchvision.utils.make_grid(to_img(decode).data))
            exit(0)

    if args.train:
        writer = SummaryWriter(log_dir="./logs2")
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
                if i % 2000 == 0:  # print every 2000 mini-batches
                    writer.add_scalar("loss", running_loss, epoch)
                    print('[%d, %5d] loss: %.3f' %
                          (epoch + 1, i + 1, running_loss))
                    running_loss = 0.0

        torch.save(model.state_dict(), './sim_autoencoder.pth')
        writer.close()
        exit(0)


if __name__ == '__main__':
    main()