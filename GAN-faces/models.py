import torch
import torch.nn as nn


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


class Discriminator(nn.Module):
    def __init__(self, channel_size):
        super(Discriminator, self).__init__()
        self.mlp = nn.Sequential(
            # input = 3 x is x is
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(num_features=32),
            nn.LeakyReLU(0.2, inplace=True),
            # current = 32 x is/2 x is/2
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            # current = 64 x is/4 x is/4
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            # current = 64 x is/8 x is/8
            nn.Flatten(),
            # current = 64*is/8*is/8 = is * is
            nn.Linear(in_features=channel_size * channel_size, out_features=1),
            nn.Dropout1d(p=0.2),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.mlp(x)


class Generator(nn.Module):
    def __init__(self, channel_size):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            # input = 64
            nn.Linear(in_features=64, out_features=channel_size * channel_size),
            # current = 64*is/8*is/8 = is*is
            nn.Unflatten(dim=1, unflattened_size=torch.Size([64, channel_size//8, channel_size//8])),
            # current = 64 x is/8 x is/8
            nn.ConvTranspose2d(in_channels=64, out_channels=512, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            # current = 512 x is/4 x is/4
            nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            # current = 256 x is/2 x is/2
            nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            # current = 128 x is x is
            nn.ConvTranspose2d(in_channels=128, out_channels=3, kernel_size=5, stride=1, padding=2, bias=False),
            nn.Tanh()
            # output = 3 x is x is
        )

    def forward(self, input):
        return self.main(input)


def create_models(channel_size, device):
    d = Discriminator(channel_size).to(device)
    d.apply(weights_init)
    g = Generator(channel_size).to(device)
    g.apply(weights_init)
    return d, g
