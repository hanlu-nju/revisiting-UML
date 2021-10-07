import torch.nn as nn
import torch.nn.functional as F


# Basic ConvNet with Pooling layer
def conv_block(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(),
        nn.MaxPool2d(2)
    )


class ConvNet(nn.Module):

    def __init__(self, x_dim=3, hid_dim=64, z_dim=64, pool=True):
        super().__init__()
        # self.encoder = nn.Sequential(
        #     conv_block(x_dim, hid_dim),
        #     conv_block(hid_dim, hid_dim),
        #     conv_block(hid_dim, hid_dim),
        #     conv_block(hid_dim, z_dim),
        # )
        self.encoder = nn.ModuleList([
            conv_block(x_dim, hid_dim),
            conv_block(hid_dim, hid_dim),
            conv_block(hid_dim, hid_dim),
            conv_block(hid_dim, z_dim),
        ])
        self.pool = pool

    def forward(self, x):

        for l in self.encoder:
            x = l(x)
        if self.pool:
            x = F.adaptive_avg_pool2d(x, 1)
        x = x.view(x.size(0), -1)
        return x

    def pre_forward(self, x, layer=None):
        if layer is None:
            return self.forward(x)
        if layer == 0:
            return x
        for i in range(layer):
            x = self.encoder[i](x)
        return x

    def post_forward(self, x, layer=None):
        if layer is None:
            return x
        if layer < len(self):
            for i in range(layer, len(self)):
                x = self.encoder[i](x)
        if self.pool:
            x = F.adaptive_avg_pool2d(x, 1)
        x = x.view(x.size(0), -1)
        return x

    def __len__(self):
        return len(self.encoder)


def convnet(pool):
    return ConvNet(pool=pool)
