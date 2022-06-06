import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm
from pretrained_network_F.projector_F import F_RandomProj
from diffaug import DiffAugment
import torch.nn.functional as F


def conv2d(*args, **kwargs):
    return spectral_norm(nn.Conv2d(*args, **kwargs))


class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.main = nn.Sequential(
            conv2d(in_channels, out_channels, 4, 2, 1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, feat):
        return self.main(feat)


class SingleDisc(nn.Module):
    def __init__(self, nc=None, ndf=None, start_sz=256, end_sz=8, head=None, separable=False, patch=False):
        super().__init__()
        channel_dict = {4: 512, 8: 512, 16: 256,
                        32: 128, 64: 64, 128: 64,
                        256: 32, 512: 16, 1024: 8}
        self.start_sz = start_sz
        nfc = channel_dict
        layers = []
        while start_sz > end_sz:
            layers.append(DownBlock(nfc[start_sz], nfc[start_sz // 2]))
            start_sz = start_sz // 2
        layers.append(conv2d(nfc[end_sz] + 1, 1, 4, 1, 0, bias=False))
        self.main = nn.Sequential(*layers)
        self.body = nn.ModuleList(layers)

    def forward(self, x, normalization=True):
        if normalization:
            std = 0
            for i, m in enumerate(self.body):
                if i == 1:
                    std = torch.std(x, dim=0, keepdim=True).mean((1, 2, 3), keepdim=True)
                if i == len(self.body) - 1:
                    b, c, h, w = x.shape
                    x = torch.cat((x, std.repeat(b, 1, h, w)), dim=1)
                x = m(x)
        else:
            x = self.main(x)
        return x


class MultiScaleD(nn.Module):
    def __init__(
            self,
            channels,
            resolutions,
    ):
        super().__init__()
        self.disc_in_channels = channels
        self.disc_in_res = resolutions
        mini_discriminator = []
        for i, (cin, res) in enumerate(zip(self.disc_in_channels, self.disc_in_res)):
            start_sz = res
            mini_discriminator += [str(i), SingleDisc(nc=cin, start_sz=start_sz, end_sz=8)],
        self.mini_discs = nn.ModuleDict(mini_discriminator)

    def forward(self, features):
        all_logits = []
        for k, disc in self.mini_discs.items():
            all_logits.append(disc(features[k]).view(features[k].size(0), -1))

        all_logits = torch.cat(all_logits, dim=1)
        return all_logits


class ProjectedDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.efficientNet = F_RandomProj()
        self.discriminator = MultiScaleD(
            channels=self.efficientNet.CHANNELS,
            resolutions=self.efficientNet.RESOLUTIONS
        )

    def train(self, mode=True):
        self.efficientNet = self.efficientNet.train(False)
        self.discriminator = self.discriminator.train(mode)
        return self

    def eval(self):
        return self.train(False)

    def forward(self, x):
        x = DiffAugment(x, policy='color,translation,cutout')
        x = F.interpolate(x, 224, mode='bilinear', align_corners=False)
        features = self.efficientNet(x)
        logits = self.discriminator(features)
        return logits


if __name__ == '__main__':
    batch = torch.normal(0, 1, size=(32, 3, 256, 256))
    F_model = F_RandomProj()
    """
    torch.Size([32, 64, 32, 32])
    torch.Size([32, 64, 16, 16])
    torch.Size([32, 128, 8, 8])
    torch.Size([32, 256, 4, 4])
    """
    # features = F_model(batch)

    # discriminator = MultiScaleD(channels=F.CHANNELS, resolutions=F.RESOLUTIONS, num_discs=4)
    # logits = discriminator(features, 3)
    projector = ProjectedDiscriminator()
    logits = projector(batch)
    pass
