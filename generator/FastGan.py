"""
Generator architecture and code taken from "Towards Faster and Stabilized GAN Training for High-fidelity Few-shot Image Synthesis" (arxiv.org/abs/2101.04775) and github.com/odegeasslbc/FastGAN-pytorch, respectively.
"""
import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm


def weights_init(m):
    classname = m.__class__.__name__
    if "Conv" in classname:
        try:
            m.weight.data.normal_(0.0, 0.02)
        except:
            pass
    elif "BatchNorm" in classname:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def conv2d(*args, **kwargs):
    return spectral_norm(nn.Conv2d(*args, **kwargs))


def convTranspose2d(*args, **kwargs):
    return spectral_norm(nn.ConvTranspose2d(*args, **kwargs))


def batchNorm2d(*args, **kwargs):
    return nn.BatchNorm2d(*args, **kwargs)


def linear(*args, **kwargs):
    return spectral_norm(nn.Linear(*args, **kwargs))


class GLU(nn.Module):
    def forward(self, x):
        nc = x.size(1)
        assert nc % 2 == 0, 'channels dont divide 2!'
        nc = int(nc / 2)
        return x[:, :nc] * torch.sigmoid(x[:, nc:])


class NoiseInjection(nn.Module):
    def __init__(self):
        super().__init__()

        self.weight = nn.Parameter(torch.zeros(1), requires_grad=True)

    def forward(self, feat, noise=None):
        if noise is None:
            batch, _, height, width = feat.shape
            noise = torch.randn(batch, 1, height, width).to(feat.device)

        return feat + self.weight * noise


class Swish(nn.Module):
    def forward(self, feat):
        return feat * torch.sigmoid(feat)


class SEBlock(nn.Module):
    def __init__(self, ch_in, ch_out):
        super().__init__()

        self.main = nn.Sequential(nn.AdaptiveAvgPool2d(4),
                                  conv2d(ch_in, ch_out, 4, 1, 0, bias=False), Swish(),
                                  conv2d(ch_out, ch_out, 1, 1, 0, bias=False), nn.Sigmoid())

    def forward(self, feat_small, feat_big):
        return feat_big * self.main(feat_small)


class InitLayer(nn.Module):
    def __init__(self, nz, channel):
        super().__init__()
        self.init = nn.Sequential(
            convTranspose2d(nz, channel * 2, 4, 1, 0, bias=False),
            batchNorm2d(channel * 2), GLU())

    def forward(self, noise):
        noise = noise.view(noise.shape[0], -1, 1, 1)
        return self.init(noise)


def UpBlock(in_planes, out_planes):
    block = nn.Sequential(
        nn.Upsample(scale_factor=2, mode='nearest'),
        conv2d(in_planes, out_planes * 2, 3, 1, 1, bias=False),
        batchNorm2d(out_planes * 2), GLU())
    return block


def UpBlockComp(in_planes, out_planes):
    block = nn.Sequential(
        nn.Upsample(scale_factor=2, mode='nearest'),
        conv2d(in_planes, out_planes * 2, 3, 1, 1, bias=False),
        NoiseInjection(),
        batchNorm2d(out_planes * 2), GLU(),
        conv2d(out_planes, out_planes * 2, 3, 1, 1, bias=False),
        NoiseInjection(),
        batchNorm2d(out_planes * 2), GLU()
    )
    return block


class Generator(nn.Module):
    def __init__(self, ngf=64, nz=100, nc=3, im_size=256):
        super(Generator, self).__init__()

        nfc_multi = {4: 16, 8: 8, 16: 4, 32: 2, 64: 2, 128: 1, 256: 0.5, 512: 0.25, 1024: 0.125}
        nfc = {}
        for k, v in nfc_multi.items():
            nfc[k] = int(v * ngf)

        self.im_size = im_size

        self.init = InitLayer(nz, channel=nfc[4])

        self.feat_8 = UpBlockComp(nfc[4], nfc[8])
        self.feat_16 = UpBlock(nfc[8], nfc[16])
        self.feat_32 = UpBlockComp(nfc[16], nfc[32])
        self.feat_64 = UpBlock(nfc[32], nfc[64])
        self.feat_128 = UpBlockComp(nfc[64], nfc[128])
        self.feat_256 = UpBlock(nfc[128], nfc[256])

        self.se_64 = SEBlock(nfc[4], nfc[64])
        self.se_128 = SEBlock(nfc[8], nfc[128])
        self.se_256 = SEBlock(nfc[16], nfc[256])

        self.to_big = conv2d(nfc[im_size], nc, 3, 1, 1, bias=False)

        self.apply(weights_init)

    def forward(self, x):
        feat_4 = self.init(x)
        feat_8 = self.feat_8(feat_4)
        feat_16 = self.feat_16(feat_8)
        feat_32 = self.feat_32(feat_16)

        feat_64 = self.se_64(feat_4, self.feat_64(feat_32))

        feat_128 = self.se_128(feat_8, self.feat_128(feat_64))

        feat_256 = self.se_256(feat_16, self.feat_256(feat_128))

        return self.to_big(feat_256)


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    batch_size = 4
    latent_dim = 100
    z = torch.randn(batch_size, latent_dim)
    G = Generator()
    print(G)
    gen_imgs = G(z).detach()
    # z = torch.Tensor(1, 100)
    gen_imgs = gen_imgs.permute(0, 2, 3, 1)

    img = gen_imgs[0].cpu()

    r_img = (img - torch.min(img)) / (torch.max(img) - torch.min(img))
    z_img = img * 0.5 + 0.5

    z_img = torch.clamp(z_img, min=0.0, max=1.0).numpy()
    r_img = torch.clamp(r_img, min=0.0, max=1.0).numpy()

    plt.imshow(z_img)
    plt.show()

    plt.imshow(r_img)
    plt.show()
