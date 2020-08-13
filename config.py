from msg_gan.datasets import FlatImageFolder, IgnoreLabelDataset
from torchbox.losses.gan import StandardGANLoss, WassersteinGANLoss, RelativisticAverageHingeGANLoss
from torchvision.datasets import CelebA
from easydict import EasyDict as edict

# G_channels_per_block = [512, 512, 512, 512, 256, 128, 64, 32, 16]
# D_channels_per_block = [16, 32, 64, 128, 256, 512, 512, 512, 512]

# Small versions of the model for debugging
G_channels_per_block = [128, 128, 128, 128, 64, 32, 16, 8, 8]
D_channels_per_block = [8, 8, 16, 32, 64, 128, 128, 128, 128]

model_configs = {
    2 ** (i + 1): edict(G_channels=G_channels_per_block[:i],
                        D_channels=D_channels_per_block[-i:])
    for i in range(4, 11)
}


datasets = edict({
    'celeba': lambda **kwargs: IgnoreLabelDataset(CelebA(**kwargs)),
    'ffhq': lambda **kwargs: FlatImageFolder(**kwargs)
})

loss_functions = edict({
    'gan_loss': StandardGANLoss,
    'wgan': lambda **kwargs: WassersteinGANLoss(use_gp=False, **kwargs),
    'wgan_gp': lambda **kwargs: WassersteinGANLoss(use_gp=True, **kwargs),
    'ra_hinge': RelativisticAverageHingeGANLoss
})
