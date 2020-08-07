from torchvision.datasets import CelebA

# G_channels_per_block = [512, 512, 512, 512, 256, 128, 64, 32, 16]
# D_channels_per_block = [16, 32, 64, 128, 256, 512, 512, 512, 512]

# Small versions of the model for debugging
G_channels_per_block = [128, 128, 128, 128, 64, 32, 16, 8, 8]
D_channels_per_block = [8, 8, 16, 32, 64, 128, 128, 128, 128]

model_configs = {
    2 ** (i + 1): dict(G_channels=G_channels_per_block[:i],
                       D_channels=D_channels_per_block[-i:])
    for i in range(4, 11)
}


datasets = {
    'celeba': lambda **kwargs: CelebA(**kwargs)
}

