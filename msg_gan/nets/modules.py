import torch
import torch.nn as nn

from torchbox.layers import EqualizedConv1d, EqualizedConvTranspose1d, PixelwiseNorm, MinibatchStdDev, EqualizedConv2d


def activation():
    return nn.LeakyReLU(0.2)


def conv(*args, **kwargs):
    #return nn.Conv2d(*args, **kwargs)
    return EqualizedConv2d(*args, **kwargs)


def conv_transpose(*args, **kwargs):
    return nn.ConvTranspose2d(*args, **kwargs)


# class GeneratorBlock(nn.Module):
#     def __init__(self, in_channels, out_channels, attn=False, is_initial=False):
#         super().__init__()
#
#         blocks = []
#         if is_initial:
#             blocks.append(conv_transpose(in_channels, out_channels, kernel_size=4))
#         else:
#             blocks.append(conv_transpose(in_channels, out_channels, kernel_size=4, stride=2, padding=1))
#
#         blocks += [
#             activation(),
#             nn.BatchNorm1d(out_channels),
#
#             conv(out_channels, out_channels, kernel_size=3, padding=1),
#             activation(),
#             nn.BatchNorm1d(out_channels),
#
#             conv(out_channels, out_channels, kernel_size=3, padding=1),
#             activation(),
#             nn.BatchNorm1d(out_channels),
#         ]
#
#         self.blocks = nn.Sequential(*blocks)
#         if attn:
#             self.attn = SelfAttention(out_channels)
#         else:
#             self.attn = nn.Identity()
#
#     def forward(self, input):
#         out = self.blocks(input)
#         out = self.attn(out)
#         return out


class GeneratorBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, is_initial=False, norm=PixelwiseNorm):
        super().__init__()

        self.is_initial = is_initial

        if is_initial:
            blocks = [
                conv_transpose(in_channels, out_channels, kernel_size=4),
                activation(),

                conv(out_channels, out_channels, kernel_size=3, padding=1),
                activation(),
                norm()
            ]
        else:
            blocks = [
                nn.Upsample(scale_factor=2.0, mode='bilinear'),

                conv(in_channels, out_channels, kernel_size=3, padding=1),
                activation(),
                norm(),

                conv(out_channels, out_channels, kernel_size=3, padding=1),
                activation(),
                norm()
            ]

        self.blocks = nn.Sequential(*blocks)

    def forward(self, input: torch.Tensor):
        if self.is_initial and input.ndim != 4:
            input = input.view((-1, input.size(1), 1, 1))
        return self.blocks(input)


class DiscriminatorBlock(nn.Module):
    def __init__(self, in_channels, out_channels, is_final=False, use_minibatch_std_dev=True):
        super().__init__()

        blocks = []

        if use_minibatch_std_dev:
            blocks.append(MinibatchStdDev())
            in_channels += 1

        blocks += [
            conv(in_channels, in_channels, kernel_size=3, padding=1),
            activation(),
        ]

        if is_final:
            blocks += [
                conv(in_channels, out_channels, kernel_size=4),
                activation(),
            ]
        else:
            blocks += [
                conv(in_channels, out_channels, kernel_size=3, padding=1),
                activation(),

                nn.AvgPool2d(kernel_size=2)
            ]

        self.blocks = nn.Sequential(*blocks)

    def forward(self, input):
        return self.blocks(input)


