import torch
import torch.nn as nn

from typing import List
from torchbox.layers import MinibatchStdDev, PixelwiseNorm, EqualizedConv2d, EqualizedConvTranspose2d


def g_activation():
    return nn.LeakyReLU(0.2)

def d_activation():
    return nn.LeakyReLU(0.2)


def conv(*args, **kwargs):
    return EqualizedConv2d(*args, **kwargs)


def conv_transpose(*args, **kwargs):
    return EqualizedConvTranspose2d(*args, **kwargs)


class ExtractBlockBuilder:
    def __init__(self, out_channels: int):
        self.out_channels = out_channels

    def __call__(self, in_channels: int):
        raise NotImplementedError


class SimpleExtractBlockBuilder(ExtractBlockBuilder):
    def __call__(self, in_channels: int, norm=PixelwiseNorm):
        return nn.Sequential(
            conv(in_channels, in_channels // 2, kernel_size=3, padding=1),
            g_activation(),
            norm(),

            conv(in_channels // 2, self.out_channels, kernel_size=1)
        )


# class BaseInjectBlock(nn.Module):
#     def __init__(self, out_channels):
#         super().__init__()
#         self.out_channels = out_channels
#
#     def forward(self, input, prev_layer_output):
#         raise NotImplementedError
#
#
# class SimpleInjectBlock(BaseInjectBlock):
#     def __init__(self, image_channels, prev_layer_channels):
#         super().__init__(image_channels + prev_layer_channels)
#
#     def forward(self, input, prev_layer_output):
#         if not prev_layer_output:
#             return input
#         out = torch.cat([input, prev_layer_output])
#         assert out.size(1) == self.out_channels
#
#
# class LinCatInjectBlock(BaseInjectBlock):
#     def __init__(self, image_channels, prev_layer_channels):
#         super().__init__(prev_layer_channels + prev_layer_channels // 2)
#         self.conv = conv(image_channels, prev_layer_channels // 2, kernel_size=1)
#
#     def forward(self, input, prev_layer_output):
#         input = self.conv(input)
#         return torch.cat([input, prev_layer_output])
#
#
# class CatLinInjectBlock(BaseInjectBlock):
#     def __init__(self, image_channels, prev_layer_channels):
#         super().__init__(prev_layer_channels + prev_layer_channels // 2)
#         self.conv = conv(image_channels + prev_layer_channels, self.out_channels, kernel_size=1)
#
#     def forward(self, input, prev_layer_output):
#         return self.conv(torch.cat([input, prev_layer_output]))
#
#
# class InjectBlockBuilder:
#     def __init__(self, image_channels: int):
#         self.image_channels = image_channels
#
#     def __call__(self, prev_layer_channels: int) -> BaseInjectBlock:
#         raise NotImplementedError
#
#
# class SimpleInjectBlockBuilder(InjectBlockBuilder):
#     def __call__(self, prev_layer_channels: int) -> BaseInjectBlock:
#         return SimpleInjectBlock(self.image_channels, prev_layer_channels)
#
#
# class LinCatInjectBlockBuilder(InjectBlockBuilder):
#     def __call__(self, prev_layer_channels: int) -> BaseInjectBlock:
#         return LinCatInjectBlock(self.image_channels, prev_layer_channels)
#
#
# class CatLinInjectBlockBuilder(InjectBlockBuilder):
#     def __call__(self, prev_layer_channels: int) -> BaseInjectBlock:
#         return CatLinInjectBlock(self.image_channels, prev_layer_channels)



class GeneratorBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, is_initial=False, norm=PixelwiseNorm):
        super().__init__()

        self.is_initial = is_initial

        if is_initial:
            blocks = [
                conv_transpose(in_channels, out_channels, kernel_size=4),
                g_activation(),

                conv(out_channels, out_channels, kernel_size=3, padding=1),
                g_activation(),
                norm()
            ]
        else:
            blocks = [
                nn.Upsample(scale_factor=2.0, mode='nearest'),

                conv(in_channels, out_channels, kernel_size=3, padding=1),
                g_activation(),
                norm(),

                conv(out_channels, out_channels, kernel_size=3, padding=1),
                g_activation(),
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
            conv(in_channels, out_channels, kernel_size=3, padding=1),
            d_activation(),
        ]

        if is_final:
            blocks += [
                conv(out_channels, out_channels, kernel_size=4),
                d_activation(),
            ]
        else:
            blocks += [
                conv(out_channels, out_channels, kernel_size=3, padding=1),
                d_activation(),

                nn.AvgPool2d(kernel_size=2)
            ]

        self.blocks = nn.Sequential(*blocks)

    def forward(self, input):
        return self.blocks(input)


class MultiScaleGenerator(nn.Module):
    def __init__(self,
                 latent_size: int,
                 block_channels: List[int],
                 extract_block_builder: ExtractBlockBuilder,
                 extract_at_indices=None):
        super().__init__()

        self.depth = len(block_channels)

        if not extract_at_indices:
            extract_at_indices = list(range(len(block_channels)))

        channels = [latent_size] + block_channels

        self.blocks = nn.ModuleList()
        self.extract_layers = nn.ModuleList()

        for i, (in_channels, out_channels) in enumerate(zip(channels[:-1], channels[1:])):
            self.blocks.append(GeneratorBlock(in_channels, out_channels, is_initial=i == 0))
            if i in extract_at_indices:
                self.extract_layers.append(extract_block_builder(channels[i + 1]))
            else:
                self.extract_layers.append(None)

    def forward(self, input):
        out = input
        outputs = []
        for block, extract_layer in zip(self.blocks, self.extract_layers):
            out = block(out)
            if extract_layer is not None:
                outputs.append(extract_layer(out))
        return outputs


class MultiScaleDiscriminator(nn.Module):
    def __init__(self,
                 block_channels: List[int],
                 inject_channel: int,
                 inject_at_indices=None):
        super().__init__()

        self.depth = len(block_channels)

        if not inject_at_indices:
            inject_at_indices = list(range(len(block_channels)))

        self.inject_at_indices = inject_at_indices

        channels = block_channels + [block_channels[-1]]

        self.blocks = nn.ModuleList()
        self.inject_layers = nn.ModuleList()

        for i, (in_channels, out_channels) in enumerate(zip(channels[:-1], channels[1:])):
            if i in inject_at_indices:
                self.inject_layers.append(conv(inject_channel,
                                               in_channels if not i else in_channels // 2,
                                               kernel_size=1))
                in_channels = (0 if not i else in_channels // 2) + in_channels
            else:
                self.inject_layers.append(None)

            self.blocks.append(DiscriminatorBlock(in_channels, out_channels, is_final=i + 1 == len(block_channels)))

        self.linear = nn.Linear(channels[-1], 1)

    def forward(self, inputs):
        assert len(inputs) == len(self.inject_at_indices)

        inputs = list(inputs)[::-1]

        out = None
        for i, (block, inject_layer) in enumerate(zip(self.blocks, self.inject_layers)):
            if inject_layer is not None:
                input = inject_layer(inputs.pop(0))
                if out is not None:
                    input = torch.cat([out, input], dim=1)
                out = input
            else:
                assert out is not None

            out = block(out)

        assert len(inputs) == 0

        out = out.squeeze()
        out = self.linear(out)
        out = out.view(-1)

        return out
