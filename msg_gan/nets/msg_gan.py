from typing import List

import torch
import torch.nn as nn

from .modules import GeneratorBlock, DiscriminatorBlock, conv, conv_transpose


class ExtractBlockBuilder:
    def __init__(self, out_channels: int):
        self.out_channels = out_channels

    def __call__(self, in_channels: int):
        raise NotImplementedError


class SimpleExtractBlockBuilder(ExtractBlockBuilder):
    def __call__(self, in_channels: int):
        return conv(in_channels, self.out_channels, kernel_size=1)


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


class MultiScaleGenerator(nn.Module):
    def __init__(self,
                 latent_size: int,
                 block_channels: List[int],
                 extract_block_builder: ExtractBlockBuilder,
                 extract_at_indices=None):
        super().__init__()

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
