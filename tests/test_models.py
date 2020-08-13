import unittest
import torch

from msg_gan.gan import (
    GeneratorBlock,
    DiscriminatorBlock,
    MultiScaleGenerator,
    SimpleExtractBlockBuilder,
    MultiScaleDiscriminator
)

batch_size = 32
latent_size = 64


def get_latent():
    return torch.randn(batch_size, latent_size)


class TestGeneratorBlock(unittest.TestCase):
    def test_initial_generator_block(self):
        out_ch = 64

        initial_block = GeneratorBlock(latent_size, out_ch, is_initial=True)

        out = initial_block(get_latent())

        self.assertEqual(tuple(out.shape), (batch_size, out_ch, 4, 4))

    def test_generator_block(self):
        in_ch = 32
        out_ch = 64

        block = GeneratorBlock(in_ch, out_ch, is_initial=False)

        in_size = 4
        x = torch.randn(batch_size, in_ch, in_size, in_size)

        out = block(x)

        self.assertEqual(tuple(out.shape), (batch_size, out_ch, in_size * 2, in_size * 2))


class TestDiscriminatorBlock(unittest.TestCase):
    def test_discriminator_block(self):
        in_ch = 64
        out_ch = 32
        in_size = 16

        block = DiscriminatorBlock(in_ch, out_ch, is_final=False)

        x = torch.randn(batch_size, in_ch, in_size, in_size)

        out = block(x)

        self.assertEqual(tuple(out.shape), (batch_size, out_ch, in_size // 2, in_size // 2))

    def test_final_discriminator_block(self):
        in_ch = 64
        out_ch = 32
        in_size = 4

        block = DiscriminatorBlock(in_ch, out_ch, is_final=True)

        x = torch.randn(batch_size, in_ch, in_size, in_size)

        out = block(x)

        self.assertEqual(tuple(out.shape), (batch_size, out_ch, 1, 1))


class TestMultiScaleGradientGAN(unittest.TestCase):
    G_channels = [256, 128, 64, 32]
    D_channels = [32, 64, 128, 256]

    def test_generator(self):
        extract_block_builder = SimpleExtractBlockBuilder(3)

        G = MultiScaleGenerator(latent_size, self.G_channels, extract_block_builder)

        outputs = G(get_latent())

        self.assertEqual(len(outputs), len(self.G_channels))

        for i, out in enumerate(outputs):
            size = 2 ** (i + 2)
            self.assertEqual(tuple(out.shape), (batch_size, 3, size, size))

    def test_discriminator(self):
        D = MultiScaleDiscriminator(self.D_channels, inject_channel=3)

        inputs = [torch.randn(batch_size, 3, 2 ** (i + 2), 2 ** (i + 2)) for i in range(len(self.D_channels))]

        out = D(inputs)

        self.assertEqual(tuple(out.shape), (batch_size, ))

    def test_msg_gan(self):
        extract_block_builder = SimpleExtractBlockBuilder(3)

        G = MultiScaleGenerator(latent_size, self.G_channels, extract_block_builder)
        D = MultiScaleDiscriminator(self.D_channels, inject_channel=3)

        out = D(G(get_latent()))

        self.assertEqual(tuple(out.shape), (batch_size,))
