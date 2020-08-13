import argparse
import logging
import copy
import math
import time
import torch
import config

from torch.utils.data import DataLoader
from torchvision.utils import save_image

from torchbox.utils.misc import get_default_device, setup_run
from torchbox.utils.logging import setup_logging
from torchbox.utils.gan import ModelWrapper, AveragedModelWrapper

from msg_gan.gan import MultiScaleGenerator, MultiScaleDiscriminator, SimpleExtractBlockBuilder
from msg_gan.utils import get_dataset, multi_scale_transform_function, create_multi_scale_image_grid, init_weights


def main(args):
    result_dir = setup_run(args.run_name, create_dirs=['checkpoints', 'samples'])
    setup_logging(log_file=result_dir / 'log.txt')

    logging.info(args)

    device = get_default_device(args.device)

    sample_dir = result_dir / 'samples'
    checkpoint_dir = result_dir / 'checkpoints'

    logging.info('Loading dataset')

    log2_scale = int(math.log2(args.image_size))

    trasnform_f = multi_scale_transform_function(log2_min_scale=2, log2_max_scale=log2_scale)

    dataset = get_dataset(args.dataset, transform=trasnform_f, dataset_kwargs=args.dataset_args)
    batches = DataLoader(dataset, args.batch_size, shuffle=True, pin_memory=True, num_workers=args.num_workers)

    logging.info('Building model')

    model_config = config.model_configs[args.image_size]

    extract_block_builder = SimpleExtractBlockBuilder(out_channels=3)

    G = MultiScaleGenerator(args.latent_size, model_config['G_channels'], extract_block_builder).to(device)
    D = MultiScaleDiscriminator(model_config['D_channels'], inject_channel=3).to(device)

    G.apply(init_weights)
    D.apply(init_weights)

    G.train()
    D.train()

    (result_dir / 'G.txt').write_text(str(G))
    (result_dir / 'D.txt').write_text(str(D))

    if args.use_ema:
        wrapped_model = ModelWrapper(G)
    else:
        wrapped_model = AveragedModelWrapper(copy.deepcopy(G))

    G_opt = torch.optim.Adam(G.parameters(), lr=args.g_lr, betas=(0.5, 0.999))
    D_opt = torch.optim.Adam(D.parameters(), lr=args.d_lr, betas=(0.5, 0.999))

    z_sample = torch.randn(args.n_sample, args.latent_size, 1).to(device)

    loss_f = config.loss_functions[args.loss](D=D)

    logging.info('Starting training')

    try:
        global_step = 0
        for epoch in range(args.epochs):
            g_loss_sum = 0
            d_loss_sum = 0

            start_time = time.time()

            cur_step = 0

            for step, reals in enumerate(batches):
                if isinstance(reals, (list, tuple)):
                    reals = reals[0]

                reals = [sample.to(device) for sample in reals]

                batch_size = reals[0].size(0)

                # Optimize D
                z = torch.randn(batch_size, args.latent_size).to(device)

                fakes = G(z)

                D_opt.zero_grad()

                loss_d = loss_f.loss_d(reals, fakes)
                loss_d.backward()

                D_opt.step()

                # Optimize G
                fakes = G(z)

                G_opt.zero_grad()

                loss_g = loss_f.loss_g(reals, fakes)
                loss_g.backward()

                G_opt.step()

                wrapped_model.update(G)

                g_loss_sum += float(loss_g)
                d_loss_sum += float(loss_d)

                if global_step % args.log_every == 0:
                    cur_step = min(step + 1, args.log_every)
                    batches_per_sec = cur_step / (time.time() - start_time)

                    logging.info(f'[EPOCH {epoch + 1:03d}] [{step:05d} / {len(batches):05d}] ' +
                                 f'loss_d: {d_loss_sum / cur_step:.5f}, loss_g: {g_loss_sum / cur_step:.5f}, ' +
                                 f'batches/s: {batches_per_sec:02.2f}')

                    g_loss_sum = d_loss_sum = 0

                    start_time = time.time()

                if global_step % args.sample_every == 0:
                    samples = G(z_sample)
                    image_grid = create_multi_scale_image_grid(samples)

                    save_image(image_grid, sample_dir / f'{global_step:06d}_fakes.png')

                cur_step += 1
                global_step += 1

            torch.save(G, str(checkpoint_dir / f'G_{global_step:06d}.pth'))
            torch.save(D, str(checkpoint_dir / f'D_{global_step:06d}.pth'))
    except KeyboardInterrupt:
        pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset', choices=list(config.datasets.keys()))
    parser.add_argument('--dataset-args', type=lambda s: dict(eval(s)), default={})
    parser.add_argument('--run-name', default='msg-gan')
    parser.add_argument('--device', type=str)
    parser.add_argument('--image-size', type=int, choices=list(config.model_configs.keys()), default=64)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--g-lr', type=float, default=0.003)
    parser.add_argument('--d-lr', type=float, default=0.003)
    parser.add_argument('--latent-size', type=int, default=512)
    parser.add_argument('--log-every', type=int, default=100)
    parser.add_argument('--sample-every', type=int, default=1000)
    parser.add_argument('--n-sample', type=int, default=4)
    parser.add_argument('--use-ema', action='store_true')
    parser.add_argument('--num-workers', type=int, default=0)
    parser.add_argument('--loss', choices=list(config.loss_functions.keys()), default='wgan_gp')
    parser.add_argument('--beta1', type=float, default=0.5)
    parser.add_argument('--beta2', type=float, default=0.999)
    args = parser.parse_args()

    main(args)
