from datetime import datetime

import pytorch_lightning as pl
import argparse
import logging
import copy
import math
import os

import config

from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.utilities.distributed import rank_zero_only

from msg_gan.datasets import TransformDataset
from torchbox.utils.misc import setup_run
from torchbox.utils.logging import setup_logging
from torchbox.utils.gan import ModelWrapper, AveragedModelWrapper

from msg_gan.gan import MultiScaleGenerator, MultiScaleDiscriminator, SimpleExtractBlockBuilder
from msg_gan.utils import get_dataset, init_weights, get_transform
from msg_gan.model import MultiScaleGradientGAN, ProgressBar


is_rank_zero = rank_zero_only.rank == 0


def main(args):
    run_name = (datetime.now().strftime('%Y-%m-%d_%H-%M-%S') + f'-{args.run_name}')
    if is_rank_zero:
        result_dir = setup_run(run_name,
                               create_dirs=['checkpoints', 'samples'],
                               results_dir=args.results_dir or 'results')
        log_file = result_dir / 'log.txt'
    else:
        result_dir = None
        log_file = None

    setup_logging(log_file=log_file)

    logging.info(args)
    logging.info('Loading dataset')

    img_size = args.image_size

    dataset = get_dataset(args.dataset, dataset_kwargs=args.dataset_args)
    dataset = TransformDataset(dataset, transform=get_transform(target_size=(img_size, img_size)))

    logging.info('Building model')

    model_config = config.model_configs[args.image_size]

    extract_block_builder = SimpleExtractBlockBuilder(out_channels=3)

    G = MultiScaleGenerator(args.latent_size, model_config.G_channels, extract_block_builder)
    D = MultiScaleDiscriminator(model_config.D_channels, inject_channel=3)

    G.apply(init_weights)
    D.apply(init_weights)

    if is_rank_zero:
        (result_dir / 'G.txt').write_text(str(G))
        (result_dir / 'D.txt').write_text(str(D))

    if args.use_ema:
        wrapped_model = ModelWrapper(G)
    else:
        wrapped_model = AveragedModelWrapper(copy.deepcopy(G))

    loss_f = config.loss_functions[args.loss](D=D)

    model = MultiScaleGradientGAN(
        args,
        G, D,
        dataset,
        loss_f,
        result_dir / 'samples' if is_rank_zero else None,
        wrapped_model
    )

    if is_rank_zero:
        ckpt_callback = ModelCheckpoint(
            filepath=result_dir / 'checkpoints',
            save_last=True,
            save_top_k=-1,
            verbose=True,
        )
    else:
        ckpt_callback = False

    pbar = ProgressBar(args.log_every)

    logger = True
    if args.tensorboard and is_rank_zero:
        logger = TensorBoardLogger(args.tensorboard_root_dir, name=result_dir.name)

    trainer = pl.Trainer(gpus=args.num_gpus,
                         distributed_backend=args.distributed_backend,
                         checkpoint_callback=ckpt_callback,
                         row_log_interval=10,
                         progress_bar_refresh_rate=args.log_every,
                         logger=logger,
                         callbacks=[pbar],
                         default_root_dir=result_dir,
                         max_epochs=args.epochs,
                         precision=16 if args.fp16 else 32)
    trainer.fit(model)


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset', choices=list(config.datasets.keys()))
    parser.add_argument('--dataset-args', type=lambda s: dict(eval(s)), default={})
    parser.add_argument('--run-name', default='msg-gan')
    parser.add_argument('--results-dir', default=os.environ.get('RESULTS_DIR'))
    parser.add_argument('--image-size', type=int, choices=list(config.model_configs.keys()), default=64)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--g-lr', type=float, default=0.001)
    parser.add_argument('--d-lr', type=float, default=0.001)
    parser.add_argument('--latent-size', type=int, default=512)
    parser.add_argument('--log-every', type=int, default=100)
    parser.add_argument('--sample-every', type=int, default=1000)
    parser.add_argument('--n-sample', type=int, default=8)
    parser.add_argument('--use-ema', action='store_true')
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--loss', choices=list(config.loss_functions.keys()), default='ra_hinge')
    parser.add_argument('--beta1', type=float, default=0.0)
    parser.add_argument('--beta2', type=float, default=0.99)
    parser.add_argument('--tensorboard', action='store_true')
    parser.add_argument('--tensorboard-root-dir', default='tensorboard')
    parser.add_argument('--num-gpus', type=int, default=1)
    parser.add_argument('--fp16', action='store_true')
    parser.add_argument('--distributed-backend')
    parser.add_argument('--fid', action='store_true')
    return parser


if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    main(args)
