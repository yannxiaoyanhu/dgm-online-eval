import torch
import os
import sys

from .dataloaders import get_dataloader


def get_device_and_num_workers(device, num_workers):
    if device is None:
        device = torch.device('cuda' if (torch.cuda.is_available()) else 'cpu')
    else:
        device = torch.device(device)

    if num_workers is None:
        num_avail_cpus = len(os.sched_getaffinity(0))
        num_workers = min(num_avail_cpus, 8)
    else:
        num_workers = num_workers

    return device, num_workers


def get_dataloader_from_path(path, model_transform, num_workers, args, sample_w_replacement=False):
    print(f'Getting DataLoader for path: {path}\n', file=sys.stderr)

    if args.gen_images:
        num_samples = args.n_gensample
    else:
        num_samples = args.nsample
    dataloader = get_dataloader(path, num_samples, args.batch_size, num_workers, seed=args.seed,
                                sample_w_replacement=sample_w_replacement,
                                transform=lambda x: model_transform(x))

    return dataloader
