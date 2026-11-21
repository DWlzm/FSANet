#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""FSANet testing / evaluation entry point.

Dataset and split defaults come from the test section of config.py; command line can override them.

Usage:
    # Same-dataset testing (default UIEB test, 90 images)
    python test.py --model_path checkpoints/<run>/unet_simplified_best.pth

    # Cross-dataset: LSUI full testing (856 images)
    python test.py --model_path <ckpt> --dataset lsui --split test

    # Full dataset (merge all splits)
    python test.py --model_path <ckpt> --dataset uieb --split full --save_images

    # Batch evaluation after parameter sweeps
    python test.py --model_path <ckpt> --dataset lsui --split test --quiet
"""
import argparse
import csv
import os
from datetime import datetime

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from tqdm import tqdm

from config import Config
from FSANet import FSANet
from utils.dataset import build_dataset
from utils.metrics import calculate_all_metrics, calculate_composite_score

METRIC_KEYS = ('psnr', 'ssim', 'mse', 'uiqm', 'uciqe', 'pcqi')


def parse_args():
    defaults = Config().test
    p = argparse.ArgumentParser(description='FSANet underwater image enhancement testing/evaluation')
    p.add_argument('--model_path', type=str, required=True, help='model weight path')
    p.add_argument('--dataset', type=str, default=defaults.dataset,
                   choices=['uieb', 'lsui'], help='test dataset (default from config.py)')
    p.add_argument('--root', type=str, default=defaults.root,
                   help='dataset root directory, leave empty to use default path')
    p.add_argument('--split', type=str, default=defaults.split,
                   choices=['train', 'val', 'test', 'full'],
                   help='full means merge all splits of that dataset')
    p.add_argument('--batch_size', type=int, default=defaults.batch_size)
    p.add_argument('--n_filters', type=int, default=Config().model.n_filters)
    p.add_argument('--enc4_layers', type=int, default=Config().model.enc4_layers)
    p.add_argument('--bottleneck_layers', type=int, default=Config().model.bottleneck_layers)
    p.add_argument('--use_abm', dest='use_abm', action='store_true', default=Config().model.use_abm)
    p.add_argument('--no_abm', dest='use_abm', action='store_false')
    p.add_argument('--gpu', type=str, default='0')
    p.add_argument('--output_dir', type=str, default=defaults.output_dir)
    p.add_argument('--save_images', action='store_true', help='save enhanced images and triptych comparisons')
    p.add_argument('--limit', type=int, default=0, help='evaluate only the first N images, 0 means all')
    p.add_argument('--quiet', action='store_true', help='do not print per-image metrics')
    return p.parse_args()


def load_state_dict(path, device):
    obj = torch.load(path, map_location=device)
    if isinstance(obj, dict) and 'model_state_dict' in obj:
        return obj['model_state_dict']
    return obj


def main():
    args = parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_dir = os.path.join(args.output_dir,
                           'test_%s_%s_%s' % (args.dataset, args.split, stamp))
    os.makedirs(out_dir, exist_ok=True)
    img_dir = os.path.join(out_dir, 'images')
    cmp_dir = os.path.join(out_dir, 'comparison')
    if args.save_images:
        os.makedirs(img_dir, exist_ok=True)
        os.makedirs(cmp_dir, exist_ok=True)

    dataset = build_dataset(args.dataset, args.split, root=args.root or None)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    model = FSANet(in_channels=3, out_channels=3, n_filters=args.n_filters,
                   use_abm=args.use_abm, enc4_layers=args.enc4_layers,
                   bottleneck_layers=args.bottleneck_layers).to(device)
    model.load_state_dict(load_state_dict(args.model_path, device))
    model.eval()

    print('Dataset  : %s' % dataset.describe())
    print('Device   : %s | split: %s | samples: %d' % (device, args.split, len(dataset)))
    print('Weights  : %s' % args.model_path)

    vals = {k: [] for k in METRIC_KEYS}
    rows = []
    with torch.no_grad():
        for i, (hazy, clean) in enumerate(tqdm(loader, desc='Evaluating')):
            hazy, clean = hazy.to(device), clean.to(device)
            output, _ = model(hazy)
            output = output.clamp(0, 1)

            if args.save_images:
                save_image(output, os.path.join(img_dir, 'output_%04d.png' % (i + 1)),
                           normalize=False, value_range=(0, 1), antialias=True)
                save_image(torch.cat([hazy, output, clean], dim=0),
                           os.path.join(cmp_dir, 'compare_%04d.png' % (i + 1)), nrow=3,
                           normalize=False, value_range=(0, 1), padding=5, antialias=True)

            for b in range(output.shape[0]):
                idx = i * args.batch_size + b
                tag = dataset.names[idx] if idx < len(dataset.names) else str(idx)
                out_np = np.clip(output[b].cpu().numpy().transpose(1, 2, 0), 0, 1)
                gt_np = np.clip(clean[b].cpu().numpy().transpose(1, 2, 0), 0, 1)
                cur = calculate_all_metrics(gt_np, out_np)
                for k in METRIC_KEYS:
                    vals[k].append(cur[k])
                rows.append({'name': tag, **{k: cur[k] for k in METRIC_KEYS}})
                if not args.quiet:
                    print('Image %d [%s]: PSNR %.4f | SSIM %.4f | MSE %.6f | UIQM %.4f | UCIQE %.4f | PCQI %.4f'
                          % (len(rows), tag, cur['psnr'], cur['ssim'], cur['mse'],
                             cur['uiqm'], cur['uciqe'], cur['pcqi']))

            if args.limit and len(rows) >= args.limit:
                break

    n = len(vals['psnr'])
    avg = {k: float(np.mean(vals[k])) for k in METRIC_KEYS}
    comp = calculate_composite_score(avg)

    print('\n===== Average Metrics (%s/%s, %d images) =====' % (args.dataset, args.split, n))
    print('  PSNR : %.4f (up)' % avg['psnr'])
    print('  SSIM : %.4f (up)' % avg['ssim'])
    print('  MSE  : %.6f (down)' % avg['mse'])
    print('  UIQM : %.4f (up)' % avg['uiqm'])
    print('  UCIQE: %.4f (up)' % avg['uciqe'])
    print('  PCQI : %.4f (up)' % avg['pcqi'])
    print('  Composite score: %.4f' % comp)

    with open(os.path.join(out_dir, 'metrics.txt'), 'w', encoding='utf-8') as f:
        f.write('Dataset: %s\nSplit: %s\nSamples: %d\nWeights: %s\n'
                % (args.dataset, args.split, n, args.model_path))
        for k in METRIC_KEYS:
            f.write('%s: %.4f\n' % (k.upper(), avg[k]))
        f.write('COMPOSITE: %.4f\n' % comp)

    with open(os.path.join(out_dir, 'metrics_per_image.csv'), 'w', newline='',
              encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['name'] + list(METRIC_KEYS))
        writer.writeheader()
        writer.writerows(rows)

    print('\nResults directory: %s' % out_dir)


if __name__ == '__main__':
    main()