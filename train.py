"""FSANet training entry point.

All hyperparameters are centralized in config.py: this file only handles parsing overrides, building the pipeline, and running the loop.
Common examples:
    python train.py --epochs 2 --batch_size 2 --gpu_ids 0
    python train.py --alpha 0.4 --beta 0.4 --gamma 0.1 --lambda_percep 0.2
    python train.py --config exps/loss_a04_b03.json
"""
import argparse
import json
import os
import random
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, DistributedSampler
from torch.nn.parallel import DataParallel, DistributedDataParallel
from torch.distributed import init_process_group, destroy_process_group
from torch.cuda.amp import GradScaler
from torchvision.utils import save_image
from tqdm import tqdm

from config import Config
from FSANet import FSANet
from utils.losses import MultiSpectralLoss
from utils.dataset import build_dataset
from utils.metrics import calculate_all_metrics, calculate_composite_score
from utils.model_config import get_metric_weights, validate_weights


# ------------------------------------------------------------------ runtime environment
def parse_args():
    p = argparse.ArgumentParser(
        description='FSANet underwater image enhancement training (hyperparameters in config.py)',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # config source
    p.add_argument('--config', type=str, default='', help='json config file path')
    p.add_argument('--override', nargs='*', default=[],
                   help='dotted overrides, e.g. loss.alpha=0.4 train.batch_size=4')

    # common hyperparameter shortcuts (use config.py defaults if not provided)
    p.add_argument('--epochs', type=int)
    p.add_argument('--batch_size', type=int)
    p.add_argument('--lr', type=float)
    p.add_argument('--weight_decay', type=float)
    p.add_argument('--alpha', type=float, help='gradient loss weight')
    p.add_argument('--beta', type=float, help='low-frequency loss weight')
    p.add_argument('--gamma', type=float, help='color-correlation loss weight')
    p.add_argument('--lambda_percep', type=float, help='perceptual loss weight')
    p.add_argument('--n_filters', type=int)
    p.add_argument('--enc4_layers', type=int)
    p.add_argument('--bottleneck_layers', type=int)
    p.add_argument('--seed', type=int)
    p.add_argument('--sched_patience', type=int)
    p.add_argument('--data_dir', type=str)
    p.add_argument('--dataset', type=str, help='training set name: uieb / lsui')
    p.add_argument('--train_split', type=str, help='train / val / test / full')
    p.add_argument('--val_dataset', type=str, help='validation set name used for monitoring during training')
    p.add_argument('--val_root', type=str, help='validation set root directory, leave empty to use data_dir')
    p.add_argument('--val_split', type=str, help='validation split: train / val / test / full')
    p.add_argument('--run_name', type=str)
    p.add_argument('--amp', dest='amp', action='store_true', default=None)
    p.add_argument('--no_amp', dest='amp', action='store_false')

    # runtime environment (not hyperparameters)
    p.add_argument('--parallel', type=str, default='none',
                   choices=['none', 'dataparallel', 'distributed'])
    p.add_argument('--world_size', type=int, default=1)
    p.add_argument('--rank', type=int, default=0)
    p.add_argument('--dist_url', type=str, default='tcp://localhost:10001')
    p.add_argument('--gpu_ids', type=str, default='0', help='comma-separated, e.g. 0 or 0,1')
    return p.parse_args()


def build_config(args):
    cfg = Config.from_file(args.config) if args.config else Config()
    cfg.apply_overrides(args.override)
    cli_map = {
        'train.epochs': args.epochs,
        'train.batch_size': args.batch_size,
        'train.amp': args.amp,
        'train.seed': args.seed,
        'optim.lr': args.lr,
        'optim.weight_decay': args.weight_decay,
        'loss.alpha': args.alpha,
        'loss.beta': args.beta,
        'loss.gamma': args.gamma,
        'loss.lambda_percep': args.lambda_percep,
        'model.n_filters': args.n_filters,
        'model.enc4_layers': args.enc4_layers,
        'model.bottleneck_layers': args.bottleneck_layers,
        'scheduler.patience': args.sched_patience,
        'data.root': args.data_dir,
        'data.dataset': args.dataset,
        'data.train_split': args.train_split,
        'data.val_dataset': args.val_dataset,
        'data.val_root': args.val_root,
        'data.val_split': args.val_split,
        'output.run_name': args.run_name,
    }
    for key, val in cli_map.items():
        if val is not None:
            cfg.set_value(key, val)
    return cfg


def set_seed(seed, deterministic=False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def init_distributed(args):
    if args.parallel == 'distributed':
        init_process_group(backend='nccl', init_method=args.dist_url,
                           world_size=args.world_size, rank=args.rank)
        torch.cuda.set_device(args.rank)
        return True
    return False


# ------------------------------------------------------------------ global initialization
args = parse_args()
cfg = build_config(args)

if args.gpu_ids:
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_ids
    gpu_ids = [int(i) for i in args.gpu_ids.split(',') if i.strip() != '']
else:
    gpu_ids = list(range(torch.cuda.device_count()))
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

is_distributed = init_distributed(args)
set_seed(cfg.train.seed, cfg.train.deterministic)

# hyperparameter extraction (subsequent functions keep these names, logic identical to the original script)
epochs = cfg.train.epochs
batch_size = cfg.train.batch_size
lr = cfg.optim.lr
data_dir = cfg.data.root
log_interval = cfg.train.log_interval

timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
run_name = cfg.output.run_name.strip() or '%s_%s' % (cfg.output.prefix, timestamp)
checkpoint_dir = os.path.join(cfg.output.root, run_name)
if args.rank == 0:
    os.makedirs(checkpoint_dir, exist_ok=True)

# dataset and loaders (training/validation can be from different datasets)
val_data_dir = cfg.data.val_root.strip() or data_dir
train_dataset = build_dataset(cfg.data.dataset, cfg.data.train_split, root=data_dir)
val_dataset = build_dataset(cfg.data.val_dataset, cfg.data.val_split, root=val_data_dir)

if args.rank == 0:
    print('Training data: %s' % train_dataset.describe())
    print('Validation data: %s' % val_dataset.describe())
    in_sample = (cfg.data.dataset.lower() == cfg.data.val_dataset.lower()
                 and os.path.abspath(data_dir) == os.path.abspath(val_data_dir)
                 and (cfg.data.train_split.lower() == 'full'
                      or cfg.data.val_split.lower() in
                      ([cfg.data.train_split.lower()] if cfg.data.train_split.lower() != 'full'
                       else list(train_dataset.subset_counts.keys()))))
    if in_sample:
        print('Note: validation set comes from inside the training set, used only to observe convergence, not as a generalization conclusion')

if is_distributed:
    train_sampler = DistributedSampler(train_dataset, shuffle=True)
    val_sampler = DistributedSampler(val_dataset, shuffle=False)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler,
                              num_workers=cfg.data.num_workers, pin_memory=cfg.data.pin_memory)
    val_loader = DataLoader(val_dataset, batch_size=cfg.data.val_batch_size, sampler=val_sampler,
                            num_workers=cfg.data.num_workers, pin_memory=cfg.data.pin_memory)
else:
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              num_workers=cfg.data.num_workers, pin_memory=cfg.data.pin_memory)
    val_loader = DataLoader(val_dataset, batch_size=cfg.data.val_batch_size, shuffle=False,
                            num_workers=cfg.data.num_workers, pin_memory=cfg.data.pin_memory)

# model
model = FSANet(in_channels=cfg.model.in_channels, out_channels=cfg.model.out_channels,
               n_filters=cfg.model.n_filters, use_abm=cfg.model.use_abm,
               enc4_layers=cfg.model.enc4_layers,
               bottleneck_layers=cfg.model.bottleneck_layers).to(device)

if cfg.model.load_pretrained_if_exists and cfg.model.pretrained_path \
        and os.path.exists(cfg.model.pretrained_path):
    try:
        model.load_state_dict(torch.load(cfg.model.pretrained_path, map_location=device))
        if args.rank == 0:
            print('Loaded pretrained model: %s' % cfg.model.pretrained_path)
    except Exception as e:
        if args.rank == 0:
            print('Failed to load pretrained model: %s' % e)

# parallelism
if args.parallel == 'dataparallel' and len(gpu_ids) > 1:
    model = DataParallel(model, device_ids=gpu_ids)
    if args.rank == 0:
        print('Using DataParallel, GPU count: %d' % len(gpu_ids))
elif args.parallel == 'distributed':
    model = DistributedDataParallel(model, device_ids=[args.rank])
    if args.rank == 0:
        print('Using DistributedDataParallel, GPU count: %d' % args.world_size)

# optimizer and scheduler
if cfg.optim.name.lower() != 'adamw':
    raise ValueError('Currently only the adamw optimizer is supported, got: %s' % cfg.optim.name)
optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=cfg.optim.weight_decay,
                        betas=tuple(cfg.optim.betas))

if cfg.scheduler.name.lower() not in ('reduce_on_plateau', 'plateau'):
    raise ValueError('Currently only the reduce_on_plateau scheduler is supported, got: %s' % cfg.scheduler.name)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode=cfg.scheduler.mode, factor=cfg.scheduler.factor,
    patience=cfg.scheduler.patience, min_lr=cfg.scheduler.min_lr)

scaler = GradScaler(enabled=cfg.train.amp)

# loss (all four component weights come from config)
multispectral_loss = MultiSpectralLoss(
    alpha=cfg.loss.alpha, beta=cfg.loss.beta, gamma=cfg.loss.gamma,
    lambda_percep=cfg.loss.lambda_percep, percep_layers=cfg.loss.percep_layers,
    vgg_pretrained=cfg.loss.vgg_pretrained, device=device)

if args.rank == 0:
    print('Evaluation metric module loaded')


# ------------------------------------------------------------------ train/validate
def train_epoch(epoch):
    model.train()
    total_loss = 0
    valid_batches = 0

    if is_distributed:
        train_loader.sampler.set_epoch(epoch)

    if args.rank == 0:
        pbar = tqdm(train_loader, desc='Epoch %d/%d [train]' % (epoch, epochs), leave=True)
    else:
        pbar = train_loader

    for batch_idx, (hazy, clean) in enumerate(pbar):
        hazy, clean = hazy.to(device), clean.to(device)

        optimizer.zero_grad()

        output, _ = model(hazy)
        loss = multispectral_loss(output, clean)

        if torch.isnan(loss) or torch.isinf(loss):
            if args.rank == 0:
                pbar.write('Warning: invalid loss value detected in batch %d, skipping this batch.' % batch_idx)
            continue

        scaler.scale(loss).backward()
        if cfg.train.grad_clip and cfg.train.grad_clip > 0:
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), cfg.train.grad_clip)
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        valid_batches += 1

        if args.rank == 0:
            avg_loss = total_loss / valid_batches
            pbar.set_postfix({'Loss': '%.6f' % avg_loss})

    avg_loss = total_loss / valid_batches if valid_batches > 0 else float('nan')
    if args.rank == 0:
        print('Training complete - Epoch %d, average loss: %.6f' % (epoch, avg_loss))
    return avg_loss


def validate(epoch):
    model.eval()
    val_loss = 0
    valid_samples = 0

    if args.rank == 0:
        pbar = tqdm(val_loader, desc='Epoch %d/%d [validate]' % (epoch, epochs), leave=True)
    else:
        pbar = val_loader

    with torch.no_grad():
        for batch_idx, (hazy, clean) in enumerate(pbar):
            hazy, clean = hazy.to(device), clean.to(device)

            output, _ = model(hazy)
            output = output.clamp(0, 1)
            loss = multispectral_loss(output, clean)
            if torch.isnan(loss) or torch.isinf(loss):
                if args.rank == 0:
                    pbar.write('Validation warning: NaN or Inf loss detected! sample %d' % batch_idx)
                continue

            val_loss += loss.item()
            valid_samples += 1

            if args.rank == 0:
                avg_loss = val_loss / valid_samples if valid_samples > 0 else float('nan')
                pbar.set_postfix({'Loss': '%.6f' % avg_loss})

                if batch_idx < cfg.eval.val_sample_count:
                    save_image(torch.cat([hazy, output, clean], dim=0),
                               os.path.join(checkpoint_dir,
                                            'val_epoch%d_sample%d.png' % (epoch, batch_idx + 1)),
                               nrow=1, padding=2, normalize=False,
                               value_range=(0, 1), antialias=True)

    avg_val_loss = val_loss / valid_samples if valid_samples > 0 else float('nan')
    if args.rank == 0:
        print('Validation complete - Epoch %d/%d, average loss: %.6f (valid samples: %d/%d)'
              % (epoch, epochs, avg_val_loss, valid_samples, len(val_loader)))
    return avg_val_loss


def evaluate_metrics(epoch):
    """Evaluate full metrics after the specified epoch ends"""
    model.eval()

    metrics = {'psnr': [], 'ssim': [], 'mse': [], 'uiqm': [], 'uciqe': [], 'pcqi': []}

    if args.rank == 0:
        print('\n===== Epoch %d/%d full evaluation =====' % (epoch, epochs))
        pbar = tqdm(val_loader, desc='Computing evaluation metrics', leave=True)
    else:
        pbar = val_loader

    with torch.no_grad():
        for i, (hazy, clean) in enumerate(pbar):
            hazy, clean = hazy.to(device), clean.to(device)

            output, _ = model(hazy)
            output = output.clamp(0, 1)

            output_np = output.squeeze().cpu().numpy().transpose(1, 2, 0)
            clean_np = clean.squeeze().cpu().numpy().transpose(1, 2, 0)

            current_metrics = calculate_all_metrics(clean_np, output_np)
            for key in metrics:
                metrics[key].append(current_metrics[key])

            if args.rank == 0:
                pbar.set_postfix({'PSNR': '%.2f' % current_metrics['psnr'],
                                  'SSIM': '%.4f' % current_metrics['ssim'],
                                  'UIQM': '%.2f' % current_metrics['uiqm']})

    avg_metrics = {k: float(np.mean(v)) for k, v in metrics.items()}

    if args.rank == 0:
        print('\nEpoch %d/%d evaluation metrics:' % (epoch, epochs))
        print('  Average PSNR: %.4f (up)' % avg_metrics['psnr'])
        print('  Average SSIM: %.4f (up)' % avg_metrics['ssim'])
        print('  Average MSE: %.6f (down)' % avg_metrics['mse'])
        print('  Average UIQM: %.4f (up)' % avg_metrics['uiqm'])
        print('  Average UCIQE: %.4f (up)' % avg_metrics['uciqe'])
        print('  Average PCQI: %.4f (up)' % avg_metrics['pcqi'])

        metrics_file = os.path.join(checkpoint_dir, 'metrics_epoch_%d.txt' % epoch)
        with open(metrics_file, 'w', encoding='utf-8') as f:
            f.write('Epoch %d/%d evaluation metrics:\n' % (epoch, epochs))
            f.write('Average PSNR: %.4f (up)\n' % avg_metrics['psnr'])
            f.write('Average SSIM: %.4f (up)\n' % avg_metrics['ssim'])
            f.write('Average MSE: %.6f (down)\n' % avg_metrics['mse'])
            f.write('Average UIQM: %.4f (up)\n' % avg_metrics['uiqm'])
            f.write('Average UCIQE: %.4f (up)\n' % avg_metrics['uciqe'])
            f.write('Average PCQI: %.4f (up)\n' % avg_metrics['pcqi'])

    return avg_metrics


# ------------------------------------------------------------------ main flow
def main():
    best_composite_score = 0.0
    best_psnr = 0.0
    best_metrics = None
    best_epoch = 0

    if args.rank == 0:
        print('Starting FSANet underwater image enhancement model training')
        for line in cfg.summary_lines():
            print(line)
        print('-' * 50)
        print('Experiment directory: %s' % checkpoint_dir)
        print('Training device: %s' % device)
        print('Parallel mode: %s' % args.parallel)
        if args.parallel == 'dataparallel':
            print('GPU count: %d' % len(gpu_ids))
        elif args.parallel == 'distributed':
            print('Distributed process count: %d' % args.world_size)
        print('Training set size: %d images' % len(train_dataset))
        print('Validation set size: %d images' % len(val_dataset))
        print('Batch size: %d' % batch_size)
        print('Initial learning rate: %g' % lr)
        print('Total training epochs: %d' % epochs)
        print('Loss weights: alpha=%g beta=%g gamma=%g lambda_percep=%g'
              % (cfg.loss.alpha, cfg.loss.beta, cfg.loss.gamma, cfg.loss.lambda_percep))
        print('-' * 50)

        print('Validation composite score weight configuration:')
        validate_weights()
        print('-' * 50)

        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())
        print('Total model parameters: %s' % format(total_params, ','))
        print('Trainable parameters: %s' % format(trainable_params, ','))
        print('-' * 50)

        sample_param = next(model.parameters())
        print('Initial parameter sample: %s' % sample_param.flatten()[:5].detach().cpu().numpy())

        # save full config (json for reproducibility + txt for human readability)
        cfg.save_json(os.path.join(checkpoint_dir, 'config.json'))
        config_file = os.path.join(checkpoint_dir, 'experiment_config.txt')
        with open(config_file, 'w', encoding='utf-8') as f:
            f.write('Experiment Config\n' + '=' * 50 + '\n')
            f.write('Experiment time: %s\n' % datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
            f.write('Experiment directory: %s\n' % checkpoint_dir)
            f.write('Parallel mode: %s\n' % args.parallel)
            if args.parallel == 'dataparallel':
                f.write('GPU count: %d\n' % len(gpu_ids))
            elif args.parallel == 'distributed':
                f.write('Distributed process count: %d\n' % args.world_size)
            f.write('Training data: %s\n' % train_dataset.describe())
            f.write('Validation data: %s\n' % val_dataset.describe())
            f.write('Training set size: %d images\n' % len(train_dataset))
            f.write('Validation set size: %d images\n' % len(val_dataset))
            f.write('Batch size: %d\n' % batch_size)
            f.write('Total training epochs: %d\n' % epochs)
            f.write('Initial learning rate: %g\n' % lr)
            f.write('Training device: %s\n' % device)
            f.write('Random seed: %d\n' % cfg.train.seed)
            f.write('Total model parameters: %s\n' % format(total_params, ','))
            f.write('Trainable parameters: %s\n' % format(trainable_params, ','))
            f.write('Model channel count n_filters: %d\n' % cfg.model.n_filters)
            f.write('Optimizer: AdamW (lr=%g, weight_decay=%g, betas=%s)\n'
                    % (cfg.optim.lr, cfg.optim.weight_decay, cfg.optim.betas))
            f.write('Learning rate scheduler: ReduceLROnPlateau (mode=%s, factor=%g, patience=%d, min_lr=%g)\n'
                    % (cfg.scheduler.mode, cfg.scheduler.factor,
                       cfg.scheduler.patience, cfg.scheduler.min_lr))
            f.write('Loss function: MultiSpectralLoss (alpha=%g, beta=%g, gamma=%g, lambda_percep=%g, percep_layers=%d)\n'
                    % (cfg.loss.alpha, cfg.loss.beta, cfg.loss.gamma,
                       cfg.loss.lambda_percep, cfg.loss.percep_layers))
            f.write('AMP: %s | grad_clip: %g\n' % (cfg.train.amp, cfg.train.grad_clip))

    for epoch in range(1, epochs + 1):
        print('\n===== Epoch %d/%d =====' % (epoch, epochs))
        print('Current learning rate: %.8f' % optimizer.param_groups[0]['lr'])

        train_loss = train_epoch(epoch)
        val_loss = validate(epoch)

        need_metrics = (epoch >= cfg.eval.metric_start_epoch) or \
                       (args.rank == 0 and epoch >= cfg.eval.save_best_start_epoch)
        metrics = evaluate_metrics(epoch) if need_metrics else None

        scheduler.step(val_loss)

        if args.rank == 0 and epoch >= cfg.eval.save_best_start_epoch and metrics is not None:
            current_param = next(model.parameters())
            print('Current parameter sample: %s' % current_param.flatten()[:5].detach().cpu().numpy())

            composite_score = calculate_composite_score(metrics)

            if composite_score > best_composite_score:
                best_composite_score = composite_score
                best_psnr = metrics['psnr']
                best_metrics = metrics
                best_epoch = epoch

                if args.parallel == 'distributed':
                    torch.save(model.module.state_dict(),
                               os.path.join(checkpoint_dir, 'unet_simplified_best.pth'))
                else:
                    torch.save(model.state_dict(),
                               os.path.join(checkpoint_dir, 'unet_simplified_best.pth'))

                print('[OK] Saved new best model, Epoch %d' % epoch)
                print('  Composite score: %.4f (up)' % composite_score)
                print('  PSNR: %.4f (up)' % metrics['psnr'])
                print('  SSIM: %.4f (up)' % metrics['ssim'])
                print('  UIQM: %.4f (up)' % metrics['uiqm'])
                print('  UCIQE: %.4f (up)' % metrics['uciqe'])
                print('  PCQI: %.4f (up)' % metrics['pcqi'])
                print('  MSE: %.6f (down)' % metrics['mse'])

                with open(os.path.join(checkpoint_dir, 'best_metrics.txt'), 'w',
                          encoding='utf-8') as f:
                    f.write('Best model (Epoch %d):\n' % epoch)
                    f.write('Composite score: %.4f (up)\n' % composite_score)
                    f.write('Training loss: %.6f\n' % train_loss)
                    f.write('Validation loss: %.6f\n' % val_loss)
                    f.write('Learning rate: %.8f\n' % optimizer.param_groups[0]['lr'])
                    f.write('Loss weights: alpha=%g beta=%g gamma=%g lambda_percep=%g\n'
                            % (cfg.loss.alpha, cfg.loss.beta, cfg.loss.gamma,
                               cfg.loss.lambda_percep))
                    f.write('-' * 50 + '\n')
                    f.write('Full evaluation metrics:\n')
                    f.write('PSNR: %.4f (up)\n' % metrics['psnr'])
                    f.write('SSIM: %.4f (up)\n' % metrics['ssim'])
                    f.write('MSE: %.6f (down)\n' % metrics['mse'])
                    f.write('UIQM: %.4f (up)\n' % metrics['uiqm'])
                    f.write('UCIQE: %.4f (up)\n' % metrics['uciqe'])
                    f.write('PCQI: %.4f (up)\n' % metrics['pcqi'])
                    f.write('-' * 50 + '\n')
                    f.write('Composite score weight configuration:\n')
                    for metric, weight in get_metric_weights().items():
                        f.write('%s weight: %.1f%%\n' % (metric.upper(), weight * 100))
                    f.write('-' * 50 + '\n')
                    f.write('Experiment time: %s\n' % datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
                    f.write('Experiment directory: %s\n' % checkpoint_dir)
            else:
                print('  Current composite score: %.4f (did not exceed best: %.4f)'
                      % (composite_score, best_composite_score))

    if args.rank == 0:
        print('\n===== Training complete =====')
        print('Best model at Epoch %d' % best_epoch)
        print('Best composite score: %.4f (up)' % best_composite_score)
        print('Best PSNR: %.4f (up)' % best_psnr)
        if best_metrics:
            print('Best SSIM: %.4f (up)' % best_metrics['ssim'])
            print('Best MSE: %.6f (down)' % best_metrics['mse'])
            print('Best UIQM: %.4f (up)' % best_metrics['uiqm'])
            print('Best UCIQE: %.4f (up)' % best_metrics['uciqe'])
            print('Best PCQI: %.4f (up)' % best_metrics['pcqi'])

        with open(os.path.join(checkpoint_dir, 'experiment_summary.txt'), 'w',
                  encoding='utf-8') as f:
            f.write('Experiment Summary\n' + '=' * 50 + '\n')
            f.write('Experiment time: %s\n' % datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
            f.write('Experiment directory: %s\n' % checkpoint_dir)
            f.write('Loss weights: alpha=%g beta=%g gamma=%g lambda_percep=%g\n'
                    % (cfg.loss.alpha, cfg.loss.beta, cfg.loss.gamma, cfg.loss.lambda_percep))
            f.write('Optimizer: AdamW lr=%g wd=%g\n' % (cfg.optim.lr, cfg.optim.weight_decay))
            f.write('Random seed: %d\n' % cfg.train.seed)
            f.write('Best model Epoch: %d\n' % best_epoch)
            f.write('Best composite score: %.4f (up)\n' % best_composite_score)
            f.write('Best PSNR: %.4f (up)\n' % best_psnr)
            if best_metrics:
                f.write('Best SSIM: %.4f (up)\n' % best_metrics['ssim'])
                f.write('Best MSE: %.6f (down)\n' % best_metrics['mse'])
                f.write('Best UIQM: %.4f (up)\n' % best_metrics['uiqm'])
                f.write('Best UCIQE: %.4f (up)\n' % best_metrics['uciqe'])
                f.write('Best PCQI: %.4f (up)\n' % best_metrics['pcqi'])
            f.write('-' * 50 + '\n')
            f.write('Composite score weight configuration:\n')
            for metric, weight in get_metric_weights().items():
                f.write('%s weight: %.1f%%\n' % (metric.upper(), weight * 100))

        print('\n===== Starting visualization of best model results =====')
        visualize_best_model()

    if is_distributed:
        destroy_process_group()


def visualize_best_model():
    """Load the best model and export visualization results"""
    show_dir = os.path.join(checkpoint_dir, 'show')
    os.makedirs(show_dir, exist_ok=True)

    best_model_path = os.path.join(checkpoint_dir, 'unet_simplified_best.pth')
    if not os.path.exists(best_model_path):
        print('Error: cannot find best model file %s' % best_model_path)
        return

    model = FSANet(in_channels=cfg.model.in_channels, out_channels=cfg.model.out_channels,
                   n_filters=cfg.model.n_filters, use_abm=cfg.model.use_abm,
                   enc4_layers=cfg.model.enc4_layers,
                   bottleneck_layers=cfg.model.bottleneck_layers).to(device)
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    model.eval()

    print('Loaded best model: %s' % best_model_path)

    vis_dataset = build_dataset(cfg.data.val_dataset, cfg.data.val_split,
                                root=cfg.data.val_root.strip() or data_dir)
    vis_loader = DataLoader(vis_dataset, batch_size=cfg.data.val_batch_size, shuffle=False)

    print('Starting to process %d validation images...' % len(vis_dataset))

    limit = cfg.eval.visualize_count
    with torch.no_grad():
        for i, (hazy, clean) in enumerate(tqdm(vis_loader, desc='Generating visualization results')):
            hazy, clean = hazy.to(device), clean.to(device)
            output, _ = model(hazy)
            output = output.clamp(0, 1)

            save_image(hazy, os.path.join(show_dir, 'input_%04d.png' % (i + 1)),
                       normalize=False, value_range=(0, 1))
            save_image(output, os.path.join(show_dir, 'enhanced_%04d.png' % (i + 1)),
                       normalize=False, value_range=(0, 1))
            save_image(clean, os.path.join(show_dir, 'ground_truth_%04d.png' % (i + 1)),
                       normalize=False, value_range=(0, 1))

            comparison = torch.cat([hazy, output, clean], dim=0)
            save_image(comparison, os.path.join(show_dir, 'comparison_%04d.png' % (i + 1)),
                       nrow=3, normalize=False, value_range=(0, 1), padding=5)

            if limit and (i + 1) >= limit:
                break

    print('Visualization complete! Results saved in: %s' % show_dir)
    print('Generated results for %d images' % (min(len(vis_dataset), limit) if limit else len(vis_dataset)))
    print('File descriptions:')
    print('- input_XXXX.png: original input image')
    print('- enhanced_XXXX.png: image enhanced by the model')
    print('- ground_truth_XXXX.png: ground truth image')
    print('- comparison_XXXX.png: three-image comparison (input-enhanced-ground truth)')
    print('\nExperiment directory: %s' % checkpoint_dir)
    print('Experiment file descriptions:')
    print('- config.json / experiment_config.txt: full hyperparameters in effect for this run')
    print('- best_metrics.txt: full metrics of the best model')
    print('- experiment_summary.txt: experiment summary')
    print('- metrics_epoch_X.txt: detailed metrics for each epoch')
    print('- unet_simplified_best.pth: best model weights')
    print('- show/: visualization results directory')


if __name__ == '__main__':
    main()