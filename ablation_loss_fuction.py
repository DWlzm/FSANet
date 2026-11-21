#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ablation_loss_fuction.py — Loss function ablation study (MCEL four components)

Design principle: everything is implemented through the config.py mechanism.
  1. Each ablation preset = a json derived from the base Config (setting a component's weight to 0 removes that component);
  2. Train via subprocess calling train.py --config <json>, then evaluate via test.py;
  3. Aggregate test metrics for each preset into ablation_loss_summary.{csv,md}.

Presets (leave-one-out):
  full         all four components (baseline)
  no_grad      alpha=0   remove gradient loss
  no_lowfreq   beta=0    remove low-frequency loss
  no_colorcorr gamma=0   remove color-correlation loss
  no_percep    lambda=0  remove perceptual loss
  --mode all additionally provides only_grad / only_lowfreq / only_colorcorr / only_percep (single-component)

Examples:
  python ablation_loss_fuction.py --dry_run                       # only print plan and configs, no training
  python ablation_loss_fuction.py --epochs 2 --limit 5            # quick smoke test
  python ablation_loss_fuction.py --epochs 150 --batch_size 8     # full ablation (sequential)
  python ablation_loss_fuction.py --presets full no_percep        # run only specified presets
"""

import argparse
import csv
import json
import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.abspath(__file__))
PYTHON = sys.executable
DEFAULT_OUT = os.path.join(ROOT, 'ablation_loss')

# preset name -> (removed component, weight overrides)
LOO_PRESETS = {
    'full':         ('none (baseline)',        {}),
    'no_grad':      ('gradient loss grad',    {'loss.alpha': 0.0}),
    'no_lowfreq':   ('low-frequency loss lowfreq', {'loss.beta': 0.0}),
    'no_colorcorr': ('color-correlation color',   {'loss.gamma': 0.0}),
    'no_percep':    ('perceptual loss percep',  {'loss.lambda_percep': 0.0}),
}
SINGLE_PRESETS = {
    'only_grad':      ('gradient loss only',    {'loss.alpha': 0.5, 'loss.beta': 0.0, 'loss.gamma': 0.0, 'loss.lambda_percep': 0.0}),
    'only_lowfreq':   ('low-frequency loss only',    {'loss.alpha': 0.0, 'loss.beta': 0.3, 'loss.gamma': 0.0, 'loss.lambda_percep': 0.0}),
    'only_colorcorr': ('color-correlation loss only', {'loss.alpha': 0.0, 'loss.beta': 0.0, 'loss.gamma': 0.2, 'loss.lambda_percep': 0.0}),
    'only_percep':    ('perceptual loss only',    {'loss.alpha': 0.0, 'loss.beta': 0.0, 'loss.gamma': 0.0, 'loss.lambda_percep': 0.1}),
}

METRIC_KEYS = ['PSNR', 'SSIM', 'MSE', 'UIQM', 'UCIQE', 'PCQI', 'COMPOSITE']


def parse_args():
    p = argparse.ArgumentParser(description='MCEL loss function ablation study (config-driven, sequential train + test)')
    p.add_argument('--mode', choices=['loo', 'all'], default='loo',
                   help='loo=full + leave-one-out (5 groups); all=plus single-component groups (9 groups total)')
    p.add_argument('--presets', nargs='*', help='run only the specified preset names, defaults to all according to mode')
    p.add_argument('--epochs', type=int, help='override train.epochs (defaults to config.py)')
    p.add_argument('--batch_size', type=int, help='override train.batch_size')
    p.add_argument('--dataset', type=str, help='training dataset name, e.g. uieb')
    p.add_argument('--train_split', type=str, help='training split, e.g. train / full')
    p.add_argument('--test_split', type=str, default='test', help='test split (default test)')
    p.add_argument('--limit', type=int, default=0, help='evaluate only the first N images for testing, 0=full')
    p.add_argument('--seed', type=int, help='fix random seed for fairness across groups (defaults to config.py)')
    p.add_argument('--output_dir', type=str, default=DEFAULT_OUT)
    p.add_argument('--run_prefix', type=str, default='abl')
    p.add_argument('--dry_run', action='store_true', help='only generate and print configs, do not run training')
    return p.parse_args()


def build_preset_config(args, preset, run_name, cfg_path):
    """Derive a json config for a single preset from the base Config."""
    from config import Config
    cfg = Config()
    overrides = dict(PRESETS[preset][1])
    if args.epochs is not None:
        overrides['train.epochs'] = args.epochs
    if args.batch_size is not None:
        overrides['train.batch_size'] = args.batch_size
    if args.dataset is not None:
        overrides['data.dataset'] = args.dataset
    if args.train_split is not None:
        overrides['data.train_split'] = args.train_split
    if args.seed is not None:
        overrides['train.seed'] = args.seed
    # For short runs, lower the metric/save thresholds, otherwise no best weights are produced
    eff_epochs = args.epochs if args.epochs is not None else cfg.train.epochs
    if eff_epochs < cfg.eval.metric_start_epoch:
        overrides['eval.metric_start_epoch'] = 1
        overrides['eval.save_best_start_epoch'] = 1
    cfg.apply_overrides([f'{k}={v}' for k, v in overrides.items()])
    cfg.set_value('output.run_name', run_name)
    with open(cfg_path, 'w', encoding='utf-8') as f:
        json.dump(cfg.to_dict(), f, ensure_ascii=False, indent=2)
    return cfg


def run_command(cmd, log_path):
    with open(log_path, 'w', encoding='utf-8') as logf:
        proc = subprocess.run(cmd, cwd=ROOT, stdout=logf, stderr=subprocess.STDOUT)
    return proc.returncode == 0


def parse_metrics(metrics_txt):
    out = {}
    if not os.path.exists(metrics_txt):
        return out
    with open(metrics_txt, encoding='utf-8') as f:
        for line in f:
            if ':' in line:
                k, v = line.split(':', 1)
                k = k.strip()
                if k in METRIC_KEYS:
                    try:
                        out[k] = float(v.strip())
                    except ValueError:
                        pass
    return out


def main():
    global PRESETS
    args = parse_args()
    PRESETS = dict(LOO_PRESETS)
    if args.mode == 'all':
        PRESETS.update(SINGLE_PRESETS)
    names = args.presets if args.presets else list(PRESETS.keys())
    bad = [n for n in names if n not in PRESETS]
    if bad:
        sys.exit('Unknown preset: %s, available: %s' % (bad, list(PRESETS.keys())))

    os.makedirs(args.output_dir, exist_ok=True)
    cfg_dir = os.path.join(args.output_dir, 'configs')
    log_dir = os.path.join(args.output_dir, 'logs')
    os.makedirs(cfg_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    print('=' * 70)
    print('Loss ablation study | %d presets: %s' % (len(names), ', '.join(names)))
    print('Output directory: %s' % args.output_dir)

    rows = []
    for i, name in enumerate(names, 1):
        run_name = '%s_%s' % (args.run_prefix, name)
        cfg_path = os.path.join(cfg_dir, '%s.json' % name)
        cfg = build_preset_config(args, name, run_name, cfg_path)
        ckpt = os.path.join(ROOT, 'checkpoints', run_name, 'unet_simplified_best.pth')

        print('\n[%d/%d] %s — removed: %s' % (i, len(names), name, PRESETS[name][0]))
        print('  weights: alpha=%.3g beta=%.3g gamma=%.3g lambda=%.3g' % (
            cfg.loss.alpha, cfg.loss.beta, cfg.loss.gamma, cfg.loss.lambda_percep))
        print('  config: %s' % os.path.relpath(cfg_path, ROOT))

        row = {
            'preset': name, 'removed': PRESETS[name][0],
            'alpha': cfg.loss.alpha, 'beta': cfg.loss.beta,
            'gamma': cfg.loss.gamma, 'lambda_percep': cfg.loss.lambda_percep,
            'epochs': cfg.train.epochs, 'run_name': run_name,
        }

        if args.dry_run:
            row['status'] = 'dry_run'
            rows.append(row)
            continue

        t0 = time.time()
        ok_train = run_command([PYTHON, 'train.py', '--config', cfg_path],
                               os.path.join(log_dir, '%s_train.log' % name))
        if not ok_train or not os.path.exists(ckpt):
            row['status'] = 'train_failed'
            rows.append(row)
            print('  Training failed or no weights, see logs/%s_train.log' % name)
            continue

        test_cmd = [PYTHON, 'test.py', '--model_path', ckpt,
                    '--split', args.test_split, '--quiet']
        if args.limit:
            test_cmd += ['--limit', str(args.limit)]
        ok_test = run_command(test_cmd, os.path.join(log_dir, '%s_test.log' % name))
        if not ok_test:
            row['status'] = 'test_failed'
            rows.append(row)
            print('  Testing failed, see logs/%s_test.log' % name)
            continue

        # test.py writes its result directory in its own log; extract the metrics.txt path from the log
        metrics_txt = None
        with open(os.path.join(log_dir, '%s_test.log' % name), encoding='utf-8') as f:
            for line in f:
                if 'metrics.txt' in line:
                    cand = line.split(':', 1)[-1].strip()
                    if os.path.exists(cand):
                        metrics_txt = cand
        metrics = parse_metrics(metrics_txt) if metrics_txt else {}
        row.update(metrics)
        row['status'] = 'ok'
        row['minutes'] = round((time.time() - t0) / 60, 1)
        rows.append(row)
        print('  PSNR=%.4f SSIM=%.4f UIQM=%.4f composite=%.4f time %.1f min' % (
            metrics.get('PSNR', float('nan')), metrics.get('SSIM', float('nan')),
            metrics.get('UIQM', float('nan')), metrics.get('COMPOSITE', float('nan')),
            row['minutes']))

    # Summary CSV
    csv_path = os.path.join(args.output_dir, 'ablation_loss_summary.csv')
    fields = ['preset', 'removed', 'alpha', 'beta', 'gamma', 'lambda_percep',
              'epochs'] + METRIC_KEYS + ['minutes', 'status', 'run_name']
    with open(csv_path, 'w', newline='', encoding='utf-8-sig') as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
        w.writeheader()
        for r in rows:
            w.writerow(r)

    # Summary Markdown table
    md_path = os.path.join(args.output_dir, 'ablation_loss_summary.md')
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write('# Loss Function Ablation Study Summary\n\n')
        f.write('Each group only changes the loss weights (setting to 0 removes that component); all other hyperparameters are identical and the random seed is fixed.\n\n')
        f.write('| Preset | Removed Component | alpha | beta | gamma | lambda | PSNR | SSIM | UIQM | UCIQE | PCQI | Composite | Status |\n')
        f.write('|---|---|---|---|---|---|---|---|---|---|---|---|---|\n')
        for r in rows:
            f.write('| %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s |\n' % (
                r.get('preset'), r.get('removed'),
                r.get('alpha'), r.get('beta'), r.get('gamma'), r.get('lambda_percep'),
                r.get('PSNR', '-'), r.get('SSIM', '-'), r.get('UIQM', '-'),
                r.get('UCIQE', '-'), r.get('PCQI', '-'), r.get('COMPOSITE', '-'),
                r.get('status')))
        f.write('\nPer-group configs are in `configs/`, training/testing logs are in `logs/`, and per-image metrics are in each `results/` directory.\n')

    print('\n' + '=' * 70)
    print('Summary written to:')
    print('  %s' % csv_path)
    print('  %s' % md_path)


if __name__ == '__main__':
    main()