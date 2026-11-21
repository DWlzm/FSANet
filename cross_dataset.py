#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Cross-dataset experiment: UIEB full training -> LSUI test.

Writes checkpoints/<run_name>/, results/test_<dataset>_<split>_*/ and cross_<run_name>.{md,json,log}.
"""
import argparse
import json
import os
import subprocess
import sys
from datetime import datetime

HERE = os.path.dirname(os.path.abspath(__file__))


def parse_args():
    p = argparse.ArgumentParser(description='UIEB full training -> LSUI full testing')
    p.add_argument('--epochs', type=int, default=100)
    p.add_argument('--batch_size', type=int, default=2)
    p.add_argument('--lr', type=float, default=None)
    p.add_argument('--alpha', type=float, default=None)
    p.add_argument('--beta', type=float, default=None)
    p.add_argument('--gamma', type=float, default=None)
    p.add_argument('--lambda_percep', type=float, default=None)
    p.add_argument('--n_filters', type=int, default=None)
    p.add_argument('--seed', type=int, default=None)
    p.add_argument('--gpu_ids', type=str, default='0')
    p.add_argument('--run_name', type=str, default='')
    p.add_argument('--train_dataset', type=str, default='uieb')
    p.add_argument('--train_split', type=str, default='full')
    p.add_argument('--val_dataset', type=str, default='uieb')
    p.add_argument('--val_split', type=str, default='val')
    p.add_argument('--test_dataset', type=str, default='lsui')
    p.add_argument('--test_split', type=str, default='test')
    p.add_argument('--skip_indomain', action='store_true', help='skip UIEB test in-domain testing')
    p.add_argument('--save_images', action='store_true', help='save enhanced images during testing')
    p.add_argument('--extra', nargs='*', default=[], help='--override expressions passed through to train.py')
    return p.parse_args()


def run_cmd(cmd, log_path):
    print('>> %s' % ' '.join(cmd))
    with open(log_path, 'a', encoding='utf-8') as f:
        f.write('\n>> %s\n' % ' '.join(cmd))
        rc = subprocess.run(cmd, cwd=HERE, stdout=f, stderr=subprocess.STDOUT).returncode
    if rc != 0:
        raise RuntimeError('Command failed (rc=%d), see %s for details' % (rc, log_path))


def read_metrics(path):
    out = {}
    with open(path, encoding='utf-8') as f:
        for line in f:
            if ':' in line:
                k, v = line.split(':', 1)
                k = k.strip().upper()
                try:
                    out[k] = float(v.strip())
                except ValueError:
                    out[k] = v.strip()
    return out


def find_test_dir(results_root, dataset, split, marker_ts):
    prefix = 'test_%s_%s_' % (dataset, split)
    cands = [d for d in os.listdir(results_root) if d.startswith(prefix)]
    cands = [d for d in cands if os.path.getmtime(os.path.join(results_root, d, 'metrics.txt')) >= marker_ts - 1]
    if not cands:
        raise RuntimeError('Cannot find test result directory %s' % prefix)
    cands.sort(key=lambda d: os.path.getmtime(os.path.join(results_root, d)))
    return os.path.join(results_root, cands[-1])


def main():
    args = parse_args()
    stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_name = args.run_name.strip() or 'cross_uieb2lsui_%s' % stamp
    log_path = os.path.join(HERE, 'cross_%s.log' % run_name)
    train_py = os.path.join(HERE, 'train.py')
    test_py = os.path.join(HERE, 'test.py')

    # ---- 1. Training: UIEB full
    gate = min(args.epochs, 10**6)
    train_cmd = [sys.executable, '-u', train_py,
                 '--epochs', str(args.epochs),
                 '--batch_size', str(args.batch_size),
                 '--gpu_ids', args.gpu_ids,
                 '--dataset', args.train_dataset,
                 '--train_split', args.train_split,
                 '--val_dataset', args.val_dataset,
                 '--val_split', args.val_split,
                 '--run_name', run_name,
                 # ensure the best weights are produced even in the last epoch (so short runs still get a checkpoint)
                 '--override', 'eval.metric_start_epoch=%d' % gate,
                 'eval.save_best_start_epoch=%d' % gate]
    for name in ('lr', 'alpha', 'beta', 'gamma', 'lambda_percep', 'n_filters', 'seed'):
        val = getattr(args, name)
        if val is not None:
            train_cmd += ['--%s' % name, str(val)]
    train_cmd += list(args.extra)
    run_cmd(train_cmd, log_path)

    run_dir = os.path.join(HERE, 'checkpoints', run_name)
    ckpt = os.path.join(run_dir, 'unet_simplified_best.pth')
    if not os.path.exists(ckpt):
        raise RuntimeError('Training did not produce best weights: %s' % ckpt)
    n_filters = args.n_filters
    if n_filters is None:
        cfg_path = os.path.join(run_dir, 'config.json')
        with open(cfg_path, encoding='utf-8') as f:
            n_filters = json.load(f)['model']['n_filters']

    # ---- 2. In-domain testing (UIEB test)
    results_root = os.path.join(HERE, 'results')
    os.makedirs(results_root, exist_ok=True)
    tests = []
    if not args.skip_indomain:
        ts = datetime.now().timestamp()
        cmd = [sys.executable, '-u', test_py, '--model_path', ckpt,
               '--dataset', args.train_dataset, '--split', 'test',
               '--n_filters', str(n_filters), '--gpu', args.gpu_ids.split(',')[0], '--quiet']
        if args.save_images:
            cmd.append('--save_images')
        run_cmd(cmd, log_path)
        tests.append(('in-domain %s/test' % args.train_dataset,
                      read_metrics(os.path.join(find_test_dir(results_root, args.train_dataset, 'test', ts), 'metrics.txt'))))

    # ---- 3. Cross-domain testing (LSUI test full)
    ts = datetime.now().timestamp()
    cmd = [sys.executable, '-u', test_py, '--model_path', ckpt,
           '--dataset', args.test_dataset, '--split', args.test_split,
           '--n_filters', str(n_filters), '--gpu', args.gpu_ids.split(',')[0], '--quiet']
    if args.save_images:
        cmd.append('--save_images')
    run_cmd(cmd, log_path)
    cross_metrics = read_metrics(os.path.join(
        find_test_dir(results_root, args.test_dataset, args.test_split, ts), 'metrics.txt'))
    tests.append(('cross-domain %s/%s' % (args.test_dataset, args.test_split), cross_metrics))

    # ---- 4. Summary report
    keys = ('PSNR', 'SSIM', 'MSE', 'UIQM', 'UCIQE', 'PCQI', 'COMPOSITE')
    lines = ['# Cross-Dataset Experiment Report (%s -> %s)' % (args.train_dataset.upper(), args.test_dataset.upper()), '',
             '- Run name: `%s`' % run_name,
             '- Time: %s' % datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
             '- Training protocol: %s split=%s (full)' % (args.train_dataset, args.train_split),
             '- Weights: `%s` (n_filters=%s)' % (os.path.relpath(ckpt, HERE), n_filters),
             '- Training epochs: %d, batch=%d, log: `%s`' % (args.epochs, args.batch_size, os.path.basename(log_path)),
             '', '| Metric | ' + ' | '.join(t[0] for t in tests) + ' |',
             '|---|' + '---|' * len(tests)]
    for k in keys:
        rows = []
        for _, m in tests:
            v = m.get(k)
            rows.append('%.4f' % v if isinstance(v, float) else str(v))
        lines.append('| %s | %s |' % (k, ' | '.join(rows)))
    lines += ['', '> Cross-domain testing is **zero-shot transfer**: the model has never seen any image from the test dataset.', '']

    md_path = os.path.join(HERE, 'cross_%s.md' % run_name)
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    json_path = os.path.join(HERE, 'cross_%s.json' % run_name)
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump({'run_name': run_name,
                   'n_filters': n_filters,
                   'epochs': args.epochs,
                   'batch_size': args.batch_size,
                   'checkpoint': ckpt,
                   'results': {name: m for name, m in tests}}, f, ensure_ascii=False, indent=2)

    print('\n===== Cross-Dataset Results =====')
    for name, m in tests:
        print('%s: PSNR %.4f | SSIM %.4f | UIQM %.4f | UCIQE %.4f | PCQI %.4f | composite %.4f'
              % (name, m.get('PSNR', 0), m.get('SSIM', 0), m.get('UIQM', 0),
                 m.get('UCIQE', 0), m.get('PCQI', 0), m.get('COMPOSITE', 0)))
    print('Report: %s' % md_path)


if __name__ == '__main__':
    main()
