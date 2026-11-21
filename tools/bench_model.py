"""FSANet complexity benchmark: params / FLOPs(MACs) / memory / latency.

Usage:
    python tools/bench_model.py                          # default n_filters=128, test 256/512
    python tools/bench_model.py --n_filters 64
    python tools/bench_model.py --sizes 256 --batch 4 --train_mem
    python tools/bench_model.py --save_json bench.json


"""
import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from FSANet import FSANet  # noqa: E402


def count_params(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    mb = total * 4 / (1024 ** 2)  # fp32
    return total, trainable, mb


def count_flops(model, device, size, verbose=False):
    from thop import profile
    model.eval()
    dummy = torch.randn(1, 3, size, size, device=device)
    with torch.no_grad():
        macs, params = profile(model, inputs=(dummy,), verbose=verbose)
    return macs, params


def measure_latency(model, device, size, iters, warmup=10):
    model.eval()
    dummy = torch.randn(1, 3, size, size, device=device)
    with torch.no_grad():
        for _ in range(warmup):
            model(dummy)
        torch.cuda.synchronize()
        times = []
        for _ in range(iters):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            model(dummy)
            end.record()
            torch.cuda.synchronize()
            times.append(start.elapsed_time(end))
    times.sort()
    med = times[len(times) // 2]
    return med, 1000.0 / med


def peak_inference_memory(model, device, size, iters=20):
    model.eval()
    dummy = torch.randn(1, 3, size, size, device=device)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_allocated()
    with torch.no_grad():
        for _ in range(iters):
            model(dummy)
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated()
    return (peak - base) / (1024 ** 2), peak / (1024 ** 2)


def peak_train_memory(model, device, size, batch, n_filters):
    from utils.losses import MultiSpectralLoss
    crit = MultiSpectralLoss(device=device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    x = torch.randn(batch, 3, size, size, device=device)
    y = torch.randn(batch, 3, size, size, device=device).clamp(0, 1)
    model.train()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_allocated()
    out, _ = model(x)
    loss = crit(out, y)
    loss.backward()
    opt.step()
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated()
    loss_params = sum(p.numel() for p in crit.parameters())
    return (peak - base) / (1024 ** 2), peak / (1024 ** 2), float(loss.item()), loss_params


def main():
    ap = argparse.ArgumentParser(description='FSANet complexity benchmark')
    ap.add_argument('--n_filters', type=int, default=32)
    ap.add_argument('--enc4_layers', type=int, default=2)
    ap.add_argument('--bottleneck_layers', type=int, default=7)
    ap.add_argument('--use_abm', dest='use_abm', action='store_true', default=True)
    ap.add_argument('--no_abm', dest='use_abm', action='store_false')
    ap.add_argument('--in_channels', type=int, default=3)
    ap.add_argument('--out_channels', type=int, default=3)
    ap.add_argument('--sizes', type=int, nargs='+', default=[256, 512])
    ap.add_argument('--iters', type=int, default=50)
    ap.add_argument('--train_mem', action='store_true', help='also measure training-step memory')
    ap.add_argument('--train_batch', type=int, default=2)
    ap.add_argument('--gpu', type=str, default='0')
    ap.add_argument('--save_json', type=str, default='')
    args = ap.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'

    model = FSANet(in_channels=args.in_channels, out_channels=args.out_channels,
                   n_filters=args.n_filters, use_abm=args.use_abm,
                   enc4_layers=args.enc4_layers,
                   bottleneck_layers=args.bottleneck_layers).to(device)

    total, trainable, mb = count_params(model)
    print('=' * 62)
    print('FSANet complexity benchmark | GPU: %s | n_filters=%d' % (gpu_name, args.n_filters))
    print('=' * 62)
    print('Parameters      : %s (%.2f M)' % (format(total, ','), total / 1e6))
    print('Trainable params: %s' % format(trainable, ','))
    print('Weight size     : %.1f MB (fp32)' % mb)

    result = {'gpu': gpu_name, 'n_filters': args.n_filters,
              'params': total, 'params_trainable': trainable,
              'weight_mb_fp32': round(mb, 2), 'input_channels': args.in_channels,
              'ops': {}}

    if not torch.cuda.is_available():
        print('No CUDA, skipping FLOPs/latency/memory measurement')
        return

    for size in args.sizes:
        macs, thop_params = count_flops(model, device, size)
        lat_ms, fps = measure_latency(model, device, size, args.iters)
        print('-' * 62)
        print('Input %dx%d:' % (size, size))
        print('  FLOPs(MACs): %.2f GMACs (%.2f GFLOPs, thop convention)' % (macs / 1e9, macs / 1e9))
        print('  Latency    : %.2f ms/image (batch=1, fp32)' % lat_ms)
        print('  Throughput : %.1f FPS' % fps)
        print('  Params check: %s' % format(int(thop_params) if thop_params else total, ','))
        result['ops'][str(size)] = {
            'macs': macs, 'gmacs': round(macs / 1e9, 3),
            'latency_ms': round(lat_ms, 2), 'fps': round(fps, 1),
        }

    infer_delta, infer_peak = peak_inference_memory(model, device, args.sizes[0])
    print('-' * 62)
    print('Inference memory: delta %.2f GB / peak %.2f GB (batch=1, %dx%d)'
          % (infer_delta / 1024, infer_peak / 1024, args.sizes[0], args.sizes[0]))
    result['infer_memory'] = {'delta_gb': round(infer_delta / 1024, 2),
                              'peak_gb': round(infer_peak / 1024, 2),
                              'batch': 1, 'size': args.sizes[0]}

    if args.train_mem:
        try:
            tr_delta, tr_peak, loss_val, loss_params = peak_train_memory(
                model, device, args.sizes[0], args.train_batch, args.n_filters)
            print('Training-step memory: delta %.2f GB / peak %.2f GB (batch=%d, including loss and backward)'
                  % (tr_delta / 1024, tr_peak / 1024, args.train_batch))
            print('  Single-step loss: %.4f  loss module (VGG) params: %s'
                  % (loss_val, format(loss_params, ',')))
            result['train_memory'] = {'delta_gb': round(tr_delta / 1024, 2),
                                      'peak_gb': round(tr_peak / 1024, 2),
                                      'batch': args.train_batch, 'size': args.sizes[0],
                                      'step_loss': round(loss_val, 4),
                                      'loss_params': loss_params}
        except torch.cuda.OutOfMemoryError as e:
            print('Training-step memory test OOM (batch=%d): %s' % (args.train_batch, e))
            result['train_memory'] = {'oom': True, 'batch': args.train_batch}

    if args.save_json:
        with open(args.save_json, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print('Results written to: %s' % args.save_json)


if __name__ == '__main__':
    main()