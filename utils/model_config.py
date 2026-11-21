"""Metric weights and normalization baselines used by the composite score."""

METRIC_WEIGHTS = {
    'psnr': 0.25,
    'ssim': 0.25,
    'uiqm': 0.15,
    'uciqe': 0.15,
    'pcqi': 0.10,
    'mse': 0.10,
}

# PSNR 35 dB / UIQM 2.5 / UCIQE 0.8 / PCQI 0.8 count as a full score; MSE is inverted.
METRIC_BASELINES = {
    'psnr': 35.0,
    'ssim': 1.0,
    'uiqm': 2.5,
    'uciqe': 0.8,
    'pcqi': 0.8,
    'mse': 0.1,
}


def validate_weights():
    total_weight = sum(METRIC_WEIGHTS.values())
    if abs(total_weight - 1.0) > 1e-6:
        raise ValueError(f"Weight sum must be 1.0, currently {total_weight:.6f}")
    print(f"[OK] Weight configuration validated, sum: {total_weight:.6f}")

    print("Current weight configuration:")
    for metric, weight in METRIC_WEIGHTS.items():
        print(f"  {metric.upper()}: {weight:.1%}")

    return True


def get_metric_weights():
    return METRIC_WEIGHTS.copy()

def get_metric_baselines():
    return METRIC_BASELINES.copy()

if __name__ == "__main__":
    validate_weights()
