# Model configuration file
# Composite score weight configuration

# Metric weight configuration (total should be 1.0)
METRIC_WEIGHTS = {
    'psnr': 0.25,    # PSNR weight 25%
    'ssim': 0.25,    # SSIM weight 25%
    'uiqm': 0.15,    # UIQM weight 15%
    'uciqe': 0.15,   # UCIQE weight 15%
    'pcqi': 0.10,    # PCQI weight 10%
    'mse': 0.10,     # MSE weight 10%
}

# Metric normalization baselines (used to normalize metrics of different scales to the 0-1 range)
METRIC_BASELINES = {
    'psnr': 35.0,    # PSNR baseline (35dB is a full score)
    'ssim': 1.0,     # SSIM is itself 0-1
    'uiqm': 2.5,     # UIQM baseline
    'uciqe': 0.8,    # UCIQE baseline
    'pcqi': 0.8,     # PCQI baseline
    'mse': 0.1,      # MSE baseline (smaller is better, so 1.0 - mse/baseline is used)
}

# Validate that weights sum correctly
def validate_weights():
    """Validate that the weight configuration is correct"""
    total_weight = sum(METRIC_WEIGHTS.values())
    if abs(total_weight - 1.0) > 1e-6:
        raise ValueError(f"Weight sum must be 1.0, currently {total_weight:.6f}")
    print(f"[OK] Weight configuration validated, sum: {total_weight:.6f}")
    
    # print weight configuration
    print("Current weight configuration:")
    for metric, weight in METRIC_WEIGHTS.items():
        print(f"  {metric.upper()}: {weight:.1%}")
    
    return True

# Get weight configuration
def get_metric_weights():
    """Get metric weight configuration"""
    return METRIC_WEIGHTS.copy()

def get_metric_baselines():
    """Get metric baseline configuration"""
    return METRIC_BASELINES.copy()

if __name__ == "__main__":
    validate_weights()