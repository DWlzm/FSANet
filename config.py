"""Centralized hyperparameters for FSANet training; train.py carries no magic numbers.

Override order (later wins): defaults here < --config json < --override k=v < named CLI flags.
Every run writes the effective config to <run>/config.json, which can be fed back via --config.
Composite-score weights and baselines stay in utils/model_config.py (metric config, not training).
"""
from dataclasses import dataclass, field, asdict
import json
import os
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class DataConfig:
    """Dataset and DataLoader; validation may come from another source (empty val_root reuses root)."""
    dataset: str = 'uieb'                 # training set: uieb / lsui
    root: str = '../UIEB'                 # training set root directory (relative to FSANet/)
    train_split: str = 'train'            # train / val / test / full (full = merge all splits)
    val_dataset: str = 'uieb'             # validation set used for monitoring during training
    val_root: str = ''                    # empty means same as root
    val_split: str = 'val'
    num_workers: int = 4
    pin_memory: bool = True
    val_batch_size: int = 1               # validation/metric computation fixed at one image at a time


@dataclass
class ModelConfig:
    """FSANet backbone (paper-style implementation, channel ratio 1:2:4:8:16)."""
    in_channels: int = 3
    out_channels: int = 3
    n_filters: int = 32                   # base width f, channels at each level f/2f/4f/8f/16f
    use_abm: bool = True                  # attention bridge (ABM) switch
    enc4_layers: int = 2                  # number of CBR layers in enc4 (first layer 4f->8f, rest 8f->8f)
    bottleneck_layers: int = 7            # number of bottleneck CBR layers (default 7, total params approx. 21.84M)
    pretrained_path: str = 'checkpoints/unet_simplified_best.pth'
    load_pretrained_if_exists: bool = True


@dataclass
class LossConfig:
    """MultiSpectralLoss weights: alpha*gradient + beta*low-freq + gamma*color-corr + lambda_percep*VGG."""
    alpha: float = 0.5                    # gradient (Sobel magnitude) L1
    beta: float = 0.3                     # low-frequency (MSE after avg_pool)
    gamma: float = 0.2                    # color-correlation L1
    lambda_percep: float = 0.1            # VGG perceptual L1
    percep_layers: int = 16               # VGG16 features[:N]
    vgg_pretrained: bool = True


@dataclass
class OptimConfig:
    name: str = 'adamw'                   # train.py currently supports adamw
    lr: float = 1e-2
    weight_decay: float = 1e-5
    betas: Tuple[float, float] = (0.9, 0.999)


@dataclass
class SchedulerConfig:
    name: str = 'reduce_on_plateau'       # currently supports reduce_on_plateau
    mode: str = 'min'
    factor: float = 0.5
    patience: int = 5
    min_lr: float = 1e-7


@dataclass
class TrainConfig:
    epochs: int = 100
    batch_size: int = 8                   # recommended 2 for 12 GB cards (batch 8 will OOM)
    amp: bool = True                      # mixed precision GradScaler
    grad_clip: float = 0.0                # clip by global norm when >0; 0 means no clipping (original behavior)
    seed: int = 42
    deterministic: bool = False           # enable cudnn.deterministic when True (slower)
    log_interval: int = 100               # unused; the progress bar refreshes per batch


@dataclass
class EvalConfig:
    metric_start_epoch: int = 51          # no metrics before this epoch
    save_best_start_epoch: int = 51       # no checkpoint before this epoch
    val_sample_count: int = 5             # number of triptychs saved per validation round
    visualize_count: int = 50             # upper limit on number of visualizations after training


@dataclass
class OutputConfig:
    root: str = 'checkpoints'
    prefix: str = 'experiment'            # directory name <prefix>_<timestamp>
    run_name: str = ''                    # when non-empty, use this name directly for the directory (easier to identify during sweeps)


@dataclass
class TestConfig:
    """Defaults for test.py (dataset and output location during testing)."""
    dataset: str = 'uieb'                 # uieb / lsui
    root: str = ''                        # empty means use the default root directory in utils.dataset
    split: str = 'test'                   # train / val / test / full
    batch_size: int = 1
    output_dir: str = 'results'


@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    optim: OptimConfig = field(default_factory=OptimConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    test: TestConfig = field(default_factory=TestConfig)

    # ---------------------------------------------------------------- serialization
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def save_json(self, path: str) -> str:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)
        return path

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Config':
        cfg = cls()
        for key, val in (data or {}).items():
            if isinstance(val, dict):
                for sub_key, sub_val in val.items():
                    cfg.apply_override('%s.%s=%s' % (key, sub_key, _to_text(sub_val)))
            else:
                cfg.apply_override('%s=%s' % (key, _to_text(val)))
        return cfg

    @classmethod
    def from_file(cls, path: str) -> 'Config':
        with open(path, encoding='utf-8') as f:
            data = json.load(f)
        # support flat dotted keys
        if any('.' in k for k in data.keys()):
            cfg = cls()
            for k, v in data.items():
                cfg.apply_override('%s=%s' % (k, _to_text(v)))
            return cfg
        return cls.from_dict(data)

    # ---------------------------------------------------------------- overrides
    def apply_override(self, expr: str) -> 'Config':
        """Override a field in the form 'a.b.c=value'; value is converted according to the original field type."""
        if '=' not in expr:
            raise ValueError('Override expression must be in the form key=value, got: %r' % expr)
        key, raw = expr.split('=', 1)
        self.set_value(key.strip(), raw.strip())
        return self

    def apply_overrides(self, exprs: List[str]) -> 'Config':
        for e in exprs or []:
            self.apply_override(e)
        return self

    def get_value(self, path: str) -> Any:
        node: Any = self
        for part in path.split('.'):
            if not hasattr(node, part):
                raise KeyError('Unknown config item: %s' % path)
            node = getattr(node, part)
        return node

    def set_value(self, path: str, raw: Any) -> None:
        parts = path.split('.')
        node: Any = self
        for part in parts[:-1]:
            if not hasattr(node, part):
                raise KeyError('Unknown config item: %s' % path)
            node = getattr(node, part)
        leaf = parts[-1]
        if not hasattr(node, leaf):
            raise KeyError('Unknown config item: %s' % path)
        setattr(node, leaf, _coerce(getattr(node, leaf), raw))

    # ---------------------------------------------------------------- display
    def summary_lines(self) -> List[str]:
        lines = ['==== Training config for this run ====']
        for section, obj in (('data', self.data), ('model', self.model), ('loss', self.loss),
                             ('optim', self.optim), ('scheduler', self.scheduler),
                             ('train', self.train), ('eval', self.eval),
                             ('output', self.output), ('test', self.test)):
            items = ', '.join('%s=%s' % (k, v) for k, v in asdict(obj).items())
            lines.append('  [%s] %s' % (section, items))
        return lines


# -------------------------------------------------------------------- utilities
def _to_text(val: Any) -> str:
    if isinstance(val, (list, tuple)):
        return json.dumps(list(val))
    return str(val)


def _coerce(current: Any, raw: Any) -> Any:
    """Convert the string/raw value according to the current field type."""
    if not isinstance(raw, str):
        target = current
        if isinstance(target, bool):
            return bool(raw)
        if isinstance(target, int):
            return int(raw)
        if isinstance(target, float):
            return float(raw)
        return raw
    text = raw.strip()
    if isinstance(current, bool):
        if text.lower() in ('1', 'true', 'yes', 'on'):
            return True
        if text.lower() in ('0', 'false', 'no', 'off'):
            return False
        raise ValueError('Cannot parse boolean value: %r' % raw)
    if isinstance(current, int):
        return int(text, 0) if text.lower().startswith('0x') else int(text)
    if isinstance(current, float):
        return float(text)
    if isinstance(current, (list, tuple)):
        parsed = json.loads(text)
        return tuple(parsed) if isinstance(current, tuple) else parsed
    return text


def default_config() -> Config:
    return Config()