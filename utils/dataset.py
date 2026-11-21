"""Paired underwater image dataset (UIEB / LSUI / EUVP compatible).

Directory convention (identical across the three datasets):

    <root>/hazy_train  <root>/clean_train
    <root>/hazy_val    <root>/clean_val      # LSUI has no val
    <root>/hazy_test   <root>/clean_test

`split='full'` means merging all existing splits under the root directory (UIEB=890, LSUI=4279),
used for "full training / full testing". Pairs are matched by identical filenames; files missing
one side are automatically skipped.
"""
import os

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

SUBSETS = ('train', 'val', 'test')

# default root directories for each dataset (relative to FSANet/)
DEFAULT_ROOTS = {
    'uieb': '../UIEB',
    'lsui': '../LSUI',
    'euvp': '../EUVP',
}


class PairedUnderwaterDataset(Dataset):
    """hazy_<split> / clean_<split> same-name paired dataset.

    Parameters
    - split : 'train' | 'val' | 'test' | 'full'
    - train : for backward compatibility; when split is not explicitly given, train=True->'train', False->'test'
    """

    VALID_SPLITS = SUBSETS + ('full',)

    def __init__(self, root_dir, train=True, transform=None, split=None):
        self.root_dir = root_dir
        self.train = train
        self.transform = transform if transform is not None else transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((256, 256), antialias=True),
        ])

        if split is None:
            split = 'train' if train else 'test'
        split = split.lower()
        if split not in self.VALID_SPLITS:
            raise ValueError('split must be one of %s, got %r' % (self.VALID_SPLITS, split))
        self.split = split
        self.splits = list(SUBSETS) if split == 'full' else [split]

        # record the actual pairs in each split, for logging/reporting
        self.samples = []       # [(hazy_path, clean_path, 'train/0001.jpg'), ...]
        self.subset_counts = {}
        self.skipped_subsets = []
        self._collect_pairs()

        if not self.samples:
            raise RuntimeError('No usable pairs found under %s (split=%s)' % (root_dir, split))

        # keep legacy attributes for single-split mode, for backward compatibility with external scripts
        if len(self.splits) == 1:
            self.hazy_dir = os.path.join(root_dir, 'hazy_%s' % self.splits[0])
            self.clean_dir = os.path.join(root_dir, 'clean_%s' % self.splits[0])
        self.image_pairs = [(tag.split('/', 1)[1], tag.split('/', 1)[1]) for _, _, tag in self.samples]
        self.names = [tag for _, _, tag in self.samples]

    def _collect_pairs(self):
        for subset in self.splits:
            hazy_dir = os.path.join(self.root_dir, 'hazy_%s' % subset)
            clean_dir = os.path.join(self.root_dir, 'clean_%s' % subset)
            if not (os.path.isdir(hazy_dir) and os.path.isdir(clean_dir)):
                self.skipped_subsets.append(subset)
                continue
            clean_files = {f for f in os.listdir(clean_dir)
                           if os.path.isfile(os.path.join(clean_dir, f))}
            count = 0
            for name in sorted(os.listdir(hazy_dir)):
                if not os.path.isfile(os.path.join(hazy_dir, name)):
                    continue          # skip directories like .ipynb_checkpoints
                if name in clean_files:
                    self.samples.append((os.path.join(hazy_dir, name),
                                         os.path.join(clean_dir, name),
                                         '%s/%s' % (subset, name)))
                    count += 1
            self.subset_counts[subset] = count

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        hazy_path, clean_path, _ = self.samples[idx]
        try:
            hazy_image = Image.open(hazy_path).convert('RGB')
            clean_image = Image.open(clean_path).convert('RGB')
        except Exception:
            print('Error loading image pair: (%s, %s)' % (hazy_path, clean_path))
            raise
        if self.transform:
            hazy_image = self.transform(hazy_image)
            clean_image = self.transform(clean_image)
        return hazy_image, clean_image

    def describe(self):
        parts = ['%s=%d' % (k, v) for k, v in self.subset_counts.items()]
        text = '%s(%s) %s' % (self.root_dir, self.split, ', '.join(parts))
        if self.skipped_subsets:
            text += ' [skipped: %s]' % '/'.join(self.skipped_subsets)
        return text


class UIEBDataset(PairedUnderwaterDataset):
    """UIEB dataset (default root ../UIEB)."""

    def __init__(self, root_dir=None, train=True, transform=None, split=None):
        super().__init__(root_dir or DEFAULT_ROOTS['uieb'], train=train,
                         transform=transform, split=split)


class LSUIDataset(PairedUnderwaterDataset):
    """LSUI dataset (default root ../LSUI)."""

    def __init__(self, root_dir=None, train=True, transform=None, split=None):
        super().__init__(root_dir or DEFAULT_ROOTS['lsui'], train=train,
                         transform=transform, split=split)


_CLASSES = {'uieb': UIEBDataset, 'lsui': LSUIDataset}


def build_dataset(name, split='train', root=None, transform=None):
    """Build a dataset by name: build_dataset('lsui', 'test')."""
    key = (name or 'uieb').strip().lower()
    if key not in _CLASSES:
        raise ValueError('Unknown dataset %r, available: %s' % (name, ', '.join(_CLASSES)))
    return _CLASSES[key](root_dir=root, split=split, transform=transform)