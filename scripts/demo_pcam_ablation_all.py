"""
demo_pcam_ablation.py  —  Ablation Study of Side-Tuning on PatchCamelyon
========================================================================

8 configurations organisées en 3 groupes :

  GROUP 1 — Side network size  (base=RN18, merge=Alpha)
    [1] side_small   : 2 conv layers, 64 filters   ~40k params
    [2] side_medium  : 3 conv layers, 128 filters  ~160k params  ← référence
    [3] side_large   : 4 conv layers, 256 + BN     ~600k params

  GROUP 2 — Merge method  (base=RN18, side=medium)
    [4] merge_alpha   : alpha*base + (1-alpha)*side  (learnable scalar)
    [5] merge_mlp     : concat → MLP → 512
    [6] merge_product : base ⊙ side  (element-wise)

  GROUP 3 — Base network  (side=medium, merge=Alpha)
    [7] base_rn18 : ResNet-18 ImageNet  (512-d)
    [8] base_rn50 : ResNet-50 ImageNet  (2048-d → proj 512)

Usage:
    python -m scripts.demo_pcam_ablation
    python -m scripts.demo_pcam_ablation --group side_size
    python -m scripts.demo_pcam_ablation --group merge
    python -m scripts.demo_pcam_ablation --group base
    python -m scripts.demo_pcam_ablation --epochs 10 --train_subset 20000
"""

import os, json, argparse, time
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tv_models
import torchvision.transforms as transforms

from tlkit.data.datasets.pcam_datasets import PCAMDataset
import tnt.torchnet as tnt

# ── Config ────────────────────────────────────────────────────────────────────

PCAM_DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'pcam')
LOG_DIR       = '/tmp/pcam_ablation/tensorboards'
SAVE_DIR      = '/tmp/pcam_ablation'

BATCH_SIZE   = 64
NUM_EPOCHS   = 10
LR           = 1e-3
TRAIN_SUBSET = 20000
VAL_SUBSET   = 2000

parser = argparse.ArgumentParser()
parser.add_argument('--group',        type=str, default='all',
                    choices=['side_size', 'merge', 'base', 'all'])
parser.add_argument('--epochs',       type=int, default=NUM_EPOCHS)
parser.add_argument('--train_subset', type=int, default=TRAIN_SUBSET)
args, _ = parser.parse_known_args()
NUM_EPOCHS   = args.epochs
TRAIN_SUBSET = args.train_subset

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Training on device:', device)


# ═══════════════════════════════════════════════════════════════════════════════
#  Side Networks
# ═══════════════════════════════════════════════════════════════════════════════

class SideNetSmall(nn.Module):
    """2 conv layers, 64 filters max — ~40k params."""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),   # 48×48
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),  # 24×24
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(64, 512), nn.ReLU(),
        )
    def forward(self, x): return self.net(x)


class SideNetMedium(nn.Module):
    """3 conv layers, 128 filters — ~160k params. Référence."""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32,  3, padding=1), nn.ReLU(), nn.MaxPool2d(2),  # 48×48
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),  # 24×24
            nn.Conv2d(64, 128,3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(128, 512), nn.ReLU(),
        )
    def forward(self, x): return self.net(x)


class SideNetLarge(nn.Module):
    """4 conv layers, 256 filters + BatchNorm — ~600k params."""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3,   64,  3, padding=1), nn.BatchNorm2d(64),  nn.ReLU(), nn.MaxPool2d(2),  # 48×48
            nn.Conv2d(64,  128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2),  # 24×24
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(), nn.MaxPool2d(2),  # 12×12
            nn.Conv2d(256, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(256, 512), nn.ReLU(), nn.Dropout(0.3),
        )
    def forward(self, x): return self.net(x)


# ═══════════════════════════════════════════════════════════════════════════════
#  Merge Operators
# ═══════════════════════════════════════════════════════════════════════════════

class MergeAlpha(nn.Module):
    """
    Alpha-blending : alpha*B(x) + (1-alpha)*S(x)
    Correspond exactement à merge_operators.Alpha du framework.
    alpha est un scalaire appris, initialisé à 0.5.
    """
    def __init__(self):
        super().__init__()
        self.alpha = nn.Parameter(torch.tensor(0.5))

    def forward(self, f_base, f_side):
        a = self.alpha.clamp(0.0, 1.0)
        return a * f_base + (1.0 - a) * f_side

    def extra_info(self):
        return f"α={self.alpha.item():.3f}"


class MergeMLP(nn.Module):
    """
    MLP merger : concat(base, side) → Linear(1024→512) → ReLU → Linear(512→512)
    Correspond à merge_operators.MLP du framework.
    Plus expressif que Alpha — apprend une combinaison non-linéaire.
    """
    def __init__(self, dim=512):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(dim * 2, dim), nn.ReLU(),
            nn.Linear(dim, dim),
        )

    def forward(self, f_base, f_side):
        return self.mlp(torch.cat([f_base, f_side], dim=1))

    def extra_info(self): return ""


class MergeProduct(nn.Module):
    """
    Element-wise product : B(x) ⊙ S(x)
    Interaction multiplicative — chaque dimension de la base
    est modulée par le side network.
    """
    def __init__(self):
        super().__init__()

    def forward(self, f_base, f_side):
        return f_base * f_side

    def extra_info(self): return ""


# ═══════════════════════════════════════════════════════════════════════════════
#  SideTuneModel — assemblage modulaire
# ═══════════════════════════════════════════════════════════════════════════════

class SideTuneModel(nn.Module):
    """
    T( Merge( B(x), S(x) ) )
    B = base encoder gelé
    S = side network entraînable
    Merge = alpha / mlp / product
    T = nn.Linear(512, 2)
    """
    def __init__(self, base_encoder, base_dim, side_net, merge_op, num_classes=2):
        super().__init__()
        self.base     = base_encoder
        self.side     = side_net
        self.merge    = merge_op
        # Projection si la base a une dim différente de 512 (ex: RN50 → 2048)
        self.base_proj = nn.Linear(base_dim, 512) if base_dim != 512 else nn.Identity()
        self.transfer  = nn.Linear(512, num_classes)

    def forward(self, x):
        with torch.no_grad():
            f_base = self.base(x).flatten(1)    # (B, base_dim)
        f_base = self.base_proj(f_base)          # (B, 512)
        f_side = self.side(x)                    # (B, 512)
        merged = self.merge(f_base, f_side)      # (B, 512)
        return self.transfer(merged)


# ═══════════════════════════════════════════════════════════════════════════════
#  Builders
# ═══════════════════════════════════════════════════════════════════════════════

def frozen_resnet18():
    m = tv_models.resnet18(weights=tv_models.ResNet18_Weights.DEFAULT)
    enc = nn.Sequential(*list(m.children())[:-1])
    for p in enc.parameters(): p.requires_grad = False
    return enc, 512


def frozen_resnet50():
    m = tv_models.resnet50(weights=tv_models.ResNet50_Weights.DEFAULT)
    enc = nn.Sequential(*list(m.children())[:-1])
    for p in enc.parameters(): p.requires_grad = False
    return enc, 2048


# ── Transforms ────────────────────────────────────────────────────────────────

T_TRAIN = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])
T_EVAL = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


def build_dataset(split):
    x_files = {'train': 'training_split.h5',
                'val':   'validation_split.h5'}
    y_files = {'train': 'camelyonpatch_level_2_split_train_y.h5',
                'val':   'camelyonpatch_level_2_split_valid_y.h5'}
    transform = T_TRAIN if split == 'train' else T_EVAL
    return PCAMDataset(
        x_path=f'{PCAM_DATA_DIR}/{x_files[split]}',
        y_path=f'{PCAM_DATA_DIR}/{y_files[split]}',
        transform=transform
    )


def get_loader(dataset, n, shuffle):
    if n and n < len(dataset):
        dataset = torch.utils.data.Subset(
            dataset, torch.randperm(len(dataset))[:n].tolist())
    return torch.utils.data.DataLoader(
        dataset, batch_size=BATCH_SIZE,
        shuffle=shuffle, num_workers=0, pin_memory=False)


# ═══════════════════════════════════════════════════════════════════════════════
#  Run one config
# ═══════════════════════════════════════════════════════════════════════════════

def run_config(name, model, description):
    print(f"\n{'─'*65}")
    print(f"  [{name}]")
    print(f"  {description}")
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Trainable params : {n_params:,}")
    print(f"{'─'*65}")

    train_ds = build_dataset('train')
    val_ds   = build_dataset('val')
    dl_train = get_loader(train_ds, TRAIN_SUBSET, shuffle=True)
    dl_val   = get_loader(val_ds,   VAL_SUBSET,   shuffle=False)

    model = model.to(device)
    os.makedirs(os.path.join(LOG_DIR, name), exist_ok=True)

    # Optimizer — weight_decay=0 pour alpha
    alpha_params = [p for n, p in model.named_parameters()
                    if 'alpha' in n and p.requires_grad]
    other_params = [p for n, p in model.named_parameters()
                    if 'alpha' not in n and p.requires_grad]
    optimizer = torch.optim.Adam([
        {'params': alpha_params, 'weight_decay': 0.0},
        {'params': other_params},
    ], lr=LR)

    # Logging tnt — même structure que demo_icifar
    mlog = tnt.logger.TensorboardMeterLogger(
        env=f'ablation_{name}',
        log_dir=os.path.join(LOG_DIR, name),
        plotstylecombined=True
    )
    mlog.add_meter('losses/task_0',        tnt.meter.ValueSummaryMeter())
    mlog.add_meter('accuracy_top1/task_0', tnt.meter.ClassErrorMeter(topk=[1], accuracy=True))

    # Training
    model.train(True)
    start = time.time()
    for epoch in range(NUM_EPOCHS):
        for x, label in tqdm(dl_train, desc=f'  Ep{epoch:02d}', leave=True):
            x, label = x.to(device), label.to(device)
            pred = model(x)
            loss = F.cross_entropy(pred, label)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            mlog.update_meter(loss.item() / x.shape[0],
                              meters={'losses/task_0'}, phase='train')
            mlog.update_meter(pred, target=label,
                              meters={'accuracy_top1/task_0'}, phase='train')

    elapsed   = time.time() - start
    train_acc = mlog.peek_meter('train')['accuracy_top1/task_0']

    # Validation
    model.train(False)
    with torch.no_grad():
        for x, label in tqdm(dl_val, desc='  Val', leave=True):
            x, label = x.to(device), label.to(device)
            pred = model(x)
            loss = F.cross_entropy(pred, label)
            mlog.update_meter(loss.item() / x.shape[0],
                              meters={'losses/task_0'}, phase='val')
            mlog.update_meter(pred, target=label,
                              meters={'accuracy_top1/task_0'}, phase='val')

    val_acc  = mlog.peek_meter('val')['accuracy_top1/task_0']
    val_loss = mlog.peek_meter('val')['losses/task_0'].item()

    # Alpha info
    extra = model.merge.extra_info() if hasattr(model.merge, 'extra_info') else ""
    print(f"  ✓ train={train_acc:.2f}%  val={val_acc:.2f}%  "
          f"loss={val_loss:.4f}  {elapsed:.0f}s  {extra}")

    return {
        'name':        name,
        'description': description,
        'val_acc':     val_acc,
        'val_loss':    val_loss,
        'train_acc':   train_acc,
        'n_params':    n_params,
        'alpha':       model.merge.alpha.item() if hasattr(model.merge, 'alpha') else None,
        'time':        round(elapsed),
    }


# ═══════════════════════════════════════════════════════════════════════════════
#  Configs registry
# ═══════════════════════════════════════════════════════════════════════════════

def get_configs(group):
    enc18, dim18 = frozen_resnet18()
    enc50, dim50 = frozen_resnet50()

    # Chaque config = (nom, model, description)
    side_size_configs = [
        ('side_small',
         SideTuneModel(enc18, dim18, SideNetSmall(),  MergeAlpha()),
         "Side=Small  (~40k params)  | Base=RN18 | Merge=Alpha"),

        ('side_medium',
         SideTuneModel(enc18, dim18, SideNetMedium(), MergeAlpha()),
         "Side=Medium (~160k params) | Base=RN18 | Merge=Alpha  ← référence"),

        ('side_large',
         SideTuneModel(enc18, dim18, SideNetLarge(),  MergeAlpha()),
         "Side=Large  (~600k params) | Base=RN18 | Merge=Alpha"),
    ]

    merge_configs = [
        ('merge_alpha',
         SideTuneModel(enc18, dim18, SideNetMedium(), MergeAlpha()),
         "Side=Medium | Base=RN18 | Merge=Alpha    (learnable scalar)  ← référence"),

        ('merge_mlp',
         SideTuneModel(enc18, dim18, SideNetMedium(), MergeMLP()),
         "Side=Medium | Base=RN18 | Merge=MLP      (concat → non-linear)"),

        ('merge_product',
         SideTuneModel(enc18, dim18, SideNetMedium(), MergeProduct()),
         "Side=Medium | Base=RN18 | Merge=Product  (element-wise ⊙)"),
    ]

    base_configs = [
        ('base_rn18',
         SideTuneModel(enc18, dim18, SideNetMedium(), MergeAlpha()),
         "Side=Medium | Base=ResNet-18 (512-d)   | Merge=Alpha  ← référence"),

        ('base_rn50',
         SideTuneModel(enc50, dim50, SideNetMedium(), MergeAlpha()),
         "Side=Medium | Base=ResNet-50 (2048→512)| Merge=Alpha"),
    ]

    all_configs = side_size_configs + merge_configs + base_configs

    if group == 'side_size':    return side_size_configs
    if group == 'merge':        return merge_configs
    if group == 'base':         return base_configs
    return all_configs  # 'all' → 8 configs


# ═══════════════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════════════

GROUP_TITLES = {
    'side_size': 'GROUP 1 — Side Network Size',
    'merge':     'GROUP 2 — Merge Method',
    'base':      'GROUP 3 — Base Network',
    'all':       'ALL GROUPS (8 configs)',
}

def main():
    configs = get_configs(args.group)

    print(f"\n{'='*65}")
    print(f"  ABLATION STUDY — Side-Tuning on PCam")
    print(f"  {GROUP_TITLES[args.group]}")
    print(f"  Epochs={NUM_EPOCHS} | Train={TRAIN_SUBSET} | Val={VAL_SUBSET}")
    print(f"{'='*65}")

    results = []
    for name, model, desc in configs:
        results.append(run_config(name, model, desc))

    # ── Tableau récapitulatif ──────────────────────────────────────────────────
    print(f"\n{'='*65}")
    print(f"  RÉSULTATS FINAUX — {GROUP_TITLES[args.group]}")
    print(f"{'='*65}")
    print(f"  {'Config':<16} {'Val Acc':>8} {'Train Acc':>10} {'Val Loss':>9} {'Params':>9} {'Alpha':>7}")
    print(f"  {'─'*62}")

    for r in sorted(results, key=lambda x: x['val_acc'], reverse=True):
        alpha_str = f"{r['alpha']:.3f}" if r['alpha'] is not None else "  —  "
        marker    = " ★" if r == max(results, key=lambda x: x['val_acc']) else ""
        print(f"  {r['name']:<16} {r['val_acc']:>7.2f}%  "
              f"{r['train_acc']:>8.2f}%  {r['val_loss']:>9.4f}  "
              f"{r['n_params']:>9,}  {alpha_str:>7}{marker}")

    # Save JSON
    os.makedirs(SAVE_DIR, exist_ok=True)
    out = os.path.join(SAVE_DIR, f'ablation_{args.group}.json')
    with open(out, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n  Résultats → {out}")


if __name__ == '__main__':
    main()
