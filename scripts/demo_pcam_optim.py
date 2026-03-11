"""
demo_pcam_optim.py  —  Optimal Config Search : RN50 + MLP + Side Size × Epochs
================================================================================

Grille complète : 3 tailles de side × 3 durées d'entraînement = 9 runs

  Side sizes  : small (~54k) | medium (~160k) | large (~1.1M)
  Epochs      : 5 | 10 | 20

  Base fixée  : ResNet-50 ImageNet (2048 → proj 512), gelé
  Merge fixé  : MLP  (concat → Linear(1024→512) → ReLU → Linear(512→512))

Chaque run est évalué sur le même val subset.
À la fin, tableau 3×3 val_acc + heatmap ASCII.

Usage:
    python -m scripts.demo_pcam_optim
    python -m scripts.demo_pcam_optim --train_subset 20000
    python -m scripts.demo_pcam_optim --sides small medium   --epoch_list 5 10
    python -m scripts.demo_pcam_optim --sides large          --epoch_list 20
"""

import os, json, argparse, time
from tqdm import tqdm
from itertools import product as iterproduct

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tv_models
import torchvision.transforms as transforms

from tlkit.data.datasets.pcam_datasets import PCAMDataset
import tnt.torchnet as tnt

# ── Args ──────────────────────────────────────────────────────────────────────

parser = argparse.ArgumentParser()
parser.add_argument('--train_subset', type=int,   default=20000)
parser.add_argument('--val_subset',   type=int,   default=2000)
parser.add_argument('--batch_size',   type=int,   default=64)
parser.add_argument('--lr',           type=float, default=1e-3)
parser.add_argument('--sides',        nargs='+',  default=['small', 'medium', 'large'],
                    choices=['small', 'medium', 'large'])
parser.add_argument('--epoch_list',   nargs='+',  type=int, default=[5, 10, 20])
args = parser.parse_args()

PCAM_DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'pcam')
LOG_DIR       = '/tmp/pcam_optim/tensorboards'
SAVE_DIR      = '/tmp/pcam_optim'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Training on device: {device}')

# ═══════════════════════════════════════════════════════════════════════════════
#  Side Networks
# ═══════════════════════════════════════════════════════════════════════════════

class SideNetSmall(nn.Module):
    """2 conv layers, 64 filters — ~54k params."""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(64, 512), nn.ReLU(),
        )
    def forward(self, x): return self.net(x)


class SideNetMedium(nn.Module):
    """3 conv layers, 128 filters — ~160k params."""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32,  3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128,3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(128, 512), nn.ReLU(),
        )
    def forward(self, x): return self.net(x)


class SideNetLarge(nn.Module):
    """4 conv layers, 256 filters + BatchNorm — ~1.1M params."""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3,   64,  3, padding=1), nn.BatchNorm2d(64),  nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64,  128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(256, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(256, 512), nn.ReLU(), nn.Dropout(0.3),
        )
    def forward(self, x): return self.net(x)


SIDE_BUILDERS = {
    'small':  SideNetSmall,
    'medium': SideNetMedium,
    'large':  SideNetLarge,
}

# ═══════════════════════════════════════════════════════════════════════════════
#  Merge MLP
# ═══════════════════════════════════════════════════════════════════════════════

class MergeMLP(nn.Module):
    """concat(base, side) → Linear(1024→512) → ReLU → Linear(512→512)"""
    def __init__(self, dim=512):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(dim * 2, dim), nn.ReLU(),
            nn.Linear(dim, dim),
        )
    def forward(self, f_base, f_side):
        return self.mlp(torch.cat([f_base, f_side], dim=1))


# ═══════════════════════════════════════════════════════════════════════════════
#  Model
# ═══════════════════════════════════════════════════════════════════════════════

class SideTuneRN50MLP(nn.Module):
    """
    Base  : ResNet-50 gelé (2048-d) → Linear proj (2048→512)
    Side  : side_net entraînable    (→ 512)
    Merge : MLP(concat)             (1024→512)
    Head  : Linear(512→2)
    """
    def __init__(self, side_net):
        super().__init__()
        rn50 = tv_models.resnet50(weights=tv_models.ResNet50_Weights.DEFAULT)
        self.base = nn.Sequential(*list(rn50.children())[:-1])
        for p in self.base.parameters():
            p.requires_grad = False

        self.base_proj = nn.Linear(2048, 512)   # entraînable — adapte RN50 → 512
        self.side      = side_net
        self.merge     = MergeMLP(dim=512)
        self.transfer  = nn.Linear(512, 2)

    def forward(self, x):
        with torch.no_grad():
            f_base = self.base(x).flatten(1)    # (B, 2048)
        f_base  = self.base_proj(f_base)         # (B, 512)
        f_side  = self.side(x)                   # (B, 512)
        merged  = self.merge(f_base, f_side)     # (B, 512)
        return self.transfer(merged)             # (B, 2)


# ═══════════════════════════════════════════════════════════════════════════════
#  Data
# ═══════════════════════════════════════════════════════════════════════════════

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


def get_loader(split, n, shuffle):
    x_map = {'train': 'training_split.h5',   'val': 'validation_split.h5'}
    y_map = {'train': 'camelyonpatch_level_2_split_train_y.h5',
             'val':   'camelyonpatch_level_2_split_valid_y.h5'}
    ds = PCAMDataset(
        x_path=f'{PCAM_DATA_DIR}/{x_map[split]}',
        y_path=f'{PCAM_DATA_DIR}/{y_map[split]}',
        transform=T_TRAIN if split == 'train' else T_EVAL,
    )
    if n and n < len(ds):
        ds = torch.utils.data.Subset(ds, torch.randperm(len(ds))[:n].tolist())
    return torch.utils.data.DataLoader(
        ds, batch_size=args.batch_size,
        shuffle=shuffle, num_workers=0, pin_memory=False,
    )


# ═══════════════════════════════════════════════════════════════════════════════
#  Train + eval one config
# ═══════════════════════════════════════════════════════════════════════════════

def run(side_name, n_epochs, dl_train, dl_val, run_id):
    name  = f'rn50_mlp_{side_name}_ep{n_epochs:02d}'
    model = SideTuneRN50MLP(SIDE_BUILDERS[side_name]()).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"\n{'─'*65}")
    print(f"  [{run_id}]  Base=RN50 | Merge=MLP | Side={side_name} | Epochs={n_epochs}")
    print(f"  Trainable params : {n_params:,}")
    print(f"{'─'*65}")

    optimizer = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad], lr=args.lr
    )

    # tnt logging
    os.makedirs(os.path.join(LOG_DIR, name), exist_ok=True)
    mlog = tnt.logger.TensorboardMeterLogger(
        env=f'optim_{name}',
        log_dir=os.path.join(LOG_DIR, name),
        plotstylecombined=True,
    )
    mlog.add_meter('losses/task_0',        tnt.meter.ValueSummaryMeter())
    mlog.add_meter('accuracy_top1/task_0', tnt.meter.ClassErrorMeter(topk=[1], accuracy=True))

    # ── Training ──────────────────────────────────────────────────────────────
    model.train(True)
    start = time.time()

    for epoch in range(n_epochs):
        for x, label in tqdm(dl_train, desc=f'  Ep{epoch:02d}/{n_epochs}', leave=True):
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

    # ── Validation ────────────────────────────────────────────────────────────
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

    print(f"  ✓ train={train_acc:.2f}%  val={val_acc:.2f}%  "
          f"loss={val_loss:.4f}  {elapsed:.0f}s")

    # Save checkpoint
    os.makedirs(SAVE_DIR, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(SAVE_DIR, f'{name}.pth'))

    return {
        'name':      name,
        'side':      side_name,
        'epochs':    n_epochs,
        'val_acc':   val_acc,
        'val_loss':  val_loss,
        'train_acc': train_acc,
        'n_params':  n_params,
        'time':      round(elapsed),
    }


# ═══════════════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════════════

def ascii_heatmap(grid, sides, epochs):
    """Affiche un tableau 3×3 val_acc avec mise en évidence du max."""
    best_val = max(v for row in grid.values() for v in row.values())
    col_w = 12

    header = f"  {'':12}" + "".join(f"{'ep='+str(e):>{col_w}}" for e in epochs)
    print(header)
    print("  " + "─" * (12 + col_w * len(epochs)))
    for s in sides:
        row = f"  {('side_'+s):<12}"
        for e in epochs:
            v = grid[s][e]
            marker = " ★" if abs(v - best_val) < 0.01 else "  "
            row += f"{v:>9.2f}%{marker}"
        print(row)


def main():
    sides      = args.sides
    epoch_list = sorted(args.epoch_list)
    total      = len(sides) * len(epoch_list)

    print(f"\n{'='*65}")
    print(f"  OPTIM SEARCH — Base=RN50 | Merge=MLP")
    print(f"  Side sizes  : {sides}")
    print(f"  Epochs      : {epoch_list}")
    print(f"  Total runs  : {total}")
    print(f"  Train subset: {args.train_subset} | Val: {args.val_subset}")
    print(f"{'='*65}")

    # Loaders partagés (même subset pour tous les runs)
    dl_train = get_loader('train', args.train_subset, shuffle=True)
    dl_val   = get_loader('val',   args.val_subset,   shuffle=False)

    results = []
    grid    = {s: {} for s in sides}   # grid[side][epochs] = val_acc

    for i, (side_name, n_epochs) in enumerate(iterproduct(sides, epoch_list), 1):
        r = run(side_name, n_epochs, dl_train, dl_val, run_id=f'{i}/{total}')
        results.append(r)
        grid[side_name][n_epochs] = r['val_acc']

    # ── Tableau récapitulatif ──────────────────────────────────────────────────
    print(f"\n{'='*65}")
    print(f"  RÉSULTATS — Base=RN50 | Merge=MLP  (val accuracy %)")
    print(f"{'='*65}\n")
    ascii_heatmap(grid, sides, epoch_list)

    print(f"\n  {'Config':<28} {'Val Acc':>8} {'Train Acc':>10} {'Val Loss':>9} {'Params':>10} {'Time':>6}")
    print(f"  {'─'*68}")
    for r in sorted(results, key=lambda x: x['val_acc'], reverse=True):
        marker = " ★" if r == max(results, key=lambda x: x['val_acc']) else ""
        print(f"  {'side_'+r['side']+' / ep='+str(r['epochs']):<28} "
              f"{r['val_acc']:>7.2f}%  {r['train_acc']:>8.2f}%  "
              f"{r['val_loss']:>9.4f}  {r['n_params']:>10,}  {r['time']:>5}s{marker}")

    best = max(results, key=lambda x: x['val_acc'])
    print(f"\n  ✅ Meilleure config : side_{best['side']} / {best['epochs']} epochs")
    print(f"     val_acc={best['val_acc']:.2f}%  train_acc={best['train_acc']:.2f}%  "
          f"val_loss={best['val_loss']:.4f}")

    # Save JSON
    out = os.path.join(SAVE_DIR, 'optim_results.json')
    with open(out, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n  Résultats → {out}")


if __name__ == '__main__':
    main()
