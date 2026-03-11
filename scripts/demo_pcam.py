"""
demo_pcam.py  —  Side-Tuning on PatchCamelyon (PCam)
Binary classification: 0 = healthy tissue, 1 = metastatic tissue

Structure calquée sur demo_icifar.py (même boucle, même logging tnt).
On implémente directement le réseau side-tuning sans passer par
LifelongSidetuneNetwork (qui utilise eval() incompatible avec nos classes custom).

Comparaison de 3 méthodes :
  - features  : ResNet-18 gelé + tête linéaire (baseline naïve)
  - finetune  : ResNet-18 entièrement entraînable (baseline forte)
  - sidetune  : Base gelée + side network + alpha-blending (notre méthode)

Usage:
    python -m scripts.demo_pcam
    python -m scripts.demo_pcam --method sidetune
    python -m scripts.demo_pcam --method features
    python -m scripts.demo_pcam --method finetune
    python -m scripts.demo_pcam --epochs 20 --train_subset 50000
"""

import os
import argparse
import numpy as np
import time
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tv_models

from tlkit.data.datasets.pcam_datasets import get_pcam_datasets
import tnt.torchnet as tnt

# ── Config ────────────────────────────────────────────────────────────────────

PCAM_DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'pcam')
LOG_DIR       = '/tmp/pcam_demo/tensorboards'
SAVE_DIR      = '/tmp/pcam_demo'

BATCH_SIZE    = 64
NUM_EPOCHS    = 20
LR            = 1e-3
TRAIN_SUBSET  = 10000   # None = dataset complet (262k)
VAL_SUBSET    = 2000    # None = dataset complet (32k)

# ── Args ──────────────────────────────────────────────────────────────────────

parser = argparse.ArgumentParser()
parser.add_argument('--method', type=str, default='all',
                    choices=['sidetune', 'features', 'finetune', 'all'])
parser.add_argument('--epochs',       type=int, default=NUM_EPOCHS)
parser.add_argument('--train_subset', type=int, default=TRAIN_SUBSET)
args, _ = parser.parse_known_args()
NUM_EPOCHS   = args.epochs
TRAIN_SUBSET = args.train_subset

# ── Device ────────────────────────────────────────────────────────────────────

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Training on device:', device)


# ═══════════════════════════════════════════════════════════════════════════════
#  Modèles — même logique que LifelongSidetuneNetwork mais sans eval()
# ═══════════════════════════════════════════════════════════════════════════════

class FeatureExtractionPCam(nn.Module):
    """
    FEATURES — baseline naïve.
    ResNet-18 ImageNet complètement gelé + classifieur linéaire.
    Équivalent à merge_operators.BaseOnly dans le framework.
    """
    def __init__(self):
        super().__init__()
        base = tv_models.resnet18(weights=tv_models.ResNet18_Weights.DEFAULT)
        self.encoder = nn.Sequential(*list(base.children())[:-1])  # (B,512,1,1)
        for p in self.encoder.parameters():
            p.requires_grad = False
        self.transfer = nn.Linear(512, 2)   # = transfer_class dans le framework

    def forward(self, x, task_idx=None):
        with torch.no_grad():
            f = self.encoder(x).flatten(1)  # (B, 512)
        return self.transfer(f)


class FineTunePCam(nn.Module):
    """
    FINETUNE — baseline forte.
    ResNet-18 entièrement entraînable.
    Équivalent à merge_operators.SideOnly dans le framework.
    """
    def __init__(self):
        super().__init__()
        base = tv_models.resnet18(weights=tv_models.ResNet18_Weights.DEFAULT)
        self.encoder = nn.Sequential(*list(base.children())[:-1])  # (B,512,1,1)
        self.transfer = nn.Linear(512, 2)

    def forward(self, x, task_idx=None):
        f = self.encoder(x).flatten(1)  # (B, 512)
        return self.transfer(f)


class SideTunePCam(nn.Module):
    """
    SIDETUNE — notre méthode.
    Implémente T( alpha*B(x) + (1-alpha)*S(x) ) exactement comme
    LifelongSidetuneNetwork avec merge_operators.Alpha.

      B(x) = ResNet-18 ImageNet, gelé          (= base dans le framework)
      S(x) = petit CNN entraînable              (= side dans le framework)
      alpha = paramètre scalaire appris         (= merge_operators.Alpha.param)
      T(·) = nn.Linear(512, 2)                 (= transfer dans le framework)
    """
    def __init__(self):
        super().__init__()

        # Base (frozen) — ResNet-18 ImageNet
        base = tv_models.resnet18(weights=tv_models.ResNet18_Weights.DEFAULT)
        self.base = nn.Sequential(*list(base.children())[:-1])
        for p in self.base.parameters():
            p.requires_grad = False

        # Side network (trainable) — petit CNN, même output dim que la base
        self.side = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),    # 48×48
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),   # 24×24
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),                                         # 1×1
            nn.Flatten(),
            nn.Linear(128, 512), nn.ReLU(),
        )

        # Alpha blending — paramètre appris (initialisé à 0.5)
        # Correspond exactement à merge_operators.Alpha.param
        self.alpha = nn.Parameter(torch.tensor(0.5))

        # Transfer (classifieur final) — même que transfer_class='nn.Linear'
        self.transfer = nn.Linear(512, 2)

    def forward(self, x, task_idx=None):
        # Base encoding (no grad)
        with torch.no_grad():
            f_base = self.base(x).flatten(1)   # (B, 512)

        # Side encoding
        f_side = self.side(x)                  # (B, 512)

        # Alpha blending : clamp dans [0,1] comme dans merge_operators.Alpha
        alpha = self.alpha.clamp(0.0, 1.0)
        merged = alpha * f_base + (1.0 - alpha) * f_side  # (B, 512)

        return self.transfer(merged)


# ── Helpers ───────────────────────────────────────────────────────────────────

def get_subset_loader(dataset, n, shuffle):
    if n is not None and n < len(dataset):
        indices = torch.randperm(len(dataset))[:n].tolist()
        dataset = torch.utils.data.Subset(dataset, indices)
    return torch.utils.data.DataLoader(
        dataset, batch_size=BATCH_SIZE,
        shuffle=shuffle, num_workers=0, pin_memory=False  # h5py non-picklable sur Windows
    )


MODEL_CLASSES = {
    'features': FeatureExtractionPCam,
    'finetune':  FineTunePCam,
    'sidetune':  SideTunePCam,
}


# ── Experiment ────────────────────────────────────────────────────────────────

def run_experiment(method, dl_train, dl_val):
    print(f"\n{'='*60}")
    print(f"  Method : {method.upper()}")
    print(f"{'='*60}")

    os.makedirs(os.path.join(LOG_DIR, method), exist_ok=True)
    os.makedirs(SAVE_DIR, exist_ok=True)

    # ── Model ─────────────────────────────────────────────────────────────────
    model = MODEL_CLASSES[method]().to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Trainable params : {n_params:,}")

    # ── Optimizer — weight_decay=0 pour alpha (comme dans train_lifelong.py) ──
    if method == 'sidetune':
        optimizer = torch.optim.Adam([
            {'params': [model.alpha], 'weight_decay': 0.0},
            {'params': [p for n, p in model.named_parameters()
                        if n != 'alpha' and p.requires_grad]},
        ], lr=LR)
    else:
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=LR, weight_decay=0.0
        )

    # ── Logging — même structure que demo_icifar ──────────────────────────────
    mlog = tnt.logger.TensorboardMeterLogger(
        env=f'pcam_{method}',
        log_dir=os.path.join(LOG_DIR, method),
        plotstylecombined=True
    )
    mlog.add_meter('losses/task_0',
                   tnt.meter.ValueSummaryMeter())
    mlog.add_meter('accuracy_top1/task_0',
                   tnt.meter.ClassErrorMeter(topk=[1], accuracy=True))

    # ── Training loop — calqué ligne par ligne sur demo_icifar ────────────────
    print('Starting training')
    model.train(True)
    start_time = time.time()

    for epoch in range(NUM_EPOCHS):
        for x, label in tqdm(dl_train, desc=f'Epoch {epoch} (Train)'):
            x, label = x.to(device), label.to(device)

            pred = model(x, task_idx=0)
            loss = F.cross_entropy(pred, label)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            mlog.update_meter(loss.item() / x.shape[0],
                              meters={'losses/task_0'}, phase='train')
            mlog.update_meter(pred, target=label,
                              meters={'accuracy_top1/task_0'}, phase='train')

    elapsed = time.time() - start_time
    meter_dict = mlog.peek_meter('train')
    print(f'Finished training in: {elapsed:.1f}s')
    print(f'\t Task 0 Train Loss    : {meter_dict["losses/task_0"].item():.4f}')
    print(f'\t Task 0 Train Accuracy: {meter_dict["accuracy_top1/task_0"]:.4f}')

    if method == 'sidetune':
        print(f'\t Alpha (base weight)  : {model.alpha.item():.4f}'
              f'  (1=base only, 0=side only)')

    # ── Validation — calqué sur demo_icifar ───────────────────────────────────
    model.to(device)
    model.train(False)
    with torch.no_grad():
        for x, label in tqdm(dl_val, desc='Validation'):
            x, label = x.to(device), label.to(device)
            pred = model(x, task_idx=0)
            loss = F.cross_entropy(pred, label)

            mlog.update_meter(loss.item() / x.shape[0],
                              meters={'losses/task_0'}, phase='val')
            mlog.update_meter(pred, target=label,
                              meters={'accuracy_top1/task_0'}, phase='val')

    meter_dict = mlog.peek_meter('val')
    val_loss = meter_dict['losses/task_0'].item()
    val_acc  = meter_dict['accuracy_top1/task_0']
    print('Results (Validation):')
    print(f'\t Task 0 Val Loss    : {val_loss:.4f}')
    print(f'\t Task 0 Val Accuracy: {val_acc:.4f}')

    # ── Save checkpoint ────────────────────────────────────────────────────────
    save_path = os.path.join(SAVE_DIR, f'model_{method}.pth')
    torch.save(model.state_dict(), save_path)
    print(f'Model saved to {save_path}')

    return {'val_loss': val_loss, 'val_acc': val_acc}


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print('Loading PCam datasets...')
    train_ds, val_ds, _ = get_pcam_datasets(PCAM_DATA_DIR)

    dl_train = get_subset_loader(train_ds, TRAIN_SUBSET, shuffle=True)
    dl_val   = get_subset_loader(val_ds,   VAL_SUBSET,   shuffle=False)
    print(f'Train subset : {len(dl_train.dataset)} | '
          f'Val subset   : {len(dl_val.dataset)}')

    methods = ['features', 'finetune', 'sidetune'] \
              if args.method == 'all' else [args.method]

    results = {}
    for method in methods:
        results[method] = run_experiment(method, dl_train, dl_val)

    # Résumé final — comme avg_acc dans demo_icifar
    print(f"\n{'='*60}")
    print('  SUMMARY — Val Accuracy')
    print(f"{'='*60}")
    for method, res in results.items():
        print(f'  {method:<12} val_acc={res["val_acc"]:.4f}  '
              f'val_loss={res["val_loss"]:.4f}')

    avg_acc = np.mean([r['val_acc'] for r in results.values()])
    print(f'\n  Average Val Accuracy : {avg_acc:.4f}')


if __name__ == '__main__':
    main()