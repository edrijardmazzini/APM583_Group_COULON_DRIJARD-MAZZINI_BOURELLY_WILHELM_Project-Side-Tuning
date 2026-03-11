"""
demo_final.py  —  Benchmark final : Adapter vs Finetune vs Sidetune
====================================================================

Les 3 meilleures méthodes du benchmark comparées sur 50k images train.

  [1] adapter   : Adapter Layers bottleneck=64 après chaque stage RN50
  [2] finetune  : Fine-tuning complet RN50
  [3] sidetune  : Side-Tuning (RN50 + MLP + Large side)  ← notre méthode

Résultats sauvegardés dans :
    <repo>/results/final_results.json
    <repo>/results/final_log.txt

Usage:
    python -m scripts.demo_final
    python -m scripts.demo_final --epochs 10 --train_subset 50000
"""

import os, json, time, argparse
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tv_models
import torchvision.transforms as transforms

from tlkit.data.datasets.pcam_datasets import PCAMDataset
import tnt.torchnet as tnt

# ── Paths ─────────────────────────────────────────────────────────────────────

# __file__ = side-tuning/scripts/demo_final.py  →  REPO_DIR = side-tuning/ 
# NON AU FINAL
# __file__ = side-tuning/scripts/demo_final.py  →  REPO_DIR = side-tuning/scripts/results/
REPO_DIR      = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
PCAM_DATA_DIR = os.path.join(REPO_DIR, 'data', 'pcam')
RESULTS_DIR   = os.path.join(REPO_DIR, 'scripts', 'results')
LOG_DIR       = os.path.join(RESULTS_DIR, 'tensorboards', 'final')

# ── Args ──────────────────────────────────────────────────────────────────────

parser = argparse.ArgumentParser()
parser.add_argument('--epochs',       type=int,   default=10)
parser.add_argument('--train_subset', type=int,   default=50000)
parser.add_argument('--val_subset',   type=int,   default=2000)
parser.add_argument('--batch_size',   type=int,   default=64)
parser.add_argument('--lr',           type=float, default=1e-3)
parser.add_argument('--methods',      nargs='+',
                    default=['adapter', 'finetune', 'sidetune'],
                    choices=['adapter', 'finetune', 'sidetune'])
args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ═══════════════════════════════════════════════════════════════════════════════
#  Models
# ═══════════════════════════════════════════════════════════════════════════════

# ── Adapter ───────────────────────────────────────────────────────────────────

class AdapterLayer(nn.Module):
    """
    Bottleneck résiduel inséré après chaque stage ResNet :
        x → LayerNorm → Linear(d→r) → ReLU → Linear(r→d) → + x
    Init à zéro → identité au départ, s'adapte progressivement.
    Référence : Houlsby et al. 2019
    """
    def __init__(self, d_model, bottleneck=64):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.down = nn.Linear(d_model, bottleneck)
        self.act  = nn.ReLU()
        self.up   = nn.Linear(bottleneck, d_model)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, x):
        if x.dim() == 4:
            res = x
            x = x.permute(0, 2, 3, 1)
            x = self.up(self.act(self.down(self.norm(x))))
            return res + x.permute(0, 3, 1, 2)
        return x + self.up(self.act(self.down(self.norm(x))))


class AdapterModel(nn.Module):
    def __init__(self, bottleneck=64):
        super().__init__()
        rn50 = tv_models.resnet50(weights=tv_models.ResNet50_Weights.DEFAULT)
        for p in rn50.parameters():
            p.requires_grad = False

        self.stem    = nn.Sequential(rn50.conv1, rn50.bn1, rn50.relu, rn50.maxpool)
        self.layer1  = rn50.layer1
        self.layer2  = rn50.layer2
        self.layer3  = rn50.layer3
        self.layer4  = rn50.layer4
        self.avgpool = rn50.avgpool

        self.adapter1 = AdapterLayer(256,  bottleneck)
        self.adapter2 = AdapterLayer(512,  bottleneck)
        self.adapter3 = AdapterLayer(1024, bottleneck)
        self.adapter4 = AdapterLayer(2048, bottleneck)
        self.head     = nn.Linear(2048, 2)

    def forward(self, x):
        x = self.stem(x)
        x = self.adapter1(self.layer1(x))
        x = self.adapter2(self.layer2(x))
        x = self.adapter3(self.layer3(x))
        x = self.adapter4(self.layer4(x))
        return self.head(self.avgpool(x).flatten(1))


# ── Finetune ──────────────────────────────────────────────────────────────────

class FinetuneModel(nn.Module):
    def __init__(self):
        super().__init__()
        rn50 = tv_models.resnet50(weights=tv_models.ResNet50_Weights.DEFAULT)
        self.encoder = nn.Sequential(*list(rn50.children())[:-1])
        self.head    = nn.Linear(2048, 2)

    def forward(self, x):
        return self.head(self.encoder(x).flatten(1))


# ── Side-Tuning ───────────────────────────────────────────────────────────────

class SideNetLarge(nn.Module):
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


class SidetuneModel(nn.Module):
    """
    Meilleure config identifiée dans demo_pcam_optim :
    Base RN50 gelée + Side Large + MLP merge → 85.90% à 10ep/20k.
    """
    def __init__(self):
        super().__init__()
        rn50 = tv_models.resnet50(weights=tv_models.ResNet50_Weights.DEFAULT)
        self.base = nn.Sequential(*list(rn50.children())[:-1])
        for p in self.base.parameters():
            p.requires_grad = False

        self.base_proj = nn.Linear(2048, 512)
        self.side      = SideNetLarge()
        self.merge_mlp = nn.Sequential(
            nn.Linear(1024, 512), nn.ReLU(),
            nn.Linear(512,  512),
        )
        self.head = nn.Linear(512, 2)

    def forward(self, x):
        with torch.no_grad():
            f_base = self.base(x).flatten(1)
        f_base = self.base_proj(f_base)
        f_side = self.side(x)
        return self.head(self.merge_mlp(torch.cat([f_base, f_side], dim=1)))


# ── Registry ──────────────────────────────────────────────────────────────────

MODEL_BUILDERS = {
    'adapter':  (AdapterModel,  "Adapter Layers  bottleneck=64  après chaque stage RN50"),
    'finetune': (FinetuneModel, "Fine-tuning complet RN50  (lr=lr/10, cosine scheduler)"),
    'sidetune': (SidetuneModel, "Side-Tuning  RN50 + MLP + Large side  ← notre méthode"),
}


# ═══════════════════════════════════════════════════════════════════════════════
#  Data
# ═══════════════════════════════════════════════════════════════════════════════

T_TRAIN = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
T_EVAL = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def get_loader(split, n, shuffle):
    x_map = {'train': 'training_split.h5',                          'val': 'validation_split.h5'}
    y_map = {'train': 'camelyonpatch_level_2_split_train_y.h5',     'val': 'camelyonpatch_level_2_split_valid_y.h5'}
    ds = PCAMDataset(
        x_path=os.path.join(PCAM_DATA_DIR, x_map[split]),
        y_path=os.path.join(PCAM_DATA_DIR, y_map[split]),
        transform=T_TRAIN if split == 'train' else T_EVAL,
    )
    if n and n < len(ds):
        ds = torch.utils.data.Subset(ds, torch.randperm(len(ds))[:n].tolist())
    return torch.utils.data.DataLoader(
        ds, batch_size=args.batch_size,
        shuffle=shuffle, num_workers=0, pin_memory=False,
    )


# ═══════════════════════════════════════════════════════════════════════════════
#  TeeLogger — écrit dans le terminal ET dans final_log.txt
# ═══════════════════════════════════════════════════════════════════════════════

class TeeLogger:
    def __init__(self, filepath):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        self.file = open(filepath, 'w', encoding='utf-8')

    def log(self, msg=''):
        print(msg)
        self.file.write(msg + '\n')
        self.file.flush()

    def close(self):
        self.file.close()


# ═══════════════════════════════════════════════════════════════════════════════
#  Training loop
# ═══════════════════════════════════════════════════════════════════════════════

def run(method_name, dl_train, dl_val, logger):
    builder, description = MODEL_BUILDERS[method_name]
    model    = builder().to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    logger.log(f"\n{'─'*65}")
    logger.log(f"  [{method_name.upper()}]  {description}")
    logger.log(f"  Trainable params : {n_params:,}")
    logger.log(f"{'─'*65}")

    lr = args.lr / 10 if method_name == 'finetune' else args.lr
    optimizer = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad], lr=lr
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs
    )

    log_dir = os.path.join(LOG_DIR, method_name)
    os.makedirs(log_dir, exist_ok=True)
    mlog = tnt.logger.TensorboardMeterLogger(
        env=f'final_{method_name}',
        log_dir=log_dir,
        plotstylecombined=True,
    )
    mlog.add_meter('losses/task_0',        tnt.meter.ValueSummaryMeter())
    mlog.add_meter('accuracy_top1/task_0', tnt.meter.ClassErrorMeter(topk=[1], accuracy=True))

    model.train(True)
    start = time.time()

    for epoch in range(args.epochs):
        for x, label in tqdm(dl_train,
                             desc=f'  Ep{epoch:02d}/{args.epochs} [{method_name}]',
                             leave=True):
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
        scheduler.step()

    elapsed   = time.time() - start
    train_acc = mlog.peek_meter('train')['accuracy_top1/task_0']

    model.train(False)
    with torch.no_grad():
        for x, label in tqdm(dl_val,
                             desc=f'  Val [{method_name}]',
                             leave=True):
            x, label = x.to(device), label.to(device)
            pred = model(x)
            loss = F.cross_entropy(pred, label)
            mlog.update_meter(loss.item() / x.shape[0],
                              meters={'losses/task_0'}, phase='val')
            mlog.update_meter(pred, target=label,
                              meters={'accuracy_top1/task_0'}, phase='val')

    val_acc  = mlog.peek_meter('val')['accuracy_top1/task_0']
    val_loss = mlog.peek_meter('val')['losses/task_0'].item()

    logger.log(f"  ✓ train={train_acc:.2f}%  val={val_acc:.2f}%  "
               f"loss={val_loss:.4f}  {elapsed:.0f}s")

    ckpt = os.path.join(RESULTS_DIR, f'final_{method_name}.pth')
    torch.save(model.state_dict(), ckpt)
    logger.log(f"  Checkpoint → {ckpt}")

    return {
        'method':        method_name,
        'description':   description,
        'val_acc':       val_acc,
        'val_loss':      val_loss,
        'train_acc':     train_acc,
        'n_params':      n_params,
        'epochs':        args.epochs,
        'train_subset':  args.train_subset,
        'time':          round(elapsed),
    }


# ═══════════════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    logger = TeeLogger(os.path.join(RESULTS_DIR, 'final_log.txt'))

    logger.log(f"{'='*65}")
    logger.log(f"  BENCHMARK FINAL — Adapter vs Finetune vs Sidetune")
    logger.log(f"  Méthodes      : {args.methods}")
    logger.log(f"  Epochs        : {args.epochs}")
    logger.log(f"  Train subset  : {args.train_subset}")
    logger.log(f"  Val subset    : {args.val_subset}")
    logger.log(f"  Device        : {device}")
    logger.log(f"{'='*65}")

    dl_train = get_loader('train', args.train_subset, shuffle=True)
    dl_val   = get_loader('val',   args.val_subset,   shuffle=False)

    results = []
    for method in args.methods:
        results.append(run(method, dl_train, dl_val, logger))

    best_acc = max(r['val_acc'] for r in results)

    logger.log(f"\n{'='*65}")
    logger.log(f"  RÉSULTATS FINAUX  —  Train={args.train_subset} | Epochs={args.epochs}")
    logger.log(f"{'='*65}")
    logger.log(f"  {'Méthode':<14} {'Val Acc':>8} {'Train Acc':>10} "
               f"{'Val Loss':>9} {'Params':>10} {'Time':>6}")
    logger.log(f"  {'─'*60}")

    for r in sorted(results, key=lambda x: x['val_acc'], reverse=True):
        marker = " ★" if abs(r['val_acc'] - best_acc) < 0.01 else ""
        logger.log(
            f"  {r['method']:<14} {r['val_acc']:>7.2f}%  "
            f"{r['train_acc']:>8.2f}%  {r['val_loss']:>9.4f}  "
            f"{r['n_params']:>10,}  {r['time']:>5}s{marker}"
        )

    best = max(results, key=lambda x: x['val_acc'])
    logger.log(f"\n  ✅ Meilleure méthode : {best['method']}  ({best['val_acc']:.2f}%)")

    json_path = os.path.join(RESULTS_DIR, 'final_results.json')
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)

    logger.log(f"\n  JSON → {json_path}")
    logger.log(f"  Log  → {os.path.join(RESULTS_DIR, 'final_log.txt')}")
    logger.close()


if __name__ == '__main__':
    main()
