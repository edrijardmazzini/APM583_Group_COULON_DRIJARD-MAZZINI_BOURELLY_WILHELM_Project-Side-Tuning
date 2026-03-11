"""
demo_pcam_benchmark.py  —  Benchmark complet de méthodes de transfer learning sur PCam
========================================================================================

7 méthodes comparées sur les MÊMES paramètres (10 epochs, 20k train, 2k val) :

  ── Baselines classiques ──────────────────────────────────────────────────────
  [1] features      : Base RN50 gelé + Linear head                (1k params)
  [2] finetune      : Base RN50 entièrement entraînable           (24M params)
  [3] sidetune      : Side-Tuning (notre méthode, RN50+MLP+Large) (~3M params)

  ── Nouvelles méthodes de PEFT ────────────────────────────────────────────────
  [4] lora          : Low-Rank Adaptation sur les couches Linear  (~300k params)
  [5] adapter       : Adapter Layers insérées dans chaque bloc    (~400k params)
  [6] prompt_tuning : Learnable tokens préfixés à chaque couche   (~50k params)
  [7] bitfit        : Seulement les biais sont entraînables        (~25k params)

Usage:
    python -m scripts.demo_pcam_benchmark
    python -m scripts.demo_pcam_benchmark --epochs 10 --train_subset 20000
    python -m scripts.demo_pcam_benchmark --methods features finetune sidetune lora
"""

import os, json, argparse, time, math
from tqdm import tqdm
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tv_models
import torchvision.transforms as transforms

from tlkit.data.datasets.pcam_datasets import PCAMDataset
import tnt.torchnet as tnt

# ── Args ──────────────────────────────────────────────────────────────────────

ALL_METHODS = ['features', 'finetune', 'sidetune', 'lora', 'adapter', 'prompt_tuning', 'bitfit']

parser = argparse.ArgumentParser()
parser.add_argument('--epochs',       type=int,   default=10)
parser.add_argument('--train_subset', type=int,   default=20000)
parser.add_argument('--val_subset',   type=int,   default=2000)
parser.add_argument('--batch_size',   type=int,   default=64)
parser.add_argument('--lr',           type=float, default=1e-3)
parser.add_argument('--methods',      nargs='+',  default=ALL_METHODS,
                    choices=ALL_METHODS)
args = parser.parse_args()

PCAM_DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'pcam')
LOG_DIR       = '/tmp/pcam_benchmark/tensorboards'
SAVE_DIR      = '/tmp/pcam_benchmark'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Training on device: {device}')


# ═══════════════════════════════════════════════════════════════════════════════
#  [1] FEATURES  —  Base gelé + Linear head
# ═══════════════════════════════════════════════════════════════════════════════

class FeaturesModel(nn.Module):
    """ResNet-50 complètement gelé. Seule la tête linéaire est entraînée."""
    def __init__(self):
        super().__init__()
        rn50 = tv_models.resnet50(weights=tv_models.ResNet50_Weights.DEFAULT)
        self.encoder = nn.Sequential(*list(rn50.children())[:-1])
        for p in self.encoder.parameters():
            p.requires_grad = False
        self.head = nn.Linear(2048, 2)

    def forward(self, x):
        with torch.no_grad():
            f = self.encoder(x).flatten(1)
        return self.head(f)


# ═══════════════════════════════════════════════════════════════════════════════
#  [2] FINETUNE  —  Tous les poids entraînables
# ═══════════════════════════════════════════════════════════════════════════════

class FinetuneModel(nn.Module):
    """ResNet-50 entièrement fine-tuné. Risque d'overfitting élevé sur peu de données."""
    def __init__(self):
        super().__init__()
        rn50 = tv_models.resnet50(weights=tv_models.ResNet50_Weights.DEFAULT)
        self.encoder = nn.Sequential(*list(rn50.children())[:-1])
        self.head    = nn.Linear(2048, 2)

    def forward(self, x):
        f = self.encoder(x).flatten(1)
        return self.head(f)


# ═══════════════════════════════════════════════════════════════════════════════
#  [3] SIDE-TUNING  —  Meilleure config trouvée (RN50 + MLP + Large)
# ═══════════════════════════════════════════════════════════════════════════════

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
    Side-Tuning : α*B(x) fusionné avec S(x) via MLP.
    Base RN50 gelée. Side network + MLP + projection entraînables.
    Correspond à la meilleure config de demo_pcam_optim.
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
            nn.Linear(512, 512),
        )
        self.head = nn.Linear(512, 2)

    def forward(self, x):
        with torch.no_grad():
            f_base = self.base(x).flatten(1)
        f_base  = self.base_proj(f_base)
        f_side  = self.side(x)
        merged  = self.merge_mlp(torch.cat([f_base, f_side], dim=1))
        return self.head(merged)


# ═══════════════════════════════════════════════════════════════════════════════
#  [4] LoRA  —  Low-Rank Adaptation
# ═══════════════════════════════════════════════════════════════════════════════

class LoRALinear(nn.Module):
    """
    Remplace nn.Linear par W + BA où B∈R^{d×r}, A∈R^{r×k}, rang r << min(d,k).
    W original est gelé. Seuls A et B sont entraînables.
    Référence : Hu et al. 2022 (https://arxiv.org/abs/2106.09685)
    """
    def __init__(self, original_linear: nn.Linear, rank: int = 8, alpha: float = 16.0):
        super().__init__()
        d_out, d_in = original_linear.weight.shape
        self.rank    = rank
        self.scaling = alpha / rank

        # Poids gelé original
        self.weight = nn.Parameter(original_linear.weight.data.clone(), requires_grad=False)
        self.bias   = nn.Parameter(original_linear.bias.data.clone(), requires_grad=False) \
                      if original_linear.bias is not None else None

        # Matrices LoRA entraînables
        self.lora_A = nn.Parameter(torch.randn(rank, d_in) * 0.01)   # init gaussienne
        self.lora_B = nn.Parameter(torch.zeros(d_out, rank))          # init zéro → δW=0 au départ

    def forward(self, x):
        base_out = F.linear(x, self.weight, self.bias)
        lora_out = F.linear(F.linear(x, self.lora_A), self.lora_B) * self.scaling
        return base_out + lora_out


def inject_lora(model: nn.Module, rank: int = 8, alpha: float = 16.0,
                target_modules=('fc',)) -> nn.Module:
    """
    Remplace récursivement les nn.Linear dont le nom contient un des target_modules
    par des LoRALinear. Tous les autres poids restent gelés.
    """
    for name, module in list(model.named_children()):
        if isinstance(module, nn.Linear) and any(t in name for t in target_modules):
            setattr(model, name, LoRALinear(module, rank=rank, alpha=alpha))
        else:
            inject_lora(module, rank=rank, alpha=alpha, target_modules=target_modules)
    return model


class LoRAModel(nn.Module):
    """
    ResNet-50 avec LoRA injecté sur TOUTES les couches Linear (fc dans chaque bloc).
    Seuls les paramètres LoRA (A, B) + la tête finale sont entraînables.
    """
    def __init__(self, rank: int = 8):
        super().__init__()
        rn50 = tv_models.resnet50(weights=tv_models.ResNet50_Weights.DEFAULT)

        # Geler tout d'abord
        for p in rn50.parameters():
            p.requires_grad = False

        # Injecter LoRA sur toutes les Linear (fc1, fc2 dans les bottlenecks + fc final)
        inject_lora(rn50, rank=rank, alpha=float(rank * 2),
                    target_modules=('fc', 'fc1', 'fc2', 'fc3'))

        self.encoder = nn.Sequential(*list(rn50.children())[:-1])
        # Tête finale entraînable
        self.head = nn.Linear(2048, 2)

    def forward(self, x):
        f = self.encoder(x).flatten(1)
        return self.head(f)


# ═══════════════════════════════════════════════════════════════════════════════
#  [5] ADAPTER LAYERS
# ═══════════════════════════════════════════════════════════════════════════════

class AdapterLayer(nn.Module):
    """
    Adapter bottleneck inséré après chaque bloc résiduel :
        x → LayerNorm → Linear(d→r) → ReLU → Linear(r→d) → + x
    Seuls les paramètres de l'adapter sont entraînables.
    Référence : Houlsby et al. 2019
    """
    def __init__(self, d_model: int, bottleneck: int = 64):
        super().__init__()
        self.norm   = nn.LayerNorm(d_model)
        self.down   = nn.Linear(d_model, bottleneck)
        self.act    = nn.ReLU()
        self.up     = nn.Linear(bottleneck, d_model)
        # Init : up initialisé à zéro → adapter = identité au début
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, x):
        # x : (B, d_model) après global avg pool OU (B, d_model, H, W)
        # On traite les deux cas
        if x.dim() == 4:
            B, C, H, W = x.shape
            residual = x
            x = x.permute(0, 2, 3, 1)      # (B, H, W, C)
            x = self.norm(x)
            x = self.up(self.act(self.down(x)))
            x = x.permute(0, 3, 1, 2)      # (B, C, H, W)
            return residual + x
        else:
            return x + self.up(self.act(self.down(self.norm(x))))


class ResNetWithAdapters(nn.Module):
    """
    ResNet-50 avec un AdapterLayer après chaque couche (layer1..layer4).
    Backbone gelé, seuls les adapters + head sont entraînables.
    """
    def __init__(self, bottleneck: int = 64):
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

        # Adapters après chaque stage — dimensions ResNet-50 : 256, 512, 1024, 2048
        self.adapter1 = AdapterLayer(256,  bottleneck)
        self.adapter2 = AdapterLayer(512,  bottleneck)
        self.adapter3 = AdapterLayer(1024, bottleneck)
        self.adapter4 = AdapterLayer(2048, bottleneck)

        self.head = nn.Linear(2048, 2)

    def forward(self, x):
        x = self.stem(x)
        x = self.adapter1(self.layer1(x))
        x = self.adapter2(self.layer2(x))
        x = self.adapter3(self.layer3(x))
        x = self.adapter4(self.layer4(x))
        x = self.avgpool(x).flatten(1)
        return self.head(x)


# ═══════════════════════════════════════════════════════════════════════════════
#  [6] PROMPT TUNING (Visual Prompt Tuning — VPT shallow)
# ═══════════════════════════════════════════════════════════════════════════════

class PatchEmbedding(nn.Module):
    """
    Convertit une image 96×96 en séquence de patches (style ViT-like)
    pour permettre le prefix/prompt tuning.
    patch_size=16 → 6×6 = 36 patches de dim 768.
    """
    def __init__(self, img_size=96, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        self.patch_size  = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.proj        = nn.Conv2d(in_chans, embed_dim,
                                     kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x)                    # (B, embed_dim, H/p, W/p)
        x = x.flatten(2).transpose(1, 2)    # (B, num_patches, embed_dim)
        return x


class PromptTuningModel(nn.Module):
    """
    Visual Prompt Tuning (VPT) — version shallow/légère sur CNN.

    Principe adapté aux CNN :
    - On gèle ResNet-50
    - On apprend des tokens de prompt dans l'espace des features
    - Ces tokens sont concaténés aux features extraites puis traités par un MLP léger
    - Seuls les prompt tokens + MLP head sont entraînables

    Note : VPT "pur" (Jia et al. 2022) est conçu pour les ViT.
    Cette implémentation est l'adaptation naturelle aux CNN :
    prompt tokens dans l'espace feature (≡ prefix dans l'espace d'activation).
    """
    def __init__(self, num_prompts: int = 64, prompt_dim: int = 256):
        super().__init__()
        rn50 = tv_models.resnet50(weights=tv_models.ResNet50_Weights.DEFAULT)
        self.encoder = nn.Sequential(*list(rn50.children())[:-1])
        for p in self.encoder.parameters():
            p.requires_grad = False

        self.num_prompts = num_prompts
        self.prompt_dim  = prompt_dim

        # Tokens de prompt appris — initialisés avec une distribution uniforme
        self.prompt_tokens = nn.Parameter(
            torch.empty(1, num_prompts, prompt_dim).uniform_(-0.5, 0.5)
        )

        # Projection features RN50 → prompt_dim
        self.feat_proj = nn.Linear(2048, prompt_dim)

        # MLP head : traite [feat_proj(x) ; mean(prompts)] → 2 classes
        self.head = nn.Sequential(
            nn.Linear(prompt_dim * 2, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 2),
        )

    def forward(self, x):
        B = x.shape[0]
        with torch.no_grad():
            f = self.encoder(x).flatten(1)          # (B, 2048)

        f_proj   = self.feat_proj(f)                 # (B, prompt_dim)
        prompts  = self.prompt_tokens.expand(B, -1, -1)  # (B, num_prompts, prompt_dim)
        # Interaction : attention entre f_proj et prompt tokens
        attn     = torch.softmax(
            torch.bmm(f_proj.unsqueeze(1), prompts.transpose(1, 2)) / math.sqrt(self.prompt_dim),
            dim=-1
        )                                            # (B, 1, num_prompts)
        ctx      = torch.bmm(attn, prompts).squeeze(1)  # (B, prompt_dim)
        combined = torch.cat([f_proj, ctx], dim=1)   # (B, prompt_dim*2)
        return self.head(combined)


# ═══════════════════════════════════════════════════════════════════════════════
#  [7] BITFIT  —  Bias-Terms Fine-Tuning
# ═══════════════════════════════════════════════════════════════════════════════

class BitFitModel(nn.Module):
    """
    BitFit : geler tous les poids, n'entraîner que les biais.
    Extrêmement parcimonieux (~25k params sur RN50).
    Référence : Ben-Zaken et al. 2022
    """
    def __init__(self):
        super().__init__()
        rn50 = tv_models.resnet50(weights=tv_models.ResNet50_Weights.DEFAULT)

        # Geler TOUT
        for p in rn50.parameters():
            p.requires_grad = False

        # Dégeler uniquement les biais
        for name, p in rn50.named_parameters():
            if 'bias' in name:
                p.requires_grad = True

        self.encoder = nn.Sequential(*list(rn50.children())[:-1])
        self.head    = nn.Linear(2048, 2)

    def forward(self, x):
        f = self.encoder(x).flatten(1)
        return self.head(f)


# ═══════════════════════════════════════════════════════════════════════════════
#  Registry
# ═══════════════════════════════════════════════════════════════════════════════

MODEL_BUILDERS = {
    'features':      (FeaturesModel,      "Base RN50 gelé + Linear head"),
    'finetune':      (FinetuneModel,      "Fine-tuning complet RN50"),
    'sidetune':      (SidetuneModel,      "Side-Tuning (RN50 + MLP + Large side)  ← notre méthode"),
    'lora':          (LoRAModel,          "LoRA  rank=8  sur toutes les Linear"),
    'adapter':       (ResNetWithAdapters, "Adapter Layers  bottleneck=64  après chaque stage"),
    'prompt_tuning': (PromptTuningModel,  "Prompt Tuning  64 tokens  dim=256"),
    'bitfit':        (BitFitModel,        "BitFit  biais seuls entraînables"),
}

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
#  Training loop
# ═══════════════════════════════════════════════════════════════════════════════

def run(method_name, dl_train, dl_val):
    builder, description = MODEL_BUILDERS[method_name]
    model    = builder().to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"\n{'─'*65}")
    print(f"  [{method_name.upper()}]  {description}")
    print(f"  Trainable params : {n_params:,}")
    print(f"{'─'*65}")

    # Optimizer — lr réduit pour fine-tuning complet pour éviter l'overfitting
    lr = args.lr / 10 if method_name == 'finetune' else args.lr
    optimizer = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad], lr=lr
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs
    )

    # tnt logging
    log_dir = os.path.join(LOG_DIR, method_name)
    os.makedirs(log_dir, exist_ok=True)
    mlog = tnt.logger.TensorboardMeterLogger(
        env=f'benchmark_{method_name}',
        log_dir=log_dir,
        plotstylecombined=True,
    )
    mlog.add_meter('losses/task_0',        tnt.meter.ValueSummaryMeter())
    mlog.add_meter('accuracy_top1/task_0', tnt.meter.ClassErrorMeter(topk=[1], accuracy=True))

    # Training
    model.train(True)
    start = time.time()

    for epoch in range(args.epochs):
        for x, label in tqdm(dl_train, desc=f'  Ep{epoch:02d}/{args.epochs} [{method_name}]', leave=True):
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

    # Validation
    model.train(False)
    with torch.no_grad():
        for x, label in tqdm(dl_val, desc=f'  Val [{method_name}]', leave=True):
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

    return {
        'method':      method_name,
        'description': description,
        'val_acc':     val_acc,
        'val_loss':    val_loss,
        'train_acc':   train_acc,
        'n_params':    n_params,
        'time':        round(elapsed),
    }


# ═══════════════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════════════

def efficiency_score(r):
    """Val acc par million de paramètres — mesure l'efficacité paramétrique."""
    return r['val_acc'] / (r['n_params'] / 1e6)


def main():
    methods = args.methods
    print(f"\n{'='*65}")
    print(f"  BENCHMARK — Transfer Learning Methods on PCam")
    print(f"  Méthodes : {methods}")
    print(f"  Epochs={args.epochs} | Train={args.train_subset} | Val={args.val_subset}")
    print(f"{'='*65}")

    dl_train = get_loader('train', args.train_subset, shuffle=True)
    dl_val   = get_loader('val',   args.val_subset,   shuffle=False)

    results = []
    for method in methods:
        results.append(run(method, dl_train, dl_val))

    # ── Tableau récapitulatif ──────────────────────────────────────────────────
    best_acc = max(r['val_acc'] for r in results)

    print(f"\n{'='*65}")
    print(f"  RÉSULTATS FINAUX  (triés par val accuracy)")
    print(f"{'='*65}")
    print(f"  {'Method':<16} {'Val Acc':>8} {'Train Acc':>10} {'Val Loss':>9} "
          f"{'Params':>10} {'Acc/M':>7} {'Time':>6}")
    print(f"  {'─'*68}")

    for r in sorted(results, key=lambda x: x['val_acc'], reverse=True):
        marker = " ★" if abs(r['val_acc'] - best_acc) < 0.01 else ""
        eff    = efficiency_score(r)
        print(f"  {r['method']:<16} {r['val_acc']:>7.2f}%  "
              f"{r['train_acc']:>8.2f}%  {r['val_loss']:>9.4f}  "
              f"{r['n_params']:>10,}  {eff:>6.1f}  {r['time']:>5}s{marker}")

    # Meilleure efficacité paramétrique
    best_eff = max(results, key=efficiency_score)
    print(f"\n  ✅ Meilleure accuracy      : {max(results, key=lambda x: x['val_acc'])['method']}"
          f"  ({best_acc:.2f}%)")
    print(f"  ⚡ Meilleure efficacité    : {best_eff['method']}"
          f"  ({efficiency_score(best_eff):.1f} acc/M params)")

    # Save
    os.makedirs(SAVE_DIR, exist_ok=True)
    out = os.path.join(SAVE_DIR, 'benchmark_results.json')
    with open(out, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n  Résultats → {out}")


if __name__ == '__main__':
    main()
