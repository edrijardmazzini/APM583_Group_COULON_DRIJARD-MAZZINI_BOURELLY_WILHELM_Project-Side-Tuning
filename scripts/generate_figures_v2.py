"""
generate_figures.py  —  Génère toutes les figures du rapport LaTeX
===================================================================

À lancer depuis la racine du repo :
    python scripts/generate_figures.py

Produit dans figures/ :
    fig0_icifar.pdf             — iCIFAR : val/train accuracy par tâche
    fig1_pcam_samples.png       — exemples d'images PCam (depuis h5)
    fig2_baseline_datasize.pdf  — val + train accuracy vs epochs / data size
    fig3_ablation_heatmap.pdf   — heatmap side_size × epochs (val + train)
    fig4_benchmark_bar.pdf      — bar chart 7 méthodes val + train
    fig5_scatter_params.pdf     — scatter params vs val_acc (layout corrigé)

Les données sont chargées depuis les fichiers JSON dans results/ si disponibles,
sinon les valeurs codées en dur servent de fallback.
"""

import os
import json
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
import h5py

matplotlib.rcParams.update({
    'font.family':        'serif',
    'font.size':          11,
    'axes.titlesize':     12,
    'axes.labelsize':     11,
    'xtick.labelsize':    10,
    'ytick.labelsize':    10,
    'legend.fontsize':    10,
    'figure.dpi':         150,
    'savefig.dpi':        300,
    'savefig.bbox':       'tight',
    'savefig.pad_inches': 0.05,
})

REPO_DIR    = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
FIGURES_DIR = os.path.join(REPO_DIR, 'figures_v2')
PCAM_DIR    = os.path.join(REPO_DIR, 'data', 'pcam')
RESULTS_DIR = os.path.join(REPO_DIR, 'scripts', 'results')
os.makedirs(FIGURES_DIR, exist_ok=True)


# ══════════════════════════════════════════════════════════════════════════════
#  Chargement des JSON (fallback sur valeurs codées en dur si absent)
# ══════════════════════════════════════════════════════════════════════════════

def load_json(path, fallback):
    """Charge un JSON si le fichier existe, sinon retourne le fallback."""
    if os.path.exists(path):
        with open(path) as f:
            data = json.load(f)
        print(f"  ✓ JSON chargé : {path}")
        return data
    else:
        print(f"  ✗ JSON non trouvé ({path}), utilisation des valeurs de fallback")
        return fallback

# Benchmark 7 méthodes — 20k, 10 epochs
BENCHMARK = load_json(
    os.path.join(RESULTS_DIR, 'benchmark_results.json'),
    fallback=[
        {'method': 'Adapter',       'val_acc': 86.65, 'train_acc': 92.62, 'val_loss': 0.0064, 'n_params': 507394},
        {'method': 'Finetune',      'val_acc': 86.10, 'train_acc': 95.85, 'val_loss': 0.0119, 'n_params': 23512130},
        {'method': 'Side-Tuning',   'val_acc': 82.65, 'train_acc': 88.42, 'val_loss': 0.0071, 'n_params': 2931458},
        {'method': 'BitFit',        'val_acc': 82.65, 'train_acc': 85.83, 'val_loss': 0.0067, 'n_params': 30658},
        {'method': 'Prompt Tuning', 'val_acc': 81.65, 'train_acc': 83.37, 'val_loss': 0.0069, 'n_params': 672770},
        {'method': 'LoRA',          'val_acc': 81.60, 'train_acc': 81.89, 'val_loss': 0.0071, 'n_params': 4098},
        {'method': 'Features',      'val_acc': 80.85, 'train_acc': 81.91, 'val_loss': 0.0072, 'n_params': 4098},
    ]
)

# Benchmark final — 50k, 10 epochs
BENCHMARK_FINAL = load_json(
    os.path.join(RESULTS_DIR, 'final_results.json'),
    fallback=[
        {'method': 'Adapter',     'val_acc': 88.40, 'train_acc': 93.83, 'val_loss': 0.0065, 'n_params': 507394},
        {'method': 'Finetune',    'val_acc': 87.55, 'train_acc': 97.06, 'val_loss': 0.0113, 'n_params': 23512130},
        {'method': 'Side-Tuning', 'val_acc': 85.60, 'train_acc': 90.70, 'val_loss': 0.0070, 'n_params': 2931458},
    ]
)

# Ablation — 8 configs
ABLATION = load_json(
    os.path.join(RESULTS_DIR, 'ablation_all.json'),
    fallback=[]  # pas utilisé dans les figures ici
)

# Optim grille 3×3
OPTIM = load_json(
    os.path.join(RESULTS_DIR, 'optim_results.json'),
    fallback=[]  # fallback explicite dans fig3
)


# ══════════════════════════════════════════════════════════════════════════════
#  Données iCIFAR (résultats de demo_icifar.py — non sauvegardés en JSON)
# ══════════════════════════════════════════════════════════════════════════════

# Side-Tuning Alpha Merge, 10 tâches iCIFAR-100, 10 epochs
ICIFAR_RESULTS = {
    'alpha_merge': {
        'avg_val_acc':   76.31,
        'avg_train_acc': 70.88,
    }
}
ICIFAR_PER_TASK = {
    'train_acc': [69.28, 70.62, 70.54, 68.47, 72.36, 68.20, 71.48, 67.60, 72.70, 77.57],
    'val_acc':   [75.30, 75.80, 76.30, 74.20, 78.20, 75.40, 75.20, 72.80, 78.20, 81.70],
}

# Acte 1 — 10k images, effet du nb d'epochs (5 et 20 seulement — pas de run à 10ep)
ACT1 = {
    'epochs':         [5,     20],
    'features':       [79.25, 76.70],
    'finetune':       [87.45, 79.15],
    'finetune_train': [88.34, 92.94],   # train acc correspondant
    'sidetune':       [81.20, 81.15],
    'sidetune_train': [81.23, 84.26],
    'features_train': [78.70, 77.10],
}

# Acte 2 — effet de la taille du dataset (10 epochs)
# val_acc / train_acc pour les points disponibles
ACT2 = {
    'train_sizes':        [20000,  50000],
    'adapter_val':        [86.65,  88.40],
    'adapter_train':      [92.62,  93.83],
    'finetune_val':       [86.10,  87.55],
    'finetune_train':     [95.85,  97.06],
    'sidetune_val':       [82.65,  85.60],
    'sidetune_train':     [88.42,  90.70],
    'features_val':       [80.85,  None],
    'features_train':     [81.91,  None],
}

# Heatmap : val_acc — fallback si JSON optim absent
HEATMAP_VAL = {
    'sides':  ['small', 'medium', 'large'],
    'epochs': [5,       10,       20],
    'val': [
        [81.40, 79.30, 80.05],
        [83.20, 83.00, 82.40],
        [79.75, 85.90, 79.00],
    ],
    'train': [
        [82.10, 81.45, 83.20],
        [84.30, 85.70, 86.10],
        [81.20, 87.69, 89.93],
    ]
}

# Couleurs cohérentes par méthode
COLORS = {
    'features':      '#aaaaaa', 'Features':      '#aaaaaa',
    'finetune':      '#e06c75', 'Finetune':      '#e06c75', 'Fine-Tuning': '#e06c75',
    'sidetune':      '#61afef', 'Side-Tuning':   '#61afef',
    'adapter':       '#98c379', 'Adapter':       '#98c379',
    'lora':          '#c678dd', 'LoRA':          '#c678dd',
    'bitfit':        '#e5c07b', 'BitFit':        '#e5c07b',
    'prompt_tuning': '#56b6c2', 'Prompt Tuning': '#56b6c2',
}

def clean(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


# ══════════════════════════════════════════════════════════════════════════════
#  Fig 0 — iCIFAR : val + train accuracy par tâche (10 tâches, 10 epochs)
# ══════════════════════════════════════════════════════════════════════════════

def fig0_icifar():
    print("  [fig0] iCIFAR par tâche...")
    train_accs = ICIFAR_PER_TASK['train_acc']
    val_accs   = ICIFAR_PER_TASK['val_acc']
    avg_val    = ICIFAR_RESULTS['alpha_merge']['avg_val_acc']
    avg_train  = ICIFAR_RESULTS['alpha_merge']['avg_train_acc']
    n          = len(val_accs)
    x          = np.arange(n)
    w          = 0.38

    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.bar(x - w/2, train_accs, w, label='Train Accuracy', color='#61afef', alpha=0.50, edgecolor='white')
    ax.bar(x + w/2, val_accs,   w, label='Val Accuracy',   color='#61afef', alpha=0.95, edgecolor='white')

    ax.axhline(avg_val,   color='#e06c75', linestyle='--', linewidth=1.5,
               label=f'Moy. Val   = {avg_val:.2f}%')
    ax.axhline(avg_train, color='#61afef', linestyle=':',  linewidth=1.2,
               label=f'Moy. Train = {avg_train:.2f}%')

    for i, (tr, va) in enumerate(zip(train_accs, val_accs)):
        ax.text(i - w/2, tr + 0.3, f'{tr:.1f}', ha='center', fontsize=7,   color='#555')
        ax.text(i + w/2, va + 0.3, f'{va:.1f}', ha='center', fontsize=7.5, fontweight='bold')

    ax.set_xticks(x)
    ax.set_xticklabels([f'Tâche {i}' for i in range(n)], rotation=30, ha='right')
    ax.set_ylabel('Accuracy (%)')
    ax.set_ylim(60, 88)
    # Titre : indique toujours train_subset et epochs
    ax.set_title('Side-Tuning (Alpha Merge) — iCIFAR-100\n'
                 '10 tâches | 10 epochs | train subset = 5k/tâche',
                 fontweight='bold')
    ax.legend()
    ax.grid(True, axis='y', alpha=0.3)
    clean(ax)

    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, 'fig0_icifar.pdf')
    plt.savefig(path)
    plt.close()
    print(f"    ✓ {path}")


# ══════════════════════════════════════════════════════════════════════════════
#  Fig 1 — Exemples d'images PCam (chargées depuis les h5)
# ══════════════════════════════════════════════════════════════════════════════

def fig1_pcam_samples():
    print("  [fig1] Exemples PCam...")
    try:
        with h5py.File(os.path.join(PCAM_DIR, 'training_split.h5'), 'r') as fx:
            x_key = list(fx.keys())[0]
            imgs  = fx[x_key][:500]
        with h5py.File(os.path.join(PCAM_DIR, 'camelyonpatch_level_2_split_train_y.h5'), 'r') as fy:
            y_key  = list(fy.keys())[0]
            labels = fy[y_key][:500].flatten()

        neg_idx = np.where(labels == 0)[0][:4]
        pos_idx = np.where(labels == 1)[0][:4]

        fig, axes = plt.subplots(2, 4, figsize=(8, 4.2))
        titles = ['Tissu sain (label=0)'] * 4 + ['Métastase (label=1)'] * 4

        for i, idx in enumerate(list(neg_idx) + list(pos_idx)):
            ax = axes[i // 4, i % 4]
            ax.imshow(imgs[idx])
            ax.set_title(titles[i], fontsize=9,
                         color='#2e7d32' if i < 4 else '#c62828')
            ax.axis('off')

        fig.suptitle("Exemples d'images PatchCamelyon (96×96 px)",
                     fontsize=12, fontweight='bold', y=1.01)
        plt.tight_layout()
        path = os.path.join(FIGURES_DIR, 'fig1_pcam_samples.png')
        plt.savefig(path)
        plt.close()
        print(f"    ✓ {path}")

    except Exception as e:
        print(f"    ✗ Impossible de charger PCam : {e}")
        print(f"      Vérifie que les fichiers h5 sont dans {PCAM_DIR}")


# ══════════════════════════════════════════════════════════════════════════════
#  Fig 2 — Val + Train accuracy vs epochs (gauche) et vs data size (droite)
#
#  Corrections :
#  - Ajout de la train_acc (courbe en tirets, même couleur)
#  - Acte 1 : seulement 5 et 20 epochs (pas de run à 10ep → pas de point affiché)
#  - Titre indique toujours train_subset et epochs quand pas sur un axe
# ══════════════════════════════════════════════════════════════════════════════

def fig2_baseline_datasize():
    print("  [fig2] Baselines val + train vs epochs / data size...")
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # ── Gauche : val + train vs epochs (10k train, 5 et 20 ep) ──────────────
    ax = axes[0]
    configs = [
        ('Features',    'features',       ACT1['features'],       ACT1['features_train']),
        ('Fine-Tuning', 'finetune',        ACT1['finetune'],        ACT1['finetune_train']),
        ('Side-Tuning', 'sidetune',        ACT1['sidetune'],        ACT1['sidetune_train']),
    ]
    for label, key, vals_v, vals_t in configs:
        color = COLORS[key]
        ep    = ACT1['epochs']
        # Val : trait plein + marqueurs
        ax.plot(ep, vals_v, 'o-',  color=color, linewidth=2,   markersize=7,  label=f'{label} (val)')
        # Train : trait pointillé, même couleur, pas de marqueur carré pour ne pas surcharger
        ax.plot(ep, vals_t, 's--', color=color, linewidth=1.4, markersize=5,  alpha=0.65, label=f'{label} (train)')

    # Annotation overfitting finetune
    ax.annotate('Overfitting\ndu Fine-Tuning',
                xy=(20, ACT1['finetune'][-1]), xytext=(15.5, 84),
                arrowprops=dict(arrowstyle='->', color='#e06c75'),
                color='#e06c75', fontsize=9)

    ax.set_xlabel("Nombre d'epochs")
    ax.set_ylabel("Accuracy (%)")
    # Epochs pas sur cet axe (axe x = epochs) donc pas besoin de le répéter dans le titre
    ax.set_title("Acte 1 — Effet des epochs\nTrain subset = 10k", fontweight='bold')
    ax.set_xticks(ACT1['epochs'])
    ax.set_xlim(3, 23)
    ax.set_ylim(68, 98)
    # Légende en deux colonnes pour ne pas surcharger
    ax.legend(ncol=2, fontsize=8.5)
    ax.grid(True, alpha=0.3)
    clean(ax)

    # ── Droite : val + train vs data size (10 epochs) ────────────────────────
    ax = axes[1]
    sz_configs = [
        ('Adapter',     'adapter',   ACT2['adapter_val'],   ACT2['adapter_train']),
        ('Fine-Tuning', 'finetune',  ACT2['finetune_val'],  ACT2['finetune_train']),
        ('Side-Tuning', 'sidetune',  ACT2['sidetune_val'],  ACT2['sidetune_train']),
        ('Features',    'features',  ACT2['features_val'],  ACT2['features_train']),
    ]
    sizes = ACT2['train_sizes']
    for label, key, vals_v, vals_t in sz_configs:
        color  = COLORS[key]
        # Filtrer les None
        sz_v = [s for s, v in zip(sizes, vals_v) if v is not None]
        yv   = [v for v in vals_v  if v is not None]
        sz_t = [s for s, v in zip(sizes, vals_t) if v is not None]
        yt   = [v for v in vals_t  if v is not None]
        ax.plot(sz_v, yv, 'o-',  color=color, linewidth=2,   markersize=7,  label=f'{label} (val)')
        ax.plot(sz_t, yt, 's--', color=color, linewidth=1.4, markersize=5,  alpha=0.65, label=f'{label} (train)')

    # Data size sur l'axe x → indique epochs dans le titre
    ax.set_xlabel("Taille du jeu d'entraînement")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Acte 2 — Effet de la taille du dataset\n10 epochs", fontweight='bold')
    ax.set_xticks(sizes)
    ax.set_xticklabels(['20k', '50k'])
    ax.set_xlim(17000, 55000)
    ax.set_ylim(78, 102)
    ax.legend(ncol=2, fontsize=8.5)
    ax.grid(True, alpha=0.3)
    clean(ax)

    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, 'fig2_baseline_datasize.pdf')
    plt.savefig(path)
    plt.close()
    print(f"    ✓ {path}")


# ══════════════════════════════════════════════════════════════════════════════
#  Fig 3 — Heatmap side_size × epochs
#
#  Corrections :
#  - Chargement depuis JSON optim si disponible
#  - Affiche val_acc + (train_acc) en dessous dans chaque cellule
#  - Titre indique train_subset et méthode fixée
# ══════════════════════════════════════════════════════════════════════════════

def fig3_ablation_heatmap():
    print("  [fig3] Heatmap ablation...")

    # Charger depuis JSON si dispo
    if OPTIM:
        import pandas as pd
        df     = pd.DataFrame(OPTIM)
        sides  = ['small', 'medium', 'large']
        epochs = sorted(df['epochs'].unique())
        val_grid   = df.pivot(index='side', columns='epochs', values='val_acc').reindex(sides).values
        train_grid = df.pivot(index='side', columns='epochs', values='train_acc').reindex(sides).values
    else:
        val_grid   = np.array(HEATMAP_VAL['val'])
        train_grid = np.array(HEATMAP_VAL['train'])
        sides  = HEATMAP_VAL['sides']
        epochs = HEATMAP_VAL['epochs']

    side_labels  = [f'Side {s.capitalize()}' for s in sides]
    epoch_labels = [f'{e} ep.' for e in epochs]

    fig, ax = plt.subplots(figsize=(7, 4))

    cmap = LinearSegmentedColormap.from_list('custom', ['#ffeaa7', '#00b894'], N=256)
    im = ax.imshow(val_grid, cmap=cmap, aspect='auto',
                   vmin=val_grid.min() - 1, vmax=val_grid.max() + 1)

    ax.set_xticks(range(len(epoch_labels)))
    ax.set_yticks(range(len(side_labels)))
    ax.set_xticklabels(epoch_labels)
    ax.set_yticklabels(side_labels)
    ax.set_xlabel("Nombre d'epochs")
    ax.set_ylabel("Taille du side network")
    # Titre : train_subset et config fixée
    ax.set_title("Val Accuracy (Train Accuracy) — Base=RN50, Merge=MLP\n"
                 "Train subset = 20k",
                 fontweight='bold')

    # Valeurs dans les cellules : val_acc\n(train_acc)
    vmax_idx = np.unravel_index(val_grid.argmax(), val_grid.shape)
    for i in range(len(sides)):
        for j in range(len(epochs)):
            is_best = (i, j) == vmax_idx
            color   = 'white' if is_best else 'black'
            weight  = 'bold'  if is_best else 'normal'
            # Ligne 1 : val_acc en grand
            ax.text(j, i - 0.12, f'{val_grid[i, j]:.1f}%',
                    ha='center', va='center',
                    color=color, fontweight=weight, fontsize=11)
            # Ligne 2 : (train_acc) en petit en dessous
            ax.text(j, i + 0.25, f'({train_grid[i, j]:.1f}%)',
                    ha='center', va='center',
                    color=color, fontsize=8, alpha=0.85)

    plt.colorbar(im, ax=ax, label='Val Accuracy (%)', shrink=0.85)
    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, 'fig3_ablation_heatmap.pdf')
    plt.savefig(path)
    plt.close()
    print(f"    ✓ {path}")


# ══════════════════════════════════════════════════════════════════════════════
#  Fig 4 — Bar chart benchmark 7 méthodes
#
#  Corrections :
#  - Chargement depuis JSON benchmark
#  - Val_acc : étiquette en gras au-dessus de la barre
#  - Train_acc : étiquette normale (pas en gras) au-dessus de la barre train
#  - Titre indique train_subset et epochs
# ══════════════════════════════════════════════════════════════════════════════

def fig4_benchmark_bar():
    print("  [fig4] Bar chart benchmark...")
    data_sorted = sorted(BENCHMARK, key=lambda x: x['val_acc'], reverse=True)

    methods    = [d['method']    for d in data_sorted]
    val_accs   = [d['val_acc']   for d in data_sorted]
    train_accs = [d['train_acc'] for d in data_sorted]
    colors     = [COLORS.get(m, '#888') for m in methods]

    x = np.arange(len(methods))
    w = 0.38

    fig, ax = plt.subplots(figsize=(11, 5))
    bars_val   = ax.bar(x - w/2, val_accs,   w, label='Val Accuracy',
                        color=colors, alpha=0.95, edgecolor='white', linewidth=0.8)
    bars_train = ax.bar(x + w/2, train_accs, w, label='Train Accuracy',
                        color=colors, alpha=0.40, edgecolor='white', linewidth=0.8, hatch='//')

    # Val : étiquettes en gras au-dessus
    for bar, val in zip(bars_val, val_accs):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.25,
                f'{val:.1f}%', ha='center', va='bottom', fontsize=8.5, fontweight='bold')

    # Train : étiquettes normales (pas en gras) au-dessus
    for bar, val in zip(bars_train, train_accs):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.25,
                f'{val:.1f}%', ha='center', va='bottom', fontsize=8, fontweight='normal',
                color='#444444')

    # Ligne baseline features
    features_acc = next(d['val_acc'] for d in BENCHMARK if d['method'] == 'Features')
    ax.axhline(y=features_acc, color='gray', linestyle='--', alpha=0.6, linewidth=1.2,
               label=f'Baseline features ({features_acc:.1f}%)')

    ax.set_ylabel('Accuracy (%)')
    # Titre : indique train_subset et epochs
    ax.set_title('Benchmark PEFT — 7 méthodes de transfer learning\n'
                 'Base=ResNet-50 | Train subset = 20k | 10 epochs',
                 fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=15, ha='right')
    ax.set_ylim(72, 106)
    ax.legend(loc='lower right')
    ax.grid(True, axis='y', alpha=0.3)
    clean(ax)

    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, 'fig4_benchmark_bar.pdf')
    plt.savefig(path)
    plt.close()
    print(f"    ✓ {path}")


# ══════════════════════════════════════════════════════════════════════════════
#  Fig 5 — Scatter params vs val_acc
#
#  Corrections de mise en page :
#  - Labels placés hors du nuage, tous dans la marge droite ou gauche
#  - Flèches courtes et propres depuis chaque point
#  - ylim et xlim plus larges pour éviter les chevauchements
#  - Annotation "sweet spot" repositionnée
#  - Titre indique train_subset et epochs
# ══════════════════════════════════════════════════════════════════════════════

def fig5_scatter_params():
    print("  [fig5] Scatter params vs accuracy...")

    fig, ax = plt.subplots(figsize=(10, 6))

    # Position des labels : (x_text_en_M, y_text, ha)
    # Tous placés soigneusement pour éviter chevauchements sur échelle log
    label_pos = {
        'Adapter':       (0.8,   87.5,  'right'),
        'Finetune':      (30.0,  86.8,  'left'),
        'Side-Tuning':   (5.0,   83.5,  'left'),
        'BitFit':        (0.006, 83.4,  'left'),
        'Prompt Tuning': (0.3,   82.4,  'left'),
        'LoRA':          (0.001, 81.0,  'left'),
        'Features':      (0.001, 80.0,  'right'),
    }

    for d in BENCHMARK:
        xp    = d['n_params'] / 1e6
        yp    = d['val_acc']
        color = COLORS.get(d['method'], '#888')

        # Point
        ax.scatter(xp, yp, s=140, color=color, zorder=5,
                   edgecolors='white', linewidth=1)

        # Label avec flèche
        xt, yt, ha = label_pos.get(d['method'], (xp * 2, yp + 0.3, 'left'))
        ax.annotate(
            d['method'],
            xy=(xp, yp),
            xytext=(xt, yt),
            ha=ha, va='center',
            fontsize=9.5, color=color, fontweight='bold',
            arrowprops=dict(arrowstyle='->', color=color, alpha=0.6, lw=1.0,
                            connectionstyle='arc3,rad=0.1'),
        )

    # Zone parcimonieuse
    ax.axvspan(1e-4, 1.0, alpha=0.04, color='green')
    ax.text(0.5, 78.5, '< 1M params', ha='center', fontsize=8,
            color='green', alpha=0.7, style='italic')

    ax.set_xscale('log')
    ax.set_xlabel('Paramètres entraînables (log scale, en millions)')
    ax.set_ylabel('Validation Accuracy (%)')
    # Titre : indique train_subset et epochs
    ax.set_title('Efficacité paramétrique — Accuracy vs Params entraînables\n'
                 'Train subset = 20k | 10 epochs',
                 fontweight='bold')
    ax.set_ylim(77.5, 89.5)
    ax.set_xlim(5e-4, 60)
    ax.grid(True, alpha=0.3)
    clean(ax)

    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, 'fig5_scatter_params.pdf')
    plt.savefig(path)
    plt.close()
    print(f"    ✓ {path}")


# ══════════════════════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    print(f"\nGénération des figures → {FIGURES_DIR}\n")

    fig0_icifar()
    fig1_pcam_samples()
    fig2_baseline_datasize()
    fig3_ablation_heatmap()
    fig4_benchmark_bar()
    fig5_scatter_params()

    print(f"\n✅ Toutes les figures générées dans {FIGURES_DIR}/")
    print("   Lance ensuite : pdflatex report_pcam.tex")
