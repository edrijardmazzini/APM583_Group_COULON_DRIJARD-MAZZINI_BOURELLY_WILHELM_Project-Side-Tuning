"""
generate_figures.py  —  Génère toutes les figures du rapport LaTeX
===================================================================

À lancer depuis la racine du repo :
    python scripts/generate_figures.py

Produit dans figures/ :
    fig1_pcam_samples.png       — exemples d'images PCam
    fig2_baseline_datasize.pdf  — accuracy vs data size (acte 1+2)
    fig3_ablation_heatmap.pdf   — heatmap side_size × epochs
    fig4_benchmark_bar.pdf      — bar chart 7 méthodes
    fig5_scatter_params.pdf     — scatter params vs val_acc

⚠ Remplis la variable ICIFAR_RESULTS avec tes résultats iCIFAR avant de lancer.
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
    'font.family':      'serif',
    'font.size':        11,
    'axes.titlesize':   12,
    'axes.labelsize':   11,
    'xtick.labelsize':  10,
    'ytick.labelsize':  10,
    'legend.fontsize':  10,
    'figure.dpi':       150,
    'savefig.dpi':      300,
    'savefig.bbox':     'tight',
    'savefig.pad_inches': 0.05,
})

REPO_DIR    = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
FIGURES_DIR = os.path.join(REPO_DIR, 'figures')
PCAM_DIR    = os.path.join(REPO_DIR, 'data', 'pcam')
os.makedirs(FIGURES_DIR, exist_ok=True)

# ══════════════════════════════════════════════════════════════════════════════
#  Resultats iCIFAR (demo_icifar.py - Side-Tuning Alpha Merge, 10 epochs)
# ══════════════════════════════════════════════════════════════════════════════

ICIFAR_RESULTS = {
    'alpha_merge': {
        'avg_val_acc':   76.31,
        'avg_train_acc': 70.88,
    }
}

# Val/Train accuracy par tache (10 taches iCIFAR-100)
ICIFAR_PER_TASK = {
    'train_acc': [69.28, 70.62, 70.54, 68.47, 72.36, 68.20, 71.48, 67.60, 72.70, 77.57],
    'val_acc':   [75.30, 75.80, 76.30, 74.20, 78.20, 75.40, 75.20, 72.80, 78.20, 81.70],
    'tasks':     ['T0','T1','T2','T3','T4','T5','T6','T7','T8','T9'],
}

# ══════════════════════════════════════════════════════════════════════════════
#  Données de tes expériences PCam
# ══════════════════════════════════════════════════════════════════════════════

# Acte 1 — 10k images, effet du nb d'epochs
ACT1 = {
    'epochs':   [5,     10,    20],
    'features': [79.25, None,  76.70],
    'finetune': [87.45, None,  79.15],
    'sidetune': [81.20, None,  81.15],
}

# Acte 2 — effet de la taille du dataset (epochs=10)
ACT2 = {
    'train_sizes': [10000,  20000,  50000],
    'adapter':     [None,   86.65,  88.40],
    'finetune':    [None,   86.10,  87.55],
    'sidetune':    [None,   82.65,  85.60],  # 10k = run 20ep best
    'features':    [None,   80.85,  None],
}

# Benchmark 7 méthodes — 20k, 10 epochs
BENCHMARK = [
    {'method': 'Adapter',       'val_acc': 86.65, 'train_acc': 92.62, 'val_loss': 0.0064, 'n_params': 507394},
    {'method': 'Finetune',      'val_acc': 86.10, 'train_acc': 95.85, 'val_loss': 0.0119, 'n_params': 23512130},
    {'method': 'Side-Tuning',   'val_acc': 82.65, 'train_acc': 88.42, 'val_loss': 0.0071, 'n_params': 2931458},
    {'method': 'BitFit',        'val_acc': 82.65, 'train_acc': 85.83, 'val_loss': 0.0067, 'n_params': 30658},
    {'method': 'Prompt Tuning', 'val_acc': 81.65, 'train_acc': 83.37, 'val_loss': 0.0069, 'n_params': 672770},
    {'method': 'LoRA',          'val_acc': 81.60, 'train_acc': 81.89, 'val_loss': 0.0071, 'n_params': 4098},
    {'method': 'Features',      'val_acc': 80.85, 'train_acc': 81.91, 'val_loss': 0.0072, 'n_params': 4098},
]

# Benchmark final — 50k, 10 epochs
BENCHMARK_FINAL = [
    {'method': 'Adapter',     'val_acc': 88.40, 'train_acc': 93.83, 'val_loss': 0.0065, 'n_params': 507394},
    {'method': 'Finetune',    'val_acc': 87.55, 'train_acc': 97.06, 'val_loss': 0.0113, 'n_params': 23512130},
    {'method': 'Side-Tuning', 'val_acc': 85.60, 'train_acc': 90.70, 'val_loss': 0.0070, 'n_params': 2931458},
]

# Heatmap side_size × epochs (RN50 + MLP)
HEATMAP = {
    'sides':  ['small', 'medium', 'large'],
    'epochs': [5,       10,       20],
    'data': [
        [81.40, 79.30, 80.05],   # small
        [83.20, 83.00, 82.40],   # medium
        [79.75, 85.90, 79.00],   # large
    ]
}

# Couleurs cohérentes par méthode
COLORS = {
    'features':      '#aaaaaa',
    'Features':      '#aaaaaa',
    'finetune':      '#e06c75',
    'Finetune':      '#e06c75',
    'sidetune':      '#61afef',
    'Side-Tuning':   '#61afef',
    'adapter':       '#98c379',
    'Adapter':       '#98c379',
    'lora':          '#c678dd',
    'LoRA':          '#c678dd',
    'bitfit':        '#e5c07b',
    'BitFit':        '#e5c07b',
    'prompt_tuning': '#56b6c2',
    'Prompt Tuning': '#56b6c2',
}


# ══════════════════════════════════════════════════════════════════════════════
#  Fig 1 — Exemples d'images PCam
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
#  Fig 2 — Baselines : overfitting à 10k (acte 1)
# ══════════════════════════════════════════════════════════════════════════════

def fig2_baseline_datasize():
    print("  [fig2] Baselines vs data size / epochs...")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    # ── Gauche : accuracy vs epochs (10k) ────────────────────────────────────
    ax = axes[0]
    for method, key in [('Features', 'features'), ('Fine-Tuning', 'finetune'), ('Side-Tuning', 'sidetune')]:
        vals = ACT1[key]
        ep   = ACT1['epochs']
        # Filtrer les None
        x = [e for e, v in zip(ep, vals) if v is not None]
        y = [v for v in vals if v is not None]
        color = COLORS.get(key, '#888888')
        ax.plot(x, y, 'o-', color=color, linewidth=2, markersize=7, label=method)

    ax.axhline(y=50, color='gray', linestyle=':', alpha=0.4, linewidth=1)
    ax.set_xlabel("Nombre d'epochs")
    ax.set_ylabel("Validation Accuracy (%)")
    ax.set_title("Acte 1 — Effet des epochs\n(10k images d'entraînement)", fontweight='bold')
    ax.set_xticks(ACT1['epochs'])
    ax.set_ylim(70, 93)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Annotation overfitting
    ax.annotate('Overfitting\ndu Fine-Tuning',
                xy=(20, 79.15), xytext=(15, 84),
                arrowprops=dict(arrowstyle='->', color='#e06c75'),
                color='#e06c75', fontsize=9)

    # ── Droite : accuracy vs train size (10 epochs) ───────────────────────────
    ax = axes[1]
    plot_data = {
        'Adapter':     (BENCHMARK_FINAL, [86.65, 88.40]),   # 20k, 50k
        'Fine-Tuning': ([86.10, 87.55], None),
        'Side-Tuning': ([82.65, 85.60], None),
        'Features':    ([80.85], None),
    }

    sizes_map = {
        'Adapter':     [20000, 50000],
        'Fine-Tuning': [20000, 50000],
        'Side-Tuning': [20000, 50000],
        'Features':    [20000],
    }
    accs_map = {
        'Adapter':     [86.65, 88.40],
        'Fine-Tuning': [86.10, 87.55],
        'Side-Tuning': [82.65, 85.60],
        'Features':    [80.85],
    }
    keys_map = {
        'Adapter': 'adapter', 'Fine-Tuning': 'finetune',
        'Side-Tuning': 'sidetune', 'Features': 'features'
    }

    for label, sizes in sizes_map.items():
        accs  = accs_map[label]
        color = COLORS.get(keys_map[label], '#888')
        ax.plot(sizes, accs, 'o-', color=color, linewidth=2, markersize=7, label=label)

    ax.set_xlabel("Taille du jeu d'entraînement")
    ax.set_ylabel("Validation Accuracy (%)")
    ax.set_title("Acte 2 — Effet de la taille du dataset\n(10 epochs)", fontweight='bold')
    ax.set_xticks([20000, 50000])
    ax.set_xticklabels(['20k', '50k'])
    ax.set_ylim(78, 92)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, 'fig2_baseline_datasize.pdf')
    plt.savefig(path)
    plt.close()
    print(f"    ✓ {path}")


# ══════════════════════════════════════════════════════════════════════════════
#  Fig 3 — Heatmap side_size × epochs
# ══════════════════════════════════════════════════════════════════════════════

def fig3_ablation_heatmap():
    print("  [fig3] Heatmap ablation...")
    data   = np.array(HEATMAP['data'])
    sides  = [f'Side {s.capitalize()}' for s in HEATMAP['sides']]
    epochs = [f'{e} ep.' for e in HEATMAP['epochs']]

    fig, ax = plt.subplots(figsize=(6, 3.5))

    cmap = LinearSegmentedColormap.from_list(
        'custom', ['#ffeaa7', '#00b894'], N=256)
    im = ax.imshow(data, cmap=cmap, aspect='auto',
                   vmin=data.min() - 1, vmax=data.max() + 1)

    ax.set_xticks(range(len(epochs)))
    ax.set_yticks(range(len(sides)))
    ax.set_xticklabels(epochs)
    ax.set_yticklabels(sides)
    ax.set_xlabel("Nombre d'epochs")
    ax.set_ylabel("Taille du side network")
    ax.set_title("Validation Accuracy — Base=RN50, Merge=MLP\n",
                 fontweight='bold')

    # Valeurs dans les cellules
    vmax_idx = np.unravel_index(data.argmax(), data.shape)
    for i in range(len(sides)):
        for j in range(len(epochs)):
            color = 'white' if (i, j) == vmax_idx else 'black'
            weight = 'bold' if (i, j) == vmax_idx else 'normal'
            ax.text(j, i, f'{data[i, j]:.1f}%',
                    ha='center', va='center',
                    color=color, fontweight=weight, fontsize=11)

    plt.colorbar(im, ax=ax, label='Val Accuracy (%)', shrink=0.8)
    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, 'fig3_ablation_heatmap.pdf')
    plt.savefig(path)
    plt.close()
    print(f"    ✓ {path}")


# ══════════════════════════════════════════════════════════════════════════════
#  Fig 4 — Bar chart benchmark 7 méthodes
# ══════════════════════════════════════════════════════════════════════════════

def fig4_benchmark_bar():
    print("  [fig4] Bar chart benchmark...")
    data_sorted = sorted(BENCHMARK, key=lambda x: x['val_acc'], reverse=True)

    methods   = [d['method'] for d in data_sorted]
    val_accs  = [d['val_acc']  for d in data_sorted]
    train_accs= [d['train_acc'] for d in data_sorted]
    colors    = [COLORS.get(m, '#888') for m in methods]

    x = np.arange(len(methods))
    w = 0.38

    fig, ax = plt.subplots(figsize=(10, 4.5))
    bars_val   = ax.bar(x - w/2, val_accs,   w, label='Val Accuracy',   color=colors, alpha=0.95, edgecolor='white', linewidth=0.8)
    bars_train = ax.bar(x + w/2, train_accs, w, label='Train Accuracy', color=colors, alpha=0.45, edgecolor='white', linewidth=0.8, hatch='//')

    # Valeurs sur les barres
    for bar, val in zip(bars_val, val_accs):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                f'{val:.1f}%', ha='center', va='bottom', fontsize=8.5, fontweight='bold')

    # Ligne baseline "features"
    features_acc = next(d['val_acc'] for d in BENCHMARK if d['method'] == 'Features')
    ax.axhline(y=features_acc, color='gray', linestyle='--', alpha=0.6, linewidth=1.2, label=f'Baseline features ({features_acc:.1f}%)')

    ax.set_ylabel('Accuracy (%)')
    ax.set_title('Benchmark PEFT — 7 méthodes de transfer learning\n(Base=ResNet-50, 20k train, 10 epochs)',
                 fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=15, ha='right')
    ax.set_ylim(72, 102)
    ax.legend(loc='lower right')
    ax.grid(True, axis='y', alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, 'fig4_benchmark_bar.pdf')
    plt.savefig(path)
    plt.close()
    print(f"    ✓ {path}")


# ══════════════════════════════════════════════════════════════════════════════
#  Fig 5 — Scatter params vs val_acc
# ══════════════════════════════════════════════════════════════════════════════

def fig5_scatter_params():
    print("  [fig5] Scatter params vs accuracy...")
    fig, ax = plt.subplots(figsize=(8, 5))

    for d in BENCHMARK:
        x     = d['n_params'] / 1e6
        y     = d['val_acc']
        color = COLORS.get(d['method'], '#888')
        ax.scatter(x, y, s=120, color=color, zorder=5,
                   edgecolors='white', linewidth=0.8)

        # Labels avec offset pour éviter les chevauchements
        offsets = {
            'Adapter':       (-0.6,  0.4),
            'Finetune':      ( 0.5,  0.2),
            'Side-Tuning':   ( 0.1,  0.4),
            'BitFit':        ( 0.05,-0.6),
            'Prompt Tuning': (-0.3,  0.4),
            'LoRA':          ( 0.05, 0.4),
            'Features':      (-0.6, -0.6),
        }
        dx, dy = offsets.get(d['method'], (0.1, 0.3))
        ax.annotate(d['method'],
                    xy=(x, y), xytext=(x + dx, y + dy),
                    fontsize=9, color=color, fontweight='bold',
                    arrowprops=dict(arrowstyle='-', color=color, alpha=0.5, lw=0.8))

    # Zone "sweet spot"
    ax.axvspan(0, 1, alpha=0.05, color='green', label='Zone parcimonieuse (<1M params)')

    ax.set_xscale('log')
    ax.set_xlabel('Nombre de paramètres entraînables (log scale, en millions)')
    ax.set_ylabel('Validation Accuracy (%)')
    ax.set_title('Efficacité paramétrique — Accuracy vs Params entraînables\n(20k train, 10 epochs)',
                 fontweight='bold')
    ax.set_ylim(78, 90)
    ax.grid(True, alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Annotation Adapter
    adapter = next(d for d in BENCHMARK if d['method'] == 'Adapter')
    ax.annotate('Sweet spot\n↓',
                xy=(adapter['n_params']/1e6, adapter['val_acc']),
                xytext=(adapter['n_params']/1e6 * 2, adapter['val_acc'] - 2.5),
                fontsize=9, color='#98c379',
                arrowprops=dict(arrowstyle='->', color='#98c379'))

    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, 'fig5_scatter_params.pdf')
    plt.savefig(path)
    plt.close()
    print(f"    ✓ {path}")




# ══════════════════════════════════════════════════════════════════════════════
#  Fig 0 — iCIFAR résultats par tâche
# ══════════════════════════════════════════════════════════════════════════════

def fig0_icifar():
    print("  [fig0] iCIFAR par tâche...")
    tasks      = ICIFAR_PER_TASK['tasks']
    train_accs = ICIFAR_PER_TASK['train_acc']
    val_accs   = ICIFAR_PER_TASK['val_acc']
    avg_val    = ICIFAR_RESULTS['alpha_merge']['avg_val_acc']

    x = np.arange(len(tasks))
    w = 0.38

    fig, ax = plt.subplots(figsize=(9, 4.2))
    ax.bar(x - w/2, train_accs, w, label='Train Accuracy', color='#61afef', alpha=0.6, edgecolor='white')
    ax.bar(x + w/2, val_accs,   w, label='Val Accuracy',   color='#61afef', alpha=0.95, edgecolor='white')

    ax.axhline(y=avg_val, color='#e06c75', linestyle='--', linewidth=1.5,
               label=f'Moy. Val = {avg_val:.2f}%')

    for i, (tr, va) in enumerate(zip(train_accs, val_accs)):
        ax.text(i - w/2, tr + 0.3, f'{tr:.1f}', ha='center', fontsize=7, color='#555')
        ax.text(i + w/2, va + 0.3, f'{va:.1f}', ha='center', fontsize=7.5, fontweight='bold')

    ax.set_xticks(x)
    ax.set_xticklabels([f'Tâche {i}' for i in range(len(tasks))], rotation=30, ha='right')
    ax.set_ylabel('Accuracy (%)')
    ax.set_ylim(60, 88)
    ax.set_title('Side-Tuning (Alpha Merge) sur iCIFAR-100\n10 tâches, 10 epochs',
                 fontweight='bold')
    ax.legend()
    ax.grid(True, axis='y', alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, 'fig0_icifar.pdf')
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
