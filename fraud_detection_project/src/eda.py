# =============================================================================
#  src/eda.py  -  Analyse Exploratoire des Donnees (EDA)
#  Auteur : Meriem | ECE Paris - Data & AI B3
# =============================================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import os

# ---- Couleurs theme noir et or ----
GOLD  = '#d4af37'
RED   = '#ff6b6b'
GREEN = '#4ade80'
BLUE  = '#60a5fa'

plt.rcParams.update({
    'figure.facecolor': '#0a0a0a',
    'axes.facecolor':   '#111111',
    'axes.edgecolor':   GOLD,
    'axes.labelcolor':  '#e8d5a0',
    'xtick.color':      '#e8d5a0',
    'ytick.color':      '#e8d5a0',
    'text.color':       '#e8d5a0',
    'grid.color':       '#222222',
    'legend.facecolor': '#111111',
    'legend.edgecolor': GOLD,
})

FIGURES_DIR = 'outputs/figures'
os.makedirs(FIGURES_DIR, exist_ok=True)


# =============================================================================
# 1. INFORMATIONS GENERALES
# =============================================================================

def print_dataset_info(df: pd.DataFrame) -> None:
    """Affiche les informations generales du dataset."""
    print("=" * 60)
    print("INFORMATIONS GENERALES DU DATASET")
    print("=" * 60)
    print(f"\nNombre de transactions : {df.shape[0]:,}")
    print(f"Nombre de features     : {df.shape[1]}")
    print(f"Valeurs manquantes     : {df.isnull().sum().sum()}")
    print(f"Doublons               : {df.duplicated().sum()}")
    print(f"\nColonnes : {list(df.columns)}")
    print(f"\nTypes de donnees :")
    print(df.dtypes.to_string())
    print(f"\nStatistiques descriptives :")
    print(df[['Time', 'Amount', 'Class']].describe().round(2).to_string())
    print(f"\nDistribution de la variable cible :")
    vc = df['Class'].value_counts()
    print(f"  Classe 0 (Normal) : {vc[0]:,} ({vc[0]/len(df)*100:.2f}%)")
    print(f"  Classe 1 (Fraude) : {vc[1]:,} ({vc[1]/len(df)*100:.2f}%)")
    print(f"  Ratio desequilibre : 1 fraude pour {vc[0]//vc[1]} transactions normales")


# =============================================================================
# 2. DISTRIBUTION DES CLASSES
# =============================================================================

def plot_class_distribution(df: pd.DataFrame) -> None:
    """Visualise la distribution des classes fraude vs normal."""
    class_counts = df['Class'].value_counts()

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Distribution des Classes - Fraude vs Normal', color=GOLD, fontsize=15, fontweight='bold')

    # Barplot
    bars = axes[0].bar(
        ['Normal (0)', 'Fraude (1)'],
        class_counts.values,
        color=[GOLD, RED],
        alpha=0.85,
        edgecolor='black',
        linewidth=1.2
    )
    axes[0].set_title('Nombre de transactions par classe', color=GOLD, fontsize=12)
    axes[0].set_ylabel('Nombre de transactions')
    axes[0].set_xlabel('Classe')
    for bar, val in zip(bars, class_counts.values):
        axes[0].text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 1000,
            f'{val:,}',
            ha='center', va='bottom', color=GOLD, fontweight='bold', fontsize=11
        )

    # Pie chart
    wedges, texts, autotexts = axes[1].pie(
        class_counts.values,
        labels=[f'Normal\n{class_counts[0]:,} transactions', f'Fraude\n{class_counts[1]:,} transactions'],
        colors=[GOLD, RED],
        autopct='%1.2f%%',
        startangle=90,
        textprops={'color': 'white', 'fontsize': 10},
        wedgeprops={'edgecolor': 'black', 'linewidth': 1.5},
        explode=(0, 0.08)
    )
    for autotext in autotexts:
        autotext.set_color('black')
        autotext.set_fontweight('bold')
    axes[1].set_title('Proportion des classes', color=GOLD, fontsize=12)

    plt.tight_layout()
    path = f'{FIGURES_DIR}/01_class_distribution.png'
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"Figure sauvegardee : {path}")


# =============================================================================
# 3. ANALYSE DES MONTANTS
# =============================================================================

def plot_amount_analysis(df: pd.DataFrame) -> None:
    """Analyse complete de la distribution des montants par classe."""
    fraud_amounts  = df[df['Class'] == 1]['Amount']
    normal_amounts = df[df['Class'] == 0]['Amount']

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle('Analyse des Montants des Transactions', color=GOLD, fontsize=15, fontweight='bold')

    # Histogramme normal
    axes[0, 0].hist(normal_amounts, bins=80, color=GOLD, alpha=0.75, edgecolor='none', density=True)
    axes[0, 0].set_title('Distribution montants - Transactions Normales', color=GOLD)
    axes[0, 0].set_xlabel('Montant (EUR)')
    axes[0, 0].set_ylabel('Densite')
    axes[0, 0].axvline(normal_amounts.mean(), color=RED, linestyle='--', linewidth=2, label=f'Moyenne: {normal_amounts.mean():.2f} EUR')
    axes[0, 0].axvline(normal_amounts.median(), color=GREEN, linestyle='--', linewidth=2, label=f'Mediane: {normal_amounts.median():.2f} EUR')
    axes[0, 0].legend()

    # Histogramme fraude
    axes[0, 1].hist(fraud_amounts, bins=50, color=RED, alpha=0.75, edgecolor='none', density=True)
    axes[0, 1].set_title('Distribution montants - Fraudes', color=GOLD)
    axes[0, 1].set_xlabel('Montant (EUR)')
    axes[0, 1].set_ylabel('Densite')
    axes[0, 1].axvline(fraud_amounts.mean(), color=GOLD, linestyle='--', linewidth=2, label=f'Moyenne: {fraud_amounts.mean():.2f} EUR')
    axes[0, 1].axvline(fraud_amounts.median(), color=GREEN, linestyle='--', linewidth=2, label=f'Mediane: {fraud_amounts.median():.2f} EUR')
    axes[0, 1].legend()

    # Boxplot comparatif
    bp = axes[1, 0].boxplot(
        [normal_amounts, fraud_amounts],
        labels=['Normal', 'Fraude'],
        patch_artist=True,
        boxprops=dict(facecolor='#1a1600', color=GOLD),
        medianprops=dict(color=GREEN, linewidth=2),
        whiskerprops=dict(color=GOLD, linewidth=1.5),
        capprops=dict(color=GOLD, linewidth=2),
        flierprops=dict(marker='o', color=RED, alpha=0.3, markersize=3)
    )
    bp['boxes'][0].set_facecolor('#1a1400')
    bp['boxes'][1].set_facecolor('#2a0a0a')
    axes[1, 0].set_title('Boxplot comparatif des montants', color=GOLD)
    axes[1, 0].set_ylabel('Montant (EUR)')

    # Violin plot
    data_violin = [normal_amounts.sample(min(5000, len(normal_amounts))), fraud_amounts]
    parts = axes[1, 1].violinplot(data_violin, positions=[1, 2], showmedians=True, showextrema=True)
    for i, pc in enumerate(parts['bodies']):
        pc.set_facecolor(GOLD if i == 0 else RED)
        pc.set_alpha(0.6)
    parts['cmedians'].set_color(GREEN)
    parts['cbars'].set_color(GOLD)
    parts['cmaxes'].set_color(GOLD)
    parts['cmins'].set_color(GOLD)
    axes[1, 1].set_xticks([1, 2])
    axes[1, 1].set_xticklabels(['Normal (echantillon)', 'Fraude'])
    axes[1, 1].set_title('Violin plot des montants', color=GOLD)
    axes[1, 1].set_ylabel('Montant (EUR)')

    plt.tight_layout()
    path = f'{FIGURES_DIR}/02_amount_analysis.png'
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"Figure sauvegardee : {path}")

    # Stats
    print(f"\nStatistiques des montants :")
    print(f"  Normal - Moyenne : {normal_amounts.mean():.2f} EUR  |  Max : {normal_amounts.max():.2f} EUR")
    print(f"  Fraude - Moyenne : {fraud_amounts.mean():.2f} EUR  |  Max : {fraud_amounts.max():.2f} EUR")


# =============================================================================
# 4. DISTRIBUTION TEMPORELLE
# =============================================================================

def plot_time_distribution(df: pd.DataFrame) -> None:
    """Analyse la distribution temporelle des transactions et fraudes."""
    df_temp = df.copy()
    df_temp['Hour'] = (df_temp['Time'] / 3600) % 24  # convertir en heures

    fig, axes = plt.subplots(3, 1, figsize=(14, 12))
    fig.suptitle('Analyse Temporelle des Transactions', color=GOLD, fontsize=15, fontweight='bold')

    # Toutes les transactions par heure
    axes[0].hist(df_temp[df_temp['Class']==0]['Hour'], bins=48, color=GOLD, alpha=0.65, label='Normal', density=True)
    axes[0].hist(df_temp[df_temp['Class']==1]['Hour'], bins=48, color=RED, alpha=0.85, label='Fraude', density=True)
    axes[0].set_title('Distribution par heure - Normal vs Fraude (densite)', color=GOLD)
    axes[0].set_xlabel('Heure de la journee')
    axes[0].set_ylabel('Densite')
    axes[0].legend()
    axes[0].set_xticks(range(0, 25, 2))

    # Evolution des fraudes dans le temps
    df_temp['TimeH'] = df_temp['Time'] / 3600
    time_bins = pd.cut(df_temp['TimeH'], bins=48)
    fraud_rate = df_temp.groupby(time_bins, observed=False)['Class'].mean() * 100
    fraud_rate.index = [i.mid for i in fraud_rate.index]
    axes[1].plot(fraud_rate.index, fraud_rate.values, color=RED, linewidth=2)
    axes[1].fill_between(fraud_rate.index, fraud_rate.values, alpha=0.3, color=RED)
    axes[1].set_title('Taux de fraude (%) dans le temps', color=GOLD)
    axes[1].set_xlabel('Heure (depuis debut du dataset)')
    axes[1].set_ylabel('Taux de fraude (%)')
    axes[1].axhline(df['Class'].mean()*100, color=GOLD, linestyle='--', linewidth=1.5, label=f'Taux moyen: {df["Class"].mean()*100:.3f}%')
    axes[1].legend()

    # Nombre de fraudes par heure
    fraud_by_hour = df_temp[df_temp['Class']==1].groupby(pd.cut(df_temp[df_temp['Class']==1]['Hour'], bins=24), observed=False).size()
    fraud_by_hour.index = range(24)
    axes[2].bar(fraud_by_hour.index, fraud_by_hour.values, color=RED, alpha=0.8, edgecolor='black', linewidth=0.5)
    axes[2].set_title('Nombre de fraudes par heure de la journee', color=GOLD)
    axes[2].set_xlabel('Heure')
    axes[2].set_ylabel('Nombre de fraudes')
    axes[2].set_xticks(range(24))

    plt.tight_layout()
    path = f'{FIGURES_DIR}/03_time_distribution.png'
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"Figure sauvegardee : {path}")


# =============================================================================
# 5. CORRELATION DES FEATURES
# =============================================================================

def plot_correlation(df: pd.DataFrame):
    """Affiche la correlation de chaque feature avec la variable cible Class."""
    corr_with_class = df.corr()['Class'].drop('Class').sort_values()

    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    fig.suptitle('Correlation des Features avec la Classe Cible', color=GOLD, fontsize=15, fontweight='bold')

    # Toutes les features
    colors = [RED if v < 0 else GREEN for v in corr_with_class.values]
    bars = axes[0].barh(corr_with_class.index, corr_with_class.values, color=colors, alpha=0.85, edgecolor='black', linewidth=0.5)
    axes[0].set_title('Correlation de toutes les features avec Class', color=GOLD)
    axes[0].set_xlabel('Coefficient de correlation de Pearson')
    axes[0].axvline(x=0, color=GOLD, linewidth=1.2, linestyle='--')
    axes[0].set_xlim(-0.35, 0.35)

    # Top 10 positives et negatives
    top5_pos = corr_with_class.tail(5)
    top5_neg = corr_with_class.head(5)
    top10 = pd.concat([top5_neg, top5_pos])
    colors10 = [RED if v < 0 else GREEN for v in top10.values]
    axes[1].barh(top10.index, top10.values, color=colors10, alpha=0.85, edgecolor='black')
    for i, (idx, val) in enumerate(top10.items()):
        axes[1].text(val + (0.005 if val >= 0 else -0.005), i,
                     f'{val:.3f}', va='center', ha='left' if val >= 0 else 'right',
                     color='white', fontsize=9, fontweight='bold')
    axes[1].axvline(x=0, color=GOLD, linewidth=1.2, linestyle='--')
    axes[1].set_title('Top 10 features les plus correlees', color=GOLD)
    axes[1].set_xlabel('Coefficient de correlation de Pearson')

    plt.tight_layout()
    path = f'{FIGURES_DIR}/04_correlation.png'
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"Figure sauvegardee : {path}")

    top_pos = corr_with_class.tail(5).index.tolist()
    top_neg = corr_with_class.head(5).index.tolist()
    print(f"\nFeatures positivement correlees avec la fraude : {top_pos}")
    print(f"Features negativement correlees avec la fraude : {top_neg}")
    return top_pos, top_neg


# =============================================================================
# 6. HEATMAP
# =============================================================================

def plot_heatmap(df: pd.DataFrame, top_pos: list, top_neg: list) -> None:
    """Heatmap de correlation des features les plus importantes."""
    important_features = top_neg + top_pos + ['Amount', 'Time', 'Class']
    corr_matrix = df[important_features].corr()
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))

    fig, ax = plt.subplots(figsize=(14, 11))
    fig.suptitle('Heatmap de Correlation - Top Features', color=GOLD, fontsize=15, fontweight='bold')

    sns.heatmap(
        corr_matrix,
        annot=True,
        fmt='.2f',
        cmap='YlOrBr',
        mask=mask,
        ax=ax,
        linewidths=0.8,
        linecolor='#1a1a1a',
        annot_kws={'size': 9},
        vmin=-1, vmax=1,
        cbar_kws={'label': 'Correlation'}
    )
    ax.set_title('Matrice de correlation (valeurs entre -1 et 1)', color=GOLD, fontsize=11, pad=15)

    plt.tight_layout()
    path = f'{FIGURES_DIR}/05_heatmap.png'
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"Figure sauvegardee : {path}")


# =============================================================================
# 7. DISTRIBUTION DES FEATURES V
# =============================================================================

def plot_features_distribution(df: pd.DataFrame, n_features: int = 12) -> None:
    """Affiche la distribution des features V1-V28 pour chaque classe."""
    features = [f'V{i}' for i in range(1, n_features + 1)]
    normal = df[df['Class'] == 0]
    fraud  = df[df['Class'] == 1]

    ncols = 4
    nrows = (n_features + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(20, nrows * 4))
    fig.suptitle(f'Distribution des Features V1-V{n_features} : Normal vs Fraude', color=GOLD, fontsize=15, fontweight='bold')
    axes_flat = axes.flatten()

    for i, feat in enumerate(features):
        ax = axes_flat[i]
        ax.hist(normal[feat], bins=60, density=True, color=GOLD, alpha=0.6, label='Normal')
        ax.hist(fraud[feat],  bins=60, density=True, color=RED,  alpha=0.8, label='Fraude')
        ax.set_title(feat, color=GOLD, fontsize=11)
        ax.set_xlabel('Valeur')
        ax.set_ylabel('Densite')
        ax.legend(fontsize=8)
        ax.axvline(normal[feat].mean(), color=GOLD, linestyle='--', linewidth=1, alpha=0.7)
        ax.axvline(fraud[feat].mean(),  color=RED,  linestyle='--', linewidth=1, alpha=0.7)

    for j in range(i + 1, len(axes_flat)):
        axes_flat[j].set_visible(False)

    plt.tight_layout()
    path = f'{FIGURES_DIR}/06_features_distribution.png'
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"Figure sauvegardee : {path}")


# =============================================================================
# 8. DETECTION DES VALEURS ABERRANTES
# =============================================================================

def plot_outliers(df: pd.DataFrame) -> None:
    """Detectiondes valeurs aberrantes avec IQR sur Amount et Time."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('Detection des Valeurs Aberrantes', color=GOLD, fontsize=15, fontweight='bold')

    for i, col in enumerate(['Amount', 'Time']):
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        outliers = df[(df[col] < lower) | (df[col] > upper)]

        axes[i].hist(df[col], bins=60, color=GOLD, alpha=0.6, density=True, label='Distribution')
        axes[i].axvline(lower, color=RED,   linestyle='--', linewidth=2, label=f'Borne inf : {lower:.1f}')
        axes[i].axvline(upper, color=RED,   linestyle='--', linewidth=2, label=f'Borne sup : {upper:.1f}')
        axes[i].axvline(Q1,    color=GREEN, linestyle=':',  linewidth=1.5, label=f'Q1 : {Q1:.1f}')
        axes[i].axvline(Q3,    color=GREEN, linestyle=':',  linewidth=1.5, label=f'Q3 : {Q3:.1f}')
        axes[i].set_title(f'Outliers dans {col} ({len(outliers):,} detectes)', color=GOLD)
        axes[i].set_xlabel(col)
        axes[i].set_ylabel('Densite')
        axes[i].legend(fontsize=8)

    plt.tight_layout()
    path = f'{FIGURES_DIR}/07_outliers.png'
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"Figure sauvegardee : {path}")

    for col in ['Amount', 'Time']:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        n_outliers = len(df[(df[col] < Q1 - 1.5*IQR) | (df[col] > Q3 + 1.5*IQR)])
        print(f"  {col} : {n_outliers:,} valeurs aberrantes ({n_outliers/len(df)*100:.2f}%)")


# =============================================================================
# PIPELINE EDA COMPLET
# =============================================================================

def run_eda(df: pd.DataFrame) -> None:
    """
    Lance l'analyse exploratoire complete dans l'ordre.

    Args:
        df : DataFrame du dataset creditcard
    """
    print("\n" + "=" * 60)
    print("  ANALYSE EXPLORATOIRE DES DONNEES (EDA)")
    print("=" * 60)

    print("\n[1/7] Informations generales...")
    print_dataset_info(df)

    print("\n[2/7] Distribution des classes...")
    plot_class_distribution(df)

    print("\n[3/7] Analyse des montants...")
    plot_amount_analysis(df)

    print("\n[4/7] Distribution temporelle...")
    plot_time_distribution(df)

    print("\n[5/7] Correlation avec la cible...")
    top_pos, top_neg = plot_correlation(df)

    print("\n[6/7] Heatmap de correlation...")
    plot_heatmap(df, top_pos, top_neg)

    print("\n[7/7] Distribution des features V1-V12...")
    plot_features_distribution(df, n_features=12)

    print("\n[8/8] Detection des outliers...")
    plot_outliers(df)

    print("\nEDA terminee.")
    print(f"Tous les graphiques sont dans : {FIGURES_DIR}/")
