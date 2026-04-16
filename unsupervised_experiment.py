"""
╔══════════════════════════════════════════════════════════════════╗
║   UNSUPERVISED LEARNING EXPERIMENT                               ║
║   DLL Injection Detection - Zero-Day Simulation                  ║
║                                                                  ║
║   Student:    Falah Nizam  SID: 2228945                          ║
║   Supervisor: Ronak Al-Hadad  -  ARU FYP                         ║
╚══════════════════════════════════════════════════════════════════╝

EXPERIMENTS:
    Exp 1 - Isolation Forest anomaly detection (all techniques)
    Exp 2 - KMeans clustering - do techniques form natural clusters?
    Exp 3 - UMAP visualisation - feature space structure
    Exp 4 - Leave-One-Technique-Out (LOTO) - pseudo zero-day test
             Train on benign + 4 techniques, test if withheld
             technique is detected as anomalous

SECTION MAPPING NOTE:
    Section mapping (NtMapViewOfSection) is discussed as a
    theoretical extension. It bypasses VirtualAllocEx entirely,
    using shared memory sections instead of cross-process writes.
    The LOTO experiment simulates this zero-day scenario using
    existing techniques as stand-ins.

OUTPUTS (saved to ml_results/):
    unsupervised_anomaly_scores.png
    unsupervised_clustering.png
    unsupervised_umap.png
    unsupervised_loto_results.png
    unsupervised_results.csv
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.ensemble import IsolationForest
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (adjusted_rand_score, silhouette_score,
                              roc_auc_score, f1_score, precision_score,
                              recall_score, confusion_matrix)
from sklearn.decomposition import PCA
warnings.filterwarnings('ignore')

# Try importing UMAP - fall back to PCA if not available
try:
    import umap
    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False
    print("  [!] UMAP not available - using PCA for visualisation")

DATASET_PATH = "master_dataset_v4.csv"
OUTPUT_DIR   = "ml_results"
RANDOM_SEED  = 42
os.makedirs(OUTPUT_DIR, exist_ok=True)

V4_FEATURES = [
    'PageFaultCount', 'ThreadCount', 'ImageLoadCount', 'ThreadToPFRatio',
    'Is_Tainted_Past', 'PF_Normal', 'PF_Suspicious', 'PF_LargeRegion',
    'PF_CopyOnWrite', 'Thread_External', 'Image_Unsigned',
    'Image_SuspiciousPath', 'Has_InjectionPattern', 'Has_CreateThread',
    'Has_UnknownCallTrace', 'Has_FullAccess',
    'Has_VM_Write', 'Has_VM_Read', 'Has_VM_Operation',
    'Has_CreateThread_Right', 'Has_InjPattern_Right', 'Access_Rights_Count'
]

TECHNIQUE_COLORS = {
    'Benign':       '#95a5a6',
    'Classic_CRT':  '#e74c3c',
    'Classic_Hook': '#e67e22',
    'Reflective':   '#3498db',
    'Sideloading':  '#27ae60',
    'Mixed':        '#9b59b6',
}

def _header(title):
    print("\n" + "=" * 65)
    print(f" {title}")
    print("=" * 65)

# ─────────────────────────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────────────────────────
_header("Loading Dataset")
df = pd.read_csv(DATASET_PATH)
print(f"  Rows: {len(df):,}  |  Benign: {(df['Label']==0).sum():,}"
      f"  |  Malicious: {(df['Label']==1).sum():,}")
print(f"  Attack types: {df['AttackType'].value_counts().to_dict()}")

X_all    = df[V4_FEATURES].values
y_binary = df['Label'].values
y_type   = df['AttackType'].values

# Subsample benign for speed (keep all malicious)
np.random.seed(RANDOM_SEED)
benign_idx   = np.where(y_binary == 0)[0]
mal_idx      = np.where(y_binary == 1)[0]
sample_benign = np.random.choice(benign_idx,
                                  size=min(10000, len(benign_idx)),
                                  replace=False)
use_idx = np.concatenate([sample_benign, mal_idx])
X_use   = X_all[use_idx]
y_use   = y_binary[use_idx]
t_use   = y_type[use_idx]

scaler  = StandardScaler()
X_scaled = scaler.fit_transform(X_use)

print(f"\n  Working set: {len(X_use):,} rows "
      f"({len(sample_benign):,} benign + {len(mal_idx):,} malicious)")

# ─────────────────────────────────────────────────────────────────
# EXPERIMENT 1 - ISOLATION FOREST
# ─────────────────────────────────────────────────────────────────
_header("Experiment 1: Isolation Forest Anomaly Detection")

# Contamination = fraction of malicious in working set
contamination = len(mal_idx) / len(X_use)
print(f"  Contamination parameter: {contamination:.4f} "
      f"({contamination*100:.1f}% malicious)")

results_iso = []
for contam_val, label in [
    (contamination,       'Matched contamination'),
    (contamination * 0.5, 'Half contamination'),
    (contamination * 2.0, 'Double contamination'),
]:
    contam_val = min(contam_val, 0.499)
    iso = IsolationForest(n_estimators=200,
                          contamination=contam_val,
                          random_state=RANDOM_SEED,
                          n_jobs=-1)
    iso.fit(X_scaled)
    # IsolationForest: -1 = anomaly (malicious), 1 = normal (benign)
    preds = iso.predict(X_scaled)
    y_pred = (preds == -1).astype(int)
    scores = -iso.score_samples(X_scaled)  # Higher = more anomalous

    f1   = f1_score(y_use, y_pred)
    prec = precision_score(y_use, y_pred)
    rec  = recall_score(y_use, y_pred)
    auc  = roc_auc_score(y_use, scores)
    fpr  = ((y_pred==1) & (y_use==0)).sum() / (y_use==0).sum()

    results_iso.append({
        'Config': label, 'F1': round(f1,4), 'Precision': round(prec,4),
        'Recall': round(rec,4), 'AUC': round(auc,4), 'FPR': round(fpr,4)
    })
    print(f"\n  [{label}]")
    print(f"  F1={f1:.4f}  Prec={prec:.4f}  Rec={rec:.4f}"
          f"  AUC={auc:.4f}  FPR={fpr:.4f}")

# Best config for plots
iso_best = IsolationForest(n_estimators=200,
                            contamination=contamination,
                            random_state=RANDOM_SEED, n_jobs=-1)
iso_best.fit(X_scaled)
anomaly_scores = -iso_best.score_samples(X_scaled)

# Per-technique anomaly scores
print(f"\n  Per-technique mean anomaly score:")
print(f"  {'Technique':<16} {'Mean Score':>12} {'Median':>10} {'> threshold':>12}")
threshold = np.percentile(anomaly_scores, (1 - contamination) * 100)
for atype in ['Benign', 'Classic_CRT', 'Classic_Hook',
              'Reflective', 'Sideloading', 'Mixed']:
    mask   = t_use == atype
    if mask.sum() == 0: continue
    scores_t = anomaly_scores[mask]
    above    = (scores_t > threshold).sum()
    print(f"  {atype:<16} {scores_t.mean():>12.4f} "
          f"{np.median(scores_t):>10.4f} "
          f"{above:>6}/{mask.sum()} ({above/mask.sum()*100:.1f}%)")

# Plot anomaly score distributions
fig, axes = plt.subplots(2, 3, figsize=(14, 8))
axes = axes.flatten()
attack_types = ['Benign', 'Classic_CRT', 'Classic_Hook',
                'Reflective', 'Sideloading', 'Mixed']

for i, atype in enumerate(attack_types):
    mask = t_use == atype
    if mask.sum() == 0:
        axes[i].text(0.5, 0.5, 'No data', ha='center', transform=axes[i].transAxes)
        continue
    color = TECHNIQUE_COLORS.get(atype, '#888')
    axes[i].hist(anomaly_scores[mask], bins=40, color=color,
                  alpha=0.75, edgecolor='white', linewidth=0.3)
    axes[i].axvline(threshold, color='red', linestyle='--',
                     linewidth=1.5, label=f'Threshold')
    above = (anomaly_scores[mask] > threshold).sum()
    det_rate = above / mask.sum() * 100
    axes[i].set_title(f'{atype}\n(n={mask.sum()}, detected={det_rate:.1f}%)',
                       fontsize=10, fontweight='bold')
    axes[i].set_xlabel('Anomaly Score', fontsize=8)
    axes[i].set_ylabel('Count', fontsize=8)
    axes[i].legend(fontsize=7)
    axes[i].grid(True, alpha=0.3)

plt.suptitle('Isolation Forest - Anomaly Score Distributions by Technique',
             fontweight='bold', fontsize=12)
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/unsupervised_anomaly_scores.png',
            dpi=150, bbox_inches='tight')
plt.close()
print(f"\n  Saved: unsupervised_anomaly_scores.png")

# ─────────────────────────────────────────────────────────────────
# EXPERIMENT 2 - KMEANS CLUSTERING
# ─────────────────────────────────────────────────────────────────
_header("Experiment 2: KMeans Clustering")

# Try k=2 (binary) and k=6 (per technique)
print("\n  Testing k=2 (binary: benign vs malicious)...")
km2 = KMeans(n_clusters=2, random_state=RANDOM_SEED, n_init=10)
km2_labels = km2.fit_predict(X_scaled)
ari2 = adjusted_rand_score(y_use, km2_labels)
sil2 = silhouette_score(X_scaled, km2_labels, sample_size=2000,
                         random_state=RANDOM_SEED)
print(f"  Adjusted Rand Index: {ari2:.4f}  (1.0=perfect, 0=random)")
print(f"  Silhouette Score:    {sil2:.4f}  (1.0=perfect separation)")

print("\n  Testing k=6 (one per technique)...")
km6 = KMeans(n_clusters=6, random_state=RANDOM_SEED, n_init=10)
km6_labels = km6.fit_predict(X_scaled)

# Encode attack types numerically for ARI
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y_type_enc = le.fit_transform(t_use)
ari6 = adjusted_rand_score(y_type_enc, km6_labels)
sil6 = silhouette_score(X_scaled, km6_labels, sample_size=2000,
                         random_state=RANDOM_SEED)
print(f"  Adjusted Rand Index: {ari6:.4f}")
print(f"  Silhouette Score:    {sil6:.4f}")

# Cluster purity analysis for k=6
print(f"\n  Cluster composition (k=6) - dominant technique per cluster:")
print(f"  {'Cluster':>8} {'Dominant Technique':<18} {'Purity':>8} {'Size':>7}")
print(f"  {'-'*50}")
for c in range(6):
    mask    = km6_labels == c
    if mask.sum() == 0: continue
    types_c = t_use[mask]
    counts  = pd.Series(types_c).value_counts()
    dominant = counts.index[0]
    purity   = counts.iloc[0] / mask.sum()
    print(f"  {c:>8} {dominant:<18} {purity:>7.1%} {mask.sum():>7,}")

# Elbow curve - optimal k
inertias = []
k_range  = range(2, 12)
for k in k_range:
    km = KMeans(n_clusters=k, random_state=RANDOM_SEED, n_init=5)
    km.fit(X_scaled)
    inertias.append(km.inertia_)

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# Elbow plot
axes[0].plot(list(k_range), inertias, 'o-', color='#003865',
              linewidth=2, markersize=6)
axes[0].axvline(x=6, color='#e74c3c', linestyle='--', linewidth=1.5,
                 label='k=6 (num techniques)')
axes[0].axvline(x=2, color='#27ae60', linestyle='--', linewidth=1.5,
                 label='k=2 (binary)')
axes[0].set_xlabel('Number of Clusters (k)')
axes[0].set_ylabel('Inertia (Within-cluster sum of squares)')
axes[0].set_title('KMeans Elbow Curve\n(Optimal k selection)')
axes[0].legend(); axes[0].grid(True, alpha=0.3)

# Cluster heatmap - technique vs cluster assignment (k=6)
ct = pd.crosstab(pd.Series(t_use, name='Technique'),
                  pd.Series(km6_labels, name='Cluster'))
ct_norm = ct.div(ct.sum(axis=1), axis=0)
sns.heatmap(ct_norm, annot=True, fmt='.2f', cmap='YlOrRd',
             ax=axes[1], vmin=0, vmax=1)
axes[1].set_title(f'Technique Distribution per Cluster (k=6)\n'
                   f'ARI={ari6:.3f}  Silhouette={sil6:.3f}')
axes[1].set_xlabel('Cluster'); axes[1].set_ylabel('True Technique')

plt.suptitle('KMeans Clustering Analysis', fontweight='bold')
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/unsupervised_clustering.png',
            dpi=150, bbox_inches='tight')
plt.close()
print(f"\n  Saved: unsupervised_clustering.png")

# ─────────────────────────────────────────────────────────────────
# EXPERIMENT 3 - DIMENSIONALITY REDUCTION VISUALISATION
# ─────────────────────────────────────────────────────────────────
_header("Experiment 3: Feature Space Visualisation")

# Subsample for speed
viz_idx = np.random.choice(len(X_scaled),
                            size=min(4000, len(X_scaled)),
                            replace=False)
X_viz   = X_scaled[viz_idx]
t_viz   = t_use[viz_idx]
y_viz   = y_use[viz_idx]

if HAS_UMAP:
    print("  Running UMAP (2D projection)...")
    reducer = umap.UMAP(n_components=2, random_state=RANDOM_SEED,
                         n_neighbors=15, min_dist=0.1)
    X_2d    = reducer.fit_transform(X_viz)
    method  = "UMAP"
else:
    print("  Running PCA (2D projection)...")
    pca  = PCA(n_components=2, random_state=RANDOM_SEED)
    X_2d = pca.fit_transform(X_viz)
    var  = pca.explained_variance_ratio_
    method = f"PCA (PC1={var[0]*100:.1f}%, PC2={var[1]*100:.1f}%)"

print(f"  Projection complete. Plotting...")

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Plot 1: Colour by technique
for atype, color in TECHNIQUE_COLORS.items():
    mask = t_viz == atype
    if mask.sum() == 0: continue
    alpha = 0.3 if atype == 'Benign' else 0.7
    size  = 4   if atype == 'Benign' else 12
    axes[0].scatter(X_2d[mask, 0], X_2d[mask, 1],
                     c=color, s=size, alpha=alpha,
                     label=f'{atype} (n={mask.sum()})')
axes[0].set_title(f'{method} - Coloured by Technique',
                   fontweight='bold')
axes[0].set_xlabel(f'{method.split(" ")[0]} Dim 1')
axes[0].set_ylabel(f'{method.split(" ")[0]} Dim 2')
axes[0].legend(fontsize=8, markerscale=2)
axes[0].grid(True, alpha=0.2)

# Plot 2: Binary - benign vs malicious with anomaly score overlay
sc = axes[1].scatter(X_2d[:, 0], X_2d[:, 1],
                      c=y_viz, cmap='RdYlGn_r',
                      s=5, alpha=0.5)
plt.colorbar(sc, ax=axes[1], label='Label (0=Benign, 1=Malicious)')
axes[1].set_title(f'{method} - Binary Label\n(Green=Benign, Red=Malicious)',
                   fontweight='bold')
axes[1].set_xlabel(f'{method.split(" ")[0]} Dim 1')
axes[1].set_ylabel(f'{method.split(" ")[0]} Dim 2')
axes[1].grid(True, alpha=0.2)

plt.suptitle('Feature Space Structure - DLL Injection Dataset (V4 Features)',
             fontweight='bold', fontsize=12)
plt.tight_layout()
save_name = 'unsupervised_umap.png' if HAS_UMAP else 'unsupervised_pca.png'
plt.savefig(f'{OUTPUT_DIR}/{save_name}', dpi=150, bbox_inches='tight')
plt.close()
print(f"  Saved: {save_name}")

# ─────────────────────────────────────────────────────────────────
# EXPERIMENT 4 - LEAVE-ONE-TECHNIQUE-OUT (Zero-Day Simulation)
# ─────────────────────────────────────────────────────────────────
_header("Experiment 4: Leave-One-Technique-Out (Zero-Day Simulation)")

print("""
  Concept: Train Isolation Forest on benign + 4 known techniques.
  Test whether the withheld 5th technique is flagged as anomalous.
  This simulates encountering a never-before-seen injection method
  (analogous to Section Mapping / NtMapViewOfSection).
""")

techniques  = ['Classic_CRT', 'Classic_Hook', 'Reflective',
                'Sideloading', 'Mixed']
loto_results = []

fig, axes = plt.subplots(2, 3, figsize=(15, 9))
axes = axes.flatten()

for i, withheld in enumerate(techniques):
    print(f"  [{i+1}/5] Withheld technique: {withheld}")

    # Training set: benign + all techniques EXCEPT withheld
    known_types  = ['Benign'] + [t for t in techniques if t != withheld]
    train_mask   = np.isin(t_use, known_types)
    test_mask    = t_use == withheld

    X_train = X_scaled[train_mask]
    X_test  = X_scaled[test_mask]
    y_train = y_use[train_mask]

    if X_test.shape[0] == 0:
        print(f"    No test samples for {withheld}, skipping")
        continue

    # Train contamination = fraction of malicious in training set
    train_contam = y_train.mean()
    train_contam = min(max(train_contam, 0.01), 0.499)

    iso_loto = IsolationForest(n_estimators=200,
                                contamination=train_contam,
                                random_state=RANDOM_SEED,
                                n_jobs=-1)
    iso_loto.fit(X_train)

    # Score the withheld technique
    scores_withheld = -iso_loto.score_samples(X_test)
    scores_benign   = -iso_loto.score_samples(
        X_scaled[t_use == 'Benign'][:500])

    # Threshold from training set
    train_scores = -iso_loto.score_samples(X_train)
    threshold_loto = np.percentile(train_scores,
                                    (1 - train_contam) * 100)

    detected     = (scores_withheld > threshold_loto).sum()
    detect_rate  = detected / len(scores_withheld)
    mean_score   = scores_withheld.mean()
    benign_mean  = scores_benign.mean()
    separation   = mean_score - benign_mean

    print(f"    Detected: {detected}/{len(scores_withheld)} "
          f"({detect_rate*100:.1f}%)  "
          f"Mean anomaly score: {mean_score:.4f}  "
          f"Separation from benign: {separation:+.4f}")

    loto_results.append({
        'Withheld':       withheld,
        'N_test':         len(scores_withheld),
        'Detected':       detected,
        'Detection_Rate': round(detect_rate, 4),
        'Mean_Score':     round(mean_score, 4),
        'Benign_Mean':    round(benign_mean, 4),
        'Separation':     round(separation, 4),
    })

    # Plot score distributions
    ax = axes[i]
    ax.hist(scores_benign, bins=30, color='#95a5a6',
             alpha=0.6, label='Benign', density=True)
    ax.hist(scores_withheld, bins=30,
             color=TECHNIQUE_COLORS.get(withheld, '#e74c3c'),
             alpha=0.7, label=withheld, density=True)
    ax.axvline(threshold_loto, color='black', linestyle='--',
                linewidth=1.5, label='Threshold')
    ax.set_title(f'Withheld: {withheld}\nDetection: {detect_rate*100:.1f}%',
                  fontsize=10, fontweight='bold')
    ax.set_xlabel('Anomaly Score', fontsize=8)
    ax.set_ylabel('Density', fontsize=8)
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

# Summary bar chart in last subplot
ax = axes[5]
if loto_results:
    loto_df = pd.DataFrame(loto_results)
    colors_bar = [TECHNIQUE_COLORS.get(t, '#888')
                  for t in loto_df['Withheld']]
    bars = ax.bar(loto_df['Withheld'],
                   loto_df['Detection_Rate'] * 100,
                   color=colors_bar, alpha=0.85, width=0.6)
    ax.set_ylabel('Detection Rate (%)')
    ax.set_title('LOTO Summary\n(Zero-Day Detection Rate)', fontweight='bold')
    ax.set_ylim(0, 110)
    ax.tick_params(axis='x', rotation=25)
    ax.grid(True, alpha=0.3, axis='y')
    for bar, val in zip(bars, loto_df['Detection_Rate'] * 100):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 1,
                f'{val:.1f}%', ha='center', va='bottom',
                fontsize=9, fontweight='bold')
    ax.axhline(y=50, color='orange', linestyle='--',
                linewidth=1, alpha=0.7, label='50% baseline')
    ax.legend(fontsize=8)

plt.suptitle('Leave-One-Technique-Out (LOTO) - Zero-Day Injection Detection\n'
             'Isolation Forest trained on 4 known techniques, tested on withheld',
             fontweight='bold', fontsize=11)
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/unsupervised_loto_results.png',
            dpi=150, bbox_inches='tight')
plt.close()
print(f"\n  Saved: unsupervised_loto_results.png")

# ─────────────────────────────────────────────────────────────────
# SUMMARY
# ─────────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print(" UNSUPERVISED EXPERIMENT SUMMARY")
print("=" * 65)

print(f"\n  Experiment 1 - Isolation Forest (full dataset):")
for r in results_iso:
    print(f"  {r['Config']:<28} F1={r['F1']:.4f}  "
          f"AUC={r['AUC']:.4f}  FPR={r['FPR']:.4f}")

print(f"\n  Experiment 2 - KMeans Clustering:")
print(f"  k=2 (binary)   ARI={ari2:.4f}  Silhouette={sil2:.4f}")
print(f"  k=6 (technique) ARI={ari6:.4f}  Silhouette={sil6:.4f}")

if loto_results:
    loto_df = pd.DataFrame(loto_results)
    print(f"\n  Experiment 4 - Leave-One-Technique-Out (Zero-Day):")
    print(f"  {'Technique':<16} {'Detection Rate':>16} {'Separation':>12}")
    print(f"  {'-'*48}")
    for _, row in loto_df.iterrows():
        verdict = ("DETECTABLE" if row['Detection_Rate'] > 0.5
                   else "EVASIVE")
        print(f"  {row['Withheld']:<16} "
              f"{row['Detection_Rate']*100:>14.1f}%  "
              f"{row['Separation']:>+10.4f}  {verdict}")

    print(f"\n  Section Mapping Prediction:")
    avg_sep = loto_df['Separation'].mean()
    evasive = loto_df[loto_df['Detection_Rate'] < 0.5]
    print(f"  Average separation score: {avg_sep:+.4f}")
    print(f"  Techniques that evade unsupervised detection:")
    for _, row in evasive.iterrows():
        print(f"    - {row['Withheld']} ({row['Detection_Rate']*100:.1f}% detected)")
    print(f"""
  Section mapping (NtMapViewOfSection) would likely be HARDER to
  detect than the above because it:
    - Does not use VirtualAllocEx (no VM_Write signal)
    - Does not create remote thread (no Thread_External signal)
    - Uses NtMapViewOfSection which ETW does not log by default
  This supports the Volatility future work recommendation.""")

# Save all results
all_results = {
    'isolation_forest': pd.DataFrame(results_iso),
    'kmeans': pd.DataFrame([
        {'Method': 'KMeans k=2', 'ARI': ari2, 'Silhouette': sil2},
        {'Method': 'KMeans k=6', 'ARI': ari6, 'Silhouette': sil6},
    ]),
}
if loto_results:
    all_results['loto'] = pd.DataFrame(loto_results)

with pd.ExcelWriter(f'{OUTPUT_DIR}/unsupervised_results.xlsx') as writer:
    for sheet, data in all_results.items():
        data.to_excel(writer, sheet_name=sheet, index=False)

print(f"\n  Saved: unsupervised_results.xlsx")
print(f"\n  All outputs in: {OUTPUT_DIR}/")
print(f"    unsupervised_anomaly_scores.png")
print(f"    unsupervised_clustering.png")
print(f"    {'unsupervised_umap.png' if HAS_UMAP else 'unsupervised_pca.png'}")
print(f"    unsupervised_loto_results.png")
print(f"    unsupervised_results.xlsx")

print(f"\n{'='*65}")
print(f" DONE")
print(f"{'='*65}")
