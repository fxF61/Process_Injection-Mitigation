"""
╔══════════════════════════════════════════════════════════════════╗
║   DLL INJECTION DETECTION PIPELINE — Final Artifact             ║
║   BEng (Hons) Computer Science — Final Year Project             ║
║                                                                  ║
║   Student:    Falah Nizam                                        ║
║   SID:        2228945                                            ║
║   Supervisor: Ronak Al-Hadad                                     ║
║   Ethics Ref: ETH2526-1235                                       ║
║   Institution: Anglia Ruskin University                          ║
╚══════════════════════════════════════════════════════════════════╝

USAGE:
    python detection_pipeline.py                 # Run all experiments (uses existing CSV)
    python detection_pipeline.py --build-dataset # Rebuild dataset from raw logs first

STAGES:
    Stage 1 — Dataset Builder      (--build-dataset flag only)
    Stage 2 — Binary Detection     V1 vs V2 vs V4 features, all 4 models
    Stage 3 — Multiclass Attribution  RF + XGBoost, V4 features
    Stage 4 — SHAP Explainability  Global + per-technique
    Stage 5 — Rule-Based Baseline  MITRE ATT&CK T1055 heuristics
    Stage 6 — Final Summary        All results consolidated

OUTPUTS (saved to ml_results/):
    binary_results.csv             All model binary detection results
    multiclass_results.csv         Per-class multiclass results
    rule_based_comparison.csv      Rule-based vs ML comparison
    dissertation_results.csv       Master results table
    *.png                          All figures

REQUIREMENTS:
    pip install scikit-learn xgboost imbalanced-learn shap matplotlib seaborn pandas numpy
"""

import os
import sys
import time
import json
import warnings
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from collections import defaultdict
warnings.filterwarnings('ignore')

# ── Auto-install missing packages ────────────────────────────────
import subprocess
def _install(pkg):
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', pkg, '-q'])

for _pkg, _name in [('sklearn','scikit-learn'),('xgboost','xgboost'),
                     ('imblearn','imbalanced-learn'),('shap','shap')]:
    try: __import__(_pkg)
    except: _install(_name)

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import (f1_score, precision_score, recall_score,
                              roc_auc_score, confusion_matrix,
                              classification_report, roc_curve)
from xgboost import XGBClassifier
import shap

# ─────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────
DATA_ROOT      = r"C:\Users\falah\Desktop\New folder\Data"
DATASET_PATH   = "master_dataset_v4.csv"
OUTPUT_DIR     = "ml_results"
RANDOM_SEED    = 42
TEST_SIZE      = 0.2
WINDOW_SIZE    = 5
TAINT_DURATION = 300

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Feature sets
V1_FEATURES = [
    'PageFaultCount', 'ThreadCount', 'ImageLoadCount',
    'ThreadToPFRatio', 'Is_Tainted_Past'
]

V2_FEATURES = [
    'PageFaultCount', 'ThreadCount', 'ImageLoadCount', 'ThreadToPFRatio',
    'Is_Tainted_Past', 'PF_Normal', 'PF_Suspicious', 'PF_LargeRegion',
    'PF_CopyOnWrite', 'Thread_External', 'Image_Unsigned',
    'Image_SuspiciousPath', 'Has_InjectionPattern', 'Has_CreateThread',
    'Has_UnknownCallTrace', 'Has_FullAccess'
]

V4_FEATURES = V2_FEATURES + [
    'Has_VM_Write', 'Has_VM_Read', 'Has_VM_Operation',
    'Has_CreateThread_Right', 'Has_InjPattern_Right', 'Access_Rights_Count'
]

ATTACK_TYPE_FOLDERS = {
    "classic_crt":  "Classic_CRT",
    "classic_hook": "Classic_Hook",
    "new-alpc":     "Reflective",
    "alpc":         "Reflective",
    "remote_rdll":  "Reflective",
    "rdll":         "Reflective",
    "side-loading": "Sideloading",
    "sideloading":  "Sideloading",
    "atomic_sample":"Mixed",
    "atomic":       "Mixed",
    "case 1":       "Classic_CRT",
    "case 2":       "Classic_CRT",
    "pinjectra":    "Classic_CRT",
}

SUSPICIOUS_PF_FLAGS = {32768, 1056768}

# ─────────────────────────────────────────────────────────────────
# SHARED HELPERS
# ─────────────────────────────────────────────────────────────────
def _header(title):
    print("\n" + "=" * 65)
    print(f" {title}")
    print("=" * 65)

def _get_split(df, features, label_col='Label'):
    X = df[features].values
    y = df[label_col].values
    return train_test_split(X, y, test_size=TEST_SIZE,
                            random_state=RANDOM_SEED, stratify=y)

def _evaluate(name, model, X_te, y_te, train_time, scaler=None):
    if scaler is not None:
        X_te = scaler.transform(X_te)
    t0     = time.time()
    y_pred = model.predict(X_te)
    inf_ms = (time.time() - t0) / len(X_te) * 1000
    y_prob = model.predict_proba(X_te)[:, 1]
    fpr_val = ((y_pred == 1) & (y_te == 0)).sum() / (y_te == 0).sum()
    return {
        'Model':     name,
        'F1':        round(f1_score(y_te, y_pred), 4),
        'Precision': round(precision_score(y_te, y_pred), 4),
        'Recall':    round(recall_score(y_te, y_pred), 4),
        'AUC':       round(roc_auc_score(y_te, y_prob), 4),
        'FPR':       round(fpr_val, 4),
        'Train_s':   round(train_time, 3),
        'Infer_ms':  round(inf_ms, 4),
        'y_pred':    y_pred,
        'y_prob':    y_prob,
    }

def _print_table(rows, title=''):
    if title: print(f"\n  {title}")
    print(f"  {'Model':<28} {'F1':>7} {'Prec':>7} {'Rec':>7} "
          f"{'AUC':>7} {'FPR':>7} {'Train_s':>9}")
    print(f"  {'-'*72}")
    for r in rows:
        print(f"  {r['Model']:<28} {r['F1']:>7.4f} {r['Precision']:>7.4f} "
              f"{r['Recall']:>7.4f} {r['AUC']:>7.4f} {r['FPR']:>7.4f} "
              f"{r['Train_s']:>9.3f}s")

def _safe_shap(explainer, X, n_features):
    sv = explainer.shap_values(X)
    if isinstance(sv, list):
        arr = np.array(sv[1])
    elif hasattr(sv, 'ndim') and sv.ndim == 3:
        arr = sv[:, :, 1]
    else:
        arr = np.array(sv)
    return arr.reshape(len(X), -1)[:, :n_features]

# ─────────────────────────────────────────────────────────────────
# STAGE 1 — DATASET BUILDER
# ─────────────────────────────────────────────────────────────────
def _get_epoch(ts_str):
    if not ts_str: return 0
    try:
        if "-" in str(ts_str):
            clean = str(ts_str).replace("Z","").split(".")[0]
            return int(datetime.strptime(clean, "%Y-%m-%d %H:%M:%S").timestamp())
        else:
            return int(datetime.strptime(str(ts_str), "%d/%m/%Y %H:%M:%S").timestamp())
    except: return 0

def _detect_encoding(path):
    for enc in ('utf-8-sig','utf-16','utf-16-le','utf-8','latin-1'):
        try:
            with open(path,'r',encoding=enc) as f: f.read(1)
            return enc
        except: continue
    return 'latin-1'

def _iter_events(json_path, encoding):
    depth, buffer = 0, []
    with open(json_path,'r',encoding=encoding) as f:
        for line in f:
            s = line.strip()
            if not s: continue
            depth += s.count('{') - s.count('}')
            buffer.append(line)
            if depth == 0 and buffer:
                raw = "".join(buffer).strip().rstrip(',')
                if raw:
                    try: yield json.loads(raw)
                    except: pass
                buffer = []

def _parse_access(access_str):
    s = str(access_str).upper()
    parts = [p.strip() for p in s.split("|")
             if p.strip() and not p.strip().startswith("**")]
    return {
        'Has_VM_Write':          int("VM_WRITE"    in s),
        'Has_VM_Read':           int("VM_READ"     in s),
        'Has_VM_Operation':      int("VM_OPERATION" in s),
        'Has_CreateThread_Right':int("CREATE_THREAD" in s),
        'Has_InjPattern_Right':  int("INJECTION_PATTERN" in s),
        'Access_Rights_Count':   len(parts),
    }

def _parse_calltrace(trace_str):
    s = str(trace_str)
    if not s or s in ("","nan","N/A"):
        return {'CallTrace_Depth': 0, 'CallTrace_Unknown_Count': 0}
    mods = [m.strip() for m in s.split("|") if m.strip()]
    return {
        'CallTrace_Depth':         len(mods),
        'CallTrace_Unknown_Count': sum(1 for m in mods if m.upper().startswith("UNKNOWN(")),
    }

def _extract_csv_signals(csv_path, tz_offset):
    indicators, tainted, votes = [], {}, defaultdict(int)
    try: df = pd.read_csv(csv_path, encoding='utf-8-sig')
    except: return indicators, tainted, "Unknown"
    for _, row in df.iterrows():
        likely   = str(row.get("LikelyInjection","0")) == "1"
        eid      = str(row.get("EventID",""))
        acc      = str(row.get("AccessDecoded",""))
        trace    = str(row.get("CallTrace",""))
        cmd      = str(row.get("CommandLine","")).lower()
        target   = str(row.get("TargetImage","")).lower().split("\\")[-1]
        source   = str(row.get("SourceImage","")).lower().split("\\")[-1]
        image    = str(row.get("Image","")).lower()
        etime    = _get_epoch(str(row.get("TimeCreated",""))) - tz_offset
        if not (likely or eid == "10"): continue
        has_inj  = "INJECTION_PATTERN" in acc
        has_ct   = "CREATE_THREAD"     in acc
        has_unk  = "UNKNOWN("          in trace
        has_refl = "reflective"        in cmd or "reflective" in trace.lower()
        has_full = "0x1fffff"          in str(row.get("GrantedAccess","")).lower()
        is_img   = eid == "7"
        susp_path= (is_img and "system32" not in image
                    and "syswow64" not in image and image.endswith(".dll"))
        if has_unk or has_refl: votes["Reflective"] += 2
        if has_ct and not has_unk: votes["Classic"]  += 2
        if susp_path: votes["Sideloading"] += 2
        if has_inj:
            best = max(votes, key=votes.get) if votes else "Classic"
            votes[best] += 1
        acc_feats   = _parse_access(acc)
        trace_feats = _parse_calltrace(trace)
        indicators.append({
            'target': target, 'source': source, 'time': etime,
            'has_injection_pattern': int(has_inj),
            'has_create_thread':     int(has_ct),
            'has_unknown_trace':     int(has_unk),
            'has_full_access':       int(has_full),
            **acc_feats, **trace_feats,
        })
        for name in [target, source]:
            if name and name != 'nan':
                if name not in tainted or etime < tainted[name]:
                    tainted[name] = etime
    detected = max(votes, key=votes.get) if votes else "Unknown"
    return indicators, tainted, detected

def _process_pair(json_path, csv_path, is_malicious=True, forced_type=None):
    print(f"  -> {os.path.basename(json_path)}")
    features, encoding = [], _detect_encoding(json_path)
    indicators, tainted, csv_type = [], {}, "Benign"
    if is_malicious and csv_path and os.path.exists(csv_path):
        # Compute TZ offset
        json_ts = None
        for evt in _iter_events(json_path, encoding):
            ts = _get_epoch(evt.get("header",{}).get("timestamp",""))
            if ts: json_ts = ts; break
        csv_ts = None
        try:
            dfc = pd.read_csv(csv_path, encoding='utf-8-sig')
            for col in ['TimeCreated','Time','Timestamp']:
                if col in dfc.columns:
                    for val in dfc[col].dropna():
                        ts = _get_epoch(str(val))
                        if ts: csv_ts = ts; break
                if csv_ts: break
        except: pass
        tz = (round((csv_ts - json_ts) / 3600) * 3600
              if json_ts and csv_ts else 0)
        indicators, tainted, csv_type = _extract_csv_signals(csv_path, tz)
    attack_type = forced_type if forced_type else csv_type
    print(f"     [i] Indicators={len(indicators)} AttackType={attack_type}")

    windows, pid_to_name = {}, {}
    for evt in _iter_events(json_path, encoding):
        header = evt.get("header", {})
        props  = evt.get("properties", {})
        ts = _get_epoch(header.get("timestamp",""))
        if ts == 0: continue
        ws  = ts - (ts % WINDOW_SIZE)
        pid = props.get("ProcessId") or header.get("process_id")
        if not pid: continue
        if "ImageFileName" in props:
            pid_to_name[pid] = props["ImageFileName"].lower()
        key = (pid, ws)
        if key not in windows:
            windows[key] = dict(pf_total=0, thread_total=0, image_total=0,
                                pf_normal=0, pf_suspicious=0, pf_large_region=0,
                                pf_cow=0, thread_external=0, thread_normal=0,
                                image_unsigned=0, image_signed=0,
                                image_suspicious_path=0)
        task   = str(header.get("task_name","")).lower()
        flags  = props.get("Flags", 0)
        siglvl = props.get("SignatureLevel", -1)
        tflags = props.get("ThreadFlags", 0)
        fname  = str(props.get("FileName","")).lower()
        opcode = header.get("event_opcode", 0)
        if "pagefault" in task:
            windows[key]["pf_total"] += 1
            if flags == 32768:   windows[key]["pf_suspicious"]   += 1
            elif flags == 1056768: windows[key]["pf_large_region"] += 1
            elif flags == 16384: windows[key]["pf_cow"]           += 1
            else:                windows[key]["pf_normal"]        += 1
        elif "thread" in task:
            windows[key]["thread_total"] += 1
            if opcode == 4 and tflags == 1: windows[key]["thread_external"] += 1
            else: windows[key]["thread_normal"] += 1
        elif "image" in task:
            windows[key]["image_total"] += 1
            if siglvl == 0: windows[key]["image_unsigned"] += 1
            else: windows[key]["image_signed"] += 1
            if fname.endswith(".dll") and all(
                p not in fname for p in ["system32","syswow64","winsxs","program files"]
            ): windows[key]["image_suspicious_path"] += 1

    for (pid, ws), m in windows.items():
        label = 0; is_tainted = 0; row_type = "Benign"
        has_inj=has_ct=has_unk=has_full=0
        has_vmw=has_vmr=has_vmo=has_ctr=has_ipr=acc_cnt=ct_depth=ct_unk=0
        proc = pid_to_name.get(pid, "unknown")
        if is_malicious:
            for ind in indicators:
                if (proc == ind['target'] or proc == ind['source']) \
                        and abs(ws - ind['time']) < 60:
                    label    = 1; row_type = attack_type
                    has_inj  = max(has_inj, ind['has_injection_pattern'])
                    has_ct   = max(has_ct,  ind['has_create_thread'])
                    has_unk  = max(has_unk, ind['has_unknown_trace'])
                    has_full = max(has_full,ind['has_full_access'])
                    has_vmw  = max(has_vmw, ind.get('Has_VM_Write',0))
                    has_vmr  = max(has_vmr, ind.get('Has_VM_Read',0))
                    has_vmo  = max(has_vmo, ind.get('Has_VM_Operation',0))
                    has_ctr  = max(has_ctr, ind.get('Has_CreateThread_Right',0))
                    has_ipr  = max(has_ipr, ind.get('Has_InjPattern_Right',0))
                    acc_cnt  = max(acc_cnt, ind.get('Access_Rights_Count',0))
                    ct_depth = max(ct_depth,ind.get('CallTrace_Depth',0))
                    ct_unk   = max(ct_unk,  ind.get('CallTrace_Unknown_Count',0))
                    break
            if proc in tainted:
                it = tainted[proc]
                if it <= ws <= it + TAINT_DURATION: is_tainted = 1
        pf  = m['pf_total']; thr = m['thread_total']
        features.append({
            'ProcessName': proc, 'WindowStart': ws,
            'PageFaultCount': pf, 'ThreadCount': thr,
            'ImageLoadCount': m['image_total'],
            'ThreadToPFRatio': round(thr/pf if pf>0 else 0, 6),
            'Is_Tainted_Past': is_tainted,
            'PF_Normal': m['pf_normal'], 'PF_Suspicious': m['pf_suspicious'],
            'PF_LargeRegion': m['pf_large_region'], 'PF_CopyOnWrite': m['pf_cow'],
            'Thread_External': m['thread_external'],
            'Image_Unsigned': m['image_unsigned'],
            'Image_SuspiciousPath': m['image_suspicious_path'],
            'Has_InjectionPattern': has_inj, 'Has_CreateThread': has_ct,
            'Has_UnknownCallTrace': has_unk, 'Has_FullAccess': has_full,
            'Has_VM_Write': has_vmw, 'Has_VM_Read': has_vmr,
            'Has_VM_Operation': has_vmo, 'Has_CreateThread_Right': has_ctr,
            'Has_InjPattern_Right': has_ipr, 'Access_Rights_Count': acc_cnt,
            'CallTrace_Depth': ct_depth, 'CallTrace_Unknown_Count': ct_unk,
            'AttackType': row_type, 'Label': label,
        })
    return features

def stage1_build_dataset():
    _header("STAGE 1: Building Dataset from Raw Logs")
    all_rows = []

    # Benign
    print("\nProcessing Benign Data...")
    benign_dir = os.path.join(DATA_ROOT, "bengin")
    for fname in sorted(os.listdir(benign_dir)):
        if not fname.endswith(".json"): continue
        jp = os.path.join(benign_dir, fname)
        cp = jp.replace(".json", ".csv")
        all_rows.extend(_process_pair(jp, cp if os.path.exists(cp) else None,
                                      is_malicious=False))

    # Malicious
    print("\nProcessing Malicious Data...")
    for root, _, files in os.walk(os.path.join(DATA_ROOT, "Malecious")):
        fn = os.path.basename(root).lower()
        fp = root.lower()
        forced = None; best = 0
        for key, label in ATTACK_TYPE_FOLDERS.items():
            if (key in fn or key in fp) and len(key) > best:
                forced = label; best = len(key)
        for f in files:
            if not f.endswith(".json"): continue
            jp = os.path.join(root, f)
            cp = jp.replace(".json", ".csv")
            if os.path.getsize(jp) == 0 or not os.path.exists(cp): continue
            all_rows.extend(_process_pair(jp, cp, is_malicious=True,
                                          forced_type=forced))

    df = pd.DataFrame(all_rows).drop_duplicates()
    df.to_csv(DATASET_PATH, index=False)
    print(f"\nDataset saved: {DATASET_PATH}")
    print(f"  Rows: {len(df):,}  |  Benign: {(df['Label']==0).sum():,}"
          f"  |  Malicious: {(df['Label']==1).sum():,}")
    return df

# ─────────────────────────────────────────────────────────────────
# STAGE 2 — BINARY DETECTION
# ─────────────────────────────────────────────────────────────────
def stage2_binary_detection(df):
    _header("STAGE 2: Binary Detection — V1 vs V2 vs V4 Feature Sets")

    y = df['Label'].values
    all_results = []

    for feat_name, feats in [('V1 (5 features)',  V1_FEATURES),
                              ('V2 (16 features)', V2_FEATURES),
                              ('V4 (22 features)', V4_FEATURES)]:
        print(f"\n  Feature set: {feat_name}")
        X_tr, X_te, y_tr, y_te = _get_split(df, feats)

        # Scale for LR and MLP
        scaler = StandardScaler()
        X_tr_sc = scaler.fit_transform(X_tr)

        ratio = (y_tr == 0).sum() / (y_tr == 1).sum()
        rows  = []

        # Logistic Regression
        t0 = time.time()
        lr = LogisticRegression(max_iter=1000, random_state=RANDOM_SEED, C=1.0)
        lr.fit(X_tr_sc, y_tr)
        rows.append(_evaluate('Logistic Regression', lr, X_te, y_te,
                              time.time()-t0, scaler))

        # Random Forest
        t0 = time.time()
        rf = RandomForestClassifier(n_estimators=200, random_state=RANDOM_SEED,
                                    n_jobs=-1)
        rf.fit(X_tr, y_tr)
        rows.append(_evaluate('Random Forest', rf, X_te, y_te, time.time()-t0))

        # XGBoost — scale_pos_weight for imbalance (no SMOTE)
        t0 = time.time()
        xgb = XGBClassifier(n_estimators=200, random_state=RANDOM_SEED,
                             eval_metric='logloss', verbosity=0,
                             scale_pos_weight=ratio)
        xgb.fit(X_tr, y_tr)
        rows.append(_evaluate('XGBoost', xgb, X_te, y_te, time.time()-t0))

        # Neural Network
        t0 = time.time()
        mlp = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=200,
                            random_state=RANDOM_SEED, early_stopping=True,
                            validation_fraction=0.1)
        mlp.fit(X_tr_sc, y_tr)
        rows.append(_evaluate('Neural Network', mlp, X_te, y_te,
                              time.time()-t0, scaler))

        _print_table(rows, feat_name)

        # Tag with feature set and store
        for r in rows:
            all_results.append({
                'FeatureSet': feat_name, 'Model': r['Model'],
                'F1': r['F1'], 'Precision': r['Precision'],
                'Recall': r['Recall'], 'AUC': r['AUC'],
                'FPR': r['FPR'], 'Train_s': r['Train_s'],
            })

    # ROC curve — V4 models only
    print("\n  Generating ROC curves (V4)...")
    X_tr, X_te, y_tr, y_te = _get_split(df, V4_FEATURES)
    scaler = StandardScaler()
    X_tr_sc = scaler.fit_transform(X_tr)
    ratio   = (y_tr==0).sum()/(y_tr==1).sum()

    roc_models = [
        ('Logistic Regression', LogisticRegression(max_iter=1000, C=1.0,
                                random_state=RANDOM_SEED), True),
        ('Random Forest',       RandomForestClassifier(n_estimators=200,
                                random_state=RANDOM_SEED, n_jobs=-1), False),
        ('XGBoost',             XGBClassifier(n_estimators=200,
                                random_state=RANDOM_SEED, eval_metric='logloss',
                                verbosity=0, scale_pos_weight=ratio), False),
        ('Neural Network',      MLPClassifier(hidden_layer_sizes=(64,32),
                                max_iter=200, random_state=RANDOM_SEED,
                                early_stopping=True), True),
    ]
    fig, ax = plt.subplots(figsize=(7, 6))
    colors = ['#e74c3c','#2ecc71','#3498db','#f39c12']
    for (mname, model, sc), color in zip(roc_models, colors):
        if sc: model.fit(X_tr_sc, y_tr); yp = model.predict_proba(X_te_sc := scaler.transform(X_te))[:,1]
        else:  model.fit(X_tr, y_tr);   yp = model.predict_proba(X_te)[:,1]
        fpr_c, tpr_c, _ = roc_curve(y_te, yp)
        auc = roc_auc_score(y_te, yp)
        ax.plot(fpr_c, tpr_c, color=color, lw=2,
                label=f"{mname} (AUC={auc:.3f})")
    ax.plot([0,1],[0,1],'k--',lw=1,label='Random')
    ax.set_xlabel('False Positive Rate'); ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curves — Binary DLL Injection Detection (V4 Features)')
    ax.legend(loc='lower right'); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/roc_curves_v4.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: roc_curves_v4.png")

    # Feature engineering comparison bar chart
    v4_rows = [r for r in all_results if r['FeatureSet']=='V4 (22 features)']
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    feat_sets  = ['V1 (5 features)', 'V2 (16 features)', 'V4 (22 features)']
    models_ord = ['Logistic Regression','Random Forest','XGBoost','Neural Network']
    colors_f   = ['#95a5a6','#3498db','#e74c3c']
    x = np.arange(len(models_ord)); w = 0.25

    for ax_idx, metric in enumerate(['F1', 'AUC']):
        for i, fs in enumerate(feat_sets):
            vals = [next((r[metric] for r in all_results
                          if r['FeatureSet']==fs and r['Model']==m), 0)
                    for m in models_ord]
            axes[ax_idx].bar(x + i*w, vals, w, label=fs,
                             color=colors_f[i], alpha=0.85)
        axes[ax_idx].set_title(f'{metric} Score by Feature Set')
        axes[ax_idx].set_ylabel(metric)
        axes[ax_idx].set_xticks(x + w)
        axes[ax_idx].set_xticklabels(['LR','RF','XGB','MLP'], fontsize=9)
        axes[ax_idx].legend(fontsize=8)
        axes[ax_idx].grid(True, alpha=0.3, axis='y')
        axes[ax_idx].set_ylim(0, 1.0)

    plt.suptitle('Feature Engineering Progression: V1 vs V2 vs V4',
                 fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/feature_engineering_comparison.png',
                dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: feature_engineering_comparison.png")

    pd.DataFrame(all_results).to_csv(f'{OUTPUT_DIR}/binary_results.csv',
                                      index=False)
    print(f"  Saved: binary_results.csv")
    return all_results

# ─────────────────────────────────────────────────────────────────
# STAGE 3 — MULTICLASS ATTRIBUTION
# ─────────────────────────────────────────────────────────────────
def stage3_multiclass(df):
    _header("STAGE 3: Multiclass Technique Attribution (V4 Features)")

    le       = LabelEncoder()
    y        = le.fit_transform(df['AttackType'])
    classes  = le.classes_
    X_tr, X_te, y_tr, y_te = _get_split(df, V4_FEATURES, label_col='AttackType')
    y_tr = le.transform(y_tr); y_te = le.transform(y_te)

    mc_results = {}

    for mname, model in [
        ('Random Forest', RandomForestClassifier(n_estimators=200,
                          random_state=RANDOM_SEED, n_jobs=-1)),
        ('XGBoost',       XGBClassifier(n_estimators=200,
                          random_state=RANDOM_SEED, eval_metric='mlogloss',
                          verbosity=0)),
    ]:
        print(f"\n  Training {mname}...")
        model.fit(X_tr, y_tr)
        y_pred = model.predict(X_te)
        macro  = f1_score(y_te, y_pred, average='macro')
        print(f"  Macro F1 = {macro:.4f}")
        print(classification_report(y_te, y_pred,
                                    target_names=classes, digits=4))
        mc_results[mname] = y_pred

    # Confusion matrix heatmap
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    for ax, (mname, y_pred) in zip(axes, mc_results.items()):
        cm      = confusion_matrix(y_te, y_pred)
        cm_norm = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-9)
        sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='YlOrRd',
                    xticklabels=classes, yticklabels=classes,
                    ax=ax, vmin=0, vmax=1)
        ax.set_title(f'{mname} — Multiclass Confusion Matrix\n'
                     f'(Macro F1={f1_score(y_te,y_pred,average="macro"):.4f})')
        ax.set_ylabel('True Attack Type')
        ax.set_xlabel('Predicted Attack Type')
        ax.tick_params(axis='x', rotation=30)
    plt.suptitle('Technique Attribution — V4 Features', fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/multiclass_confusion_v4.png',
                dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n  Saved: multiclass_confusion_v4.png")

    # Per-class F1 bar chart
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(classes)); w = 0.35
    for i, (mname, y_pred) in enumerate(mc_results.items()):
        f1s = f1_score(y_te, y_pred, average=None)
        bars = ax.bar(x + i*w - w/2, f1s, w,
                      label=mname, alpha=0.85,
                      color=['#2ecc71','#3498db'][i])
        for bar in bars:
            ax.text(bar.get_x()+bar.get_width()/2., bar.get_height()+0.01,
                    f'{bar.get_height():.2f}', ha='center', va='bottom',
                    fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels(classes, rotation=20)
    ax.set_ylabel('F1 Score')
    ax.set_title('Per-Technique F1 Score — Multiclass Attribution (V4)')
    ax.legend(); ax.set_ylim(0, 1.1); ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/multiclass_per_class_f1_v4.png',
                dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: multiclass_per_class_f1_v4.png")

    # Save reports
    rf_pred = mc_results['Random Forest']
    pd.DataFrame(classification_report(y_te, rf_pred, target_names=classes,
                                       output_dict=True)).T.to_csv(
        f'{OUTPUT_DIR}/multiclass_rf_v4_report.csv')
    print(f"  Saved: multiclass_rf_v4_report.csv")
    return mc_results, y_te, classes

# ─────────────────────────────────────────────────────────────────
# STAGE 4 — SHAP EXPLAINABILITY
# ─────────────────────────────────────────────────────────────────
def stage4_shap(df):
    _header("STAGE 4: SHAP Explainability (V4 Features)")

    X_tr, X_te, y_tr, y_te = _get_split(df, V4_FEATURES)
    rf = RandomForestClassifier(n_estimators=100, random_state=RANDOM_SEED,
                                n_jobs=-1)
    rf.fit(X_tr, y_tr)

    print("\n  Computing global SHAP values (500 samples)...")
    explainer  = shap.TreeExplainer(rf)
    X_sample   = X_te[:500]
    shap_mal   = _safe_shap(explainer, X_sample, len(V4_FEATURES))
    shap_mean  = np.abs(shap_mal).mean(axis=0)
    shap_norm  = shap_mean / (shap_mean.sum() + 1e-9)

    shap_df = pd.DataFrame({
        'Feature':  list(V4_FEATURES),
        'SHAP_V4':  [float(v) for v in shap_norm],
        'RF_Gini':  [float(v) for v in rf.feature_importances_],
    }).sort_values('SHAP_V4', ascending=False).reset_index(drop=True)

    print("\n  Global SHAP — Top 10 Features:")
    print(f"  {'Feature':<28} {'SHAP':>8} {'Gini':>8} {'Agreement':>10}")
    print(f"  {'-'*58}")
    for _, row in shap_df.head(10).iterrows():
        agree = "HIGH" if abs(row['SHAP_V4'] - row['RF_Gini']) < 0.05 else "LOW"
        bar   = '#' * int(row['SHAP_V4'] * 300)
        print(f"  {row['Feature']:<28} {row['SHAP_V4']*100:>6.1f}%  "
              f"{row['RF_Gini']*100:>6.1f}%  {agree:>10}  {bar}")

    # SHAP vs Gini comparison plot
    top10 = shap_df.head(10)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    x = np.arange(len(top10)); w = 0.35
    axes[0].bar(x-w/2, top10['RF_Gini'], w, label='RF Gini',
                color='#3498db', alpha=0.85)
    axes[0].bar(x+w/2, top10['SHAP_V4'], w, label='SHAP',
                color='#e74c3c', alpha=0.85)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(top10['Feature'], rotation=40, ha='right', fontsize=8)
    axes[0].set_title('Gini vs SHAP Feature Importance\n(Top 10 Features)')
    axes[0].set_ylabel('Normalised Importance')
    axes[0].legend(); axes[0].grid(True, alpha=0.3, axis='y')

    # Per-technique SHAP
    idx_all = np.arange(len(df))
    _, idx_te = train_test_split(idx_all, test_size=TEST_SIZE,
                                  random_state=RANDOM_SEED,
                                  stratify=df['Label'].values)
    df_test = df.iloc[idx_te].copy()

    attack_types  = ['Classic_CRT','Classic_Hook','Reflective',
                     'Sideloading','Mixed']
    per_tech_shap = {}
    print("\n  Per-technique SHAP:")
    for atype in attack_types:
        mask   = (df_test['AttackType'] == atype).values
        X_type = df_test[mask][V4_FEATURES].values
        if len(X_type) == 0: continue
        sample  = X_type[:min(100, len(X_type))]
        sv_arr  = _safe_shap(explainer, sample, len(V4_FEATURES))
        mean_sv = np.abs(sv_arr).mean(axis=0).flatten()
        norm_sv = mean_sv / (mean_sv.sum() + 1e-9)
        per_tech_shap[atype] = norm_sv
        top3 = norm_sv.argsort()[-3:][::-1]
        print(f"\n  {atype} — Top 3:")
        for i in top3:
            print(f"    {V4_FEATURES[i]:<28} {norm_sv[i]*100:>5.1f}%")

    # Per-technique heatmap
    atypes_found = [a for a in attack_types if a in per_tech_shap]
    shap_mat     = np.array([per_tech_shap[a] for a in atypes_found])
    top10_pos    = [V4_FEATURES.index(f) for f in top10['Feature'].tolist()
                    if f in V4_FEATURES]
    shap_mat_top = shap_mat[:, top10_pos]

    sns.heatmap(shap_mat_top,
                xticklabels=top10['Feature'].tolist(),
                yticklabels=atypes_found,
                cmap='YlOrRd', annot=True, fmt='.2f',
                ax=axes[1], vmin=0, vmax=0.5)
    axes[1].set_title('Per-Technique SHAP Importance\n(Top 10 Features)')
    axes[1].set_xlabel('Feature')
    axes[1].set_ylabel('Attack Type')
    axes[1].tick_params(axis='x', rotation=40)

    plt.suptitle('SHAP Explainability Analysis — V4 Features', fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/shap_analysis_v4.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n  Saved: shap_analysis_v4.png")

    shap_df.to_csv(f'{OUTPUT_DIR}/shap_vs_gini_v4.csv', index=False)
    print(f"  Saved: shap_vs_gini_v4.csv")
    return shap_df

# ─────────────────────────────────────────────────────────────────
# STAGE 5 — RULE-BASED BASELINE
# ─────────────────────────────────────────────────────────────────
def _apply_rules_binary(df_in):
    r1 = df_in['Thread_External'] > 0
    r2 = df_in['Has_InjectionPattern'] == 1
    r3 = df_in['Has_FullAccess'] == 1
    r4 = (df_in['Image_Unsigned']==1) & (df_in['Image_SuspiciousPath']==1)
    r5 = (df_in['Has_CreateThread']==1) & (df_in['PF_CopyOnWrite']>0)
    r6 = (df_in['PF_Suspicious']>0) & (df_in['Thread_External']>0)
    r7 = (df_in['Thread_External']>0) & (df_in['PF_CopyOnWrite']>0)
    preds = (r1|r2|r3|r4|r5|r6|r7).astype(int)
    return preds, {'R1_RemoteThread': r1.sum(), 'R2_InjectionPattern': r2.sum(),
                   'R3_FullAccess': r3.sum(),    'R4_Sideloading': r4.sum(),
                   'R5_ThreadCOW': r5.sum(),      'R6_SuspPFThread': r6.sum(),
                   'R7_ThreadCOW2': r7.sum()}

def _confidence_score(df_in):
    weights = {'Thread_External':0.30,'PF_CopyOnWrite':0.25,
               'Has_InjectionPattern':0.20,'Has_FullAccess':0.10,
               'Image_Unsigned':0.05,'Image_SuspiciousPath':0.05,
               'Has_CreateThread':0.03,'PF_Suspicious':0.02}
    scores = np.zeros(len(df_in))
    for i, (_, row) in enumerate(df_in.iterrows()):
        s = sum(w for feat, w in weights.items()
                if (row.get(feat,0) > 0 if feat != 'Has_FullAccess'
                    else row.get(feat,0) == 1))
        scores[i] = min(s, 1.0)
    return scores

def stage5_rule_based(df):
    _header("STAGE 5: Rule-Based Baseline (MITRE ATT&CK T1055)")

    idx_all = np.arange(len(df))
    _, idx_te = train_test_split(idx_all, test_size=TEST_SIZE,
                                  random_state=RANDOM_SEED,
                                  stratify=df['Label'].values)
    df_test = df.iloc[idx_te].copy()
    y_test  = df_test['Label'].values

    y_pred, rule_counts = _apply_rules_binary(df_test)
    y_scores = _confidence_score(df_test)

    f1_rb   = f1_score(y_test, y_pred)
    prec_rb = precision_score(y_test, y_pred)
    rec_rb  = recall_score(y_test, y_pred)
    auc_rb  = roc_auc_score(y_test, y_scores)
    fpr_rb  = ((y_pred==1)&(y_test==0)).sum()/(y_test==0).sum()

    print(f"\n  Binary Detection Results:")
    print(f"  F1={f1_rb:.4f}  Prec={prec_rb:.4f}  "
          f"Rec={rec_rb:.4f}  AUC={auc_rb:.4f}  FPR={fpr_rb:.4f}")

    print(f"\n  Rule firing counts:")
    for rule, count in rule_counts.items():
        print(f"  {rule:<25} {count:>6,}  ({count/len(df_test)*100:.1f}%)")

    # Per attack type breakdown
    df_test['y_pred_rules'] = y_pred
    print(f"\n  {'Attack Type':<16} {'Total':>6} {'Detected':>9} {'Rate':>7}")
    print(f"  {'-'*45}")
    benign = df_test[df_test['AttackType']=='Benign']
    fp = (benign['y_pred_rules']==1).sum()
    print(f"  {'Benign (FPR)':<16} {len(benign):>6,} {fp:>9,} "
          f"{fp/len(benign)*100:>6.1f}%")
    for atype in ['Classic_CRT','Classic_Hook','Reflective','Sideloading','Mixed']:
        sub = df_test[(df_test['AttackType']==atype)&(df_test['Label']==1)]
        if len(sub)==0: continue
        det = (sub['y_pred_rules']==1).sum()
        print(f"  {atype:<16} {len(sub):>6,} {det:>9,} "
              f"{det/len(sub)*100:>6.1f}%")

    # Comparison plot
    ml_f1s = {}
    for feat_name, feats in [('V2', V2_FEATURES), ('V4', V4_FEATURES)]:
        X_tr, X_te, y_tr, y_te = _get_split(df, feats)
        scaler = StandardScaler()
        X_tr_sc = scaler.fit_transform(X_tr)
        lr = LogisticRegression(max_iter=1000, random_state=RANDOM_SEED)
        lr.fit(X_tr_sc, y_tr)
        ml_f1s[f'LR {feat_name}'] = f1_score(y_te, lr.predict(scaler.transform(X_te)))
        rf = RandomForestClassifier(n_estimators=200, random_state=RANDOM_SEED, n_jobs=-1)
        rf.fit(X_tr, y_tr)
        ml_f1s[f'RF {feat_name}'] = f1_score(y_te, rf.predict(X_te))

    fig, ax = plt.subplots(figsize=(10, 5))
    labels  = ['Rule-Based\n(Human)'] + list(ml_f1s.keys())
    values  = [f1_rb] + list(ml_f1s.values())
    colors  = ['#e74c3c'] + ['#95a5a6','#3498db','#2ecc71','#f39c12']
    bars = ax.bar(labels, values, color=colors, alpha=0.85, width=0.6)
    ax.set_ylabel('F1 Score')
    ax.set_title('Rule-Based vs ML Detection Performance')
    ax.set_ylim(0, 0.9)
    ax.grid(True, alpha=0.3, axis='y')
    ax.axhline(y=f1_rb, color='#e74c3c', linestyle='--', alpha=0.5, lw=1)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x()+bar.get_width()/2., bar.get_height()+0.01,
                f'{val:.3f}', ha='center', va='bottom',
                fontsize=9, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/rule_vs_ml_v4.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n  Saved: rule_vs_ml_v4.png")

    return {'F1': f1_rb, 'Precision': prec_rb, 'Recall': rec_rb,
            'AUC': auc_rb, 'FPR': fpr_rb}

# ─────────────────────────────────────────────────────────────────
# STAGE 6 — FINAL SUMMARY
# ─────────────────────────────────────────────────────────────────
def stage6_summary(binary_results, rule_results):
    _header("STAGE 6: Final Summary")

    print(f"\n  {'Approach':<30} {'F1':>7} {'Prec':>7} {'Rec':>7} "
          f"{'AUC':>7} {'FPR':>7}")
    print(f"  {'-'*65}")

    # Rule-based
    print(f"  {'Rule-Based (Human)':<30} {rule_results['F1']:>7.4f} "
          f"{rule_results['Precision']:>7.4f} {rule_results['Recall']:>7.4f} "
          f"{rule_results['AUC']:>7.4f} {rule_results['FPR']:>7.4f}")
    print(f"  {'-'*65}")

    # ML results — V4 only for summary
    v4_rows = [r for r in binary_results if r['FeatureSet']=='V4 (22 features)']
    for r in v4_rows:
        print(f"  {r['Model'] + ' V4':<30} {r['F1']:>7.4f} "
              f"{r['Precision']:>7.4f} {r['Recall']:>7.4f} "
              f"{r['AUC']:>7.4f} {r['FPR']:>7.4f}")

    # Feature engineering progression
    print(f"\n  Feature Engineering Progression (Random Forest):")
    print(f"  {'Feature Set':<20} {'F1':>7} {'AUC':>7}")
    print(f"  {'-'*36}")
    for fs in ['V1 (5 features)','V2 (16 features)','V4 (22 features)']:
        r = next((x for x in binary_results
                  if x['FeatureSet']==fs and x['Model']=='Random Forest'), None)
        if r:
            print(f"  {fs:<20} {r['F1']:>7.4f} {r['AUC']:>7.4f}")

    # Save master results CSV
    rows = [{'Approach': 'Rule-Based (Human)', **rule_results,
             'FeatureSet': 'N/A', 'Type': 'Rule-Based'}]
    for r in binary_results:
        rows.append({'Approach': f"{r['Model']} ({r['FeatureSet']})",
                     'F1': r['F1'], 'Precision': r['Precision'],
                     'Recall': r['Recall'], 'AUC': r['AUC'],
                     'FPR': r['FPR'], 'FeatureSet': r['FeatureSet'],
                     'Type': 'ML'})
    pd.DataFrame(rows).to_csv(f'{OUTPUT_DIR}/dissertation_results.csv',
                               index=False)

    print(f"\n  All outputs saved to: {OUTPUT_DIR}/")
    print(f"    binary_results.csv              — all binary detection numbers")
    print(f"    multiclass_rf_v4_report.csv     — per-class multiclass results")
    print(f"    shap_vs_gini_v4.csv             — SHAP vs Gini comparison")
    print(f"    rule_based_comparison.csv        — rule-based breakdown")
    print(f"    dissertation_results.csv         — master results table")
    print(f"    roc_curves_v4.png               — ROC curves")
    print(f"    feature_engineering_comparison.png — V1 vs V2 vs V4")
    print(f"    multiclass_confusion_v4.png      — multiclass heatmap")
    print(f"    multiclass_per_class_f1_v4.png  — per-class F1")
    print(f"    shap_analysis_v4.png            — SHAP + per-technique")
    print(f"    rule_vs_ml_v4.png               — rule-based vs ML")

# ─────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="DLL Injection Detection Pipeline — Falah Nizam (2228945)")
    parser.add_argument('--build-dataset', action='store_true',
                        help='Rebuild dataset from raw logs (requires VM data)')
    args = parser.parse_args()

    print("=" * 65)
    print(" DLL INJECTION DETECTION PIPELINE")
    print(" Falah Nizam — 2228945 — ARU FYP")
    print("=" * 65)

    # Load or build dataset
    if args.build_dataset:
        df = stage1_build_dataset()
    else:
        if not os.path.exists(DATASET_PATH):
            print(f"\nERROR: {DATASET_PATH} not found.")
            print("Run with --build-dataset to build from raw logs.")
            sys.exit(1)
        print(f"\nLoading dataset: {DATASET_PATH}")
        df = pd.read_csv(DATASET_PATH)
        print(f"  Rows: {len(df):,}  |  "
              f"Benign: {(df['Label']==0).sum():,}  |  "
              f"Malicious: {(df['Label']==1).sum():,}")

    # Run all stages
    t_start = time.time()
    binary_results              = stage2_binary_detection(df)
    mc_results, y_te, classes   = stage3_multiclass(df)
    shap_df                     = stage4_shap(df)
    rule_results                = stage5_rule_based(df)
    stage6_summary(binary_results, rule_results)

    elapsed = time.time() - t_start
    print(f"\n{'='*65}")
    print(f" PIPELINE COMPLETE — Total time: {elapsed/60:.1f} minutes")
    print(f"{'='*65}")
