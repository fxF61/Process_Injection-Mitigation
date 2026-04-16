# DLL Injection Detection — ML vs Human Analysis


A machine learning pipeline for detecting and attributing DLL injection attacks using native Windows ETW and Sysmon telemetry, with empirical comparison against MITRE ATT&CK rule-based detection.

---

## Key Results

| Approach | F1 | Precision | AUC | FPR |
|---|---|---|---|---|
| **Logistic Regression (V4)** | **0.732** | **0.985** | **0.863** | **0.05%** |
| Neural Network (V4) | 0.730 | 0.985 | 0.851 | 0.05% |
| Random Forest (V4) | 0.720 | 0.934 | 0.839 | 0.25% |
| XGBoost (V4) | 0.527 | 0.433 | 0.851 | 5.30% |
| Rule-Based (MITRE ATT&CK) | 0.286 | 0.363 | 0.587 | 2.50% |

**ML outperforms rule-based detection by 2.56× F1 and 50× lower false alarm rate.**

### Technique Attribution (Multiclass, Macro F1 = 0.570)

| Technique | F1 | Status |
|---|---|---|
| Benign | 0.987 | ✓ Solved |
| Sideloading | 0.914 | ✓ Solved |
| Mixed | 0.750 | ~ Largely solved |
| Reflective | 0.279 | ✗ ETW ceiling |
| Classic_Hook | 0.263 | ✗ ETW ceiling |
| Classic_CRT | 0.231 | ✗ ETW ceiling |

---

## Repository Structure

```
├── detection_pipeline.py       # Supervised pipeline — Stages 1-6
├── unsupervised_experiment.py  # Unsupervised pipeline — Stage 7
├── master_dataset_v4.csv       # Labelled dataset (51,736 observations)
├── requirements.txt            # Python dependencies
└── ml_results/                 # Pre-generated figures and CSVs
```
---

## Dataset

`master_dataset_v4.csv` — 51,736 five-second process observation windows collected from real malware execution in an isolated Windows 10 VM.

| Class | Count | % |
|---|---|---|
| Benign | 48,832 | 94.4% |
| Reflective | 861 | 1.7% |
| Classic_CRT | 760 | 1.5% |
| Classic_Hook | 717 | 1.4% |
| Sideloading | 413 | 0.8% |
| Mixed | 153 | 0.3% |

**22 features** derived from ETW kernel events and Sysmon EID 10 access mask data, including the novel `Is_Tainted_Past` temporal taint propagation feature.

---

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run supervised pipeline (Stages 1-6)
python detection_pipeline.py

# Run unsupervised pipeline (Stage 7)
python unsupervised_experiment.py

# Rebuild dataset from raw logs (requires VM log files)
python detection_pipeline.py --build-dataset
```

---

## Pipeline Stages

### detection_pipeline.py

| Stage | Description | Key Output |
|---|---|---|
| 1 | Dataset Builder | `master_dataset_v4.csv` |
| 2 | Binary Detection | `roc_curves_v4.png`, `binary_results.csv` |
| 3 | Multiclass Attribution | `multiclass_confusion_v4.png` |
| 4 | SHAP Explainability | `shap_analysis_v4.png` |
| 5 | Rule-Based Baseline | `rule_vs_ml_v4.png` |
| 6 | Final Summary | `dissertation_results.csv` |

### unsupervised_experiment.py

| Experiment | Description | Key Output |
|---|---|---|
| 1 | Isolation Forest | `unsupervised_anomaly_scores.png` |
| 2 | KMeans Clustering | `unsupervised_clustering.png` |
| 3 | PCA Visualisation | `unsupervised_pca.png` |
| 4 | LOTO Zero-Day Simulation | `unsupervised_loto_results.png` |

---

## Key Findings

**1 — ETW is sufficient for binary detection**  
AUC=0.863 and FPR=0.05% achievable with native Windows telemetry at no additional cost.

**2 — ETW cannot attribute Classic_CRT, Classic_Hook or Reflective injection**  
All three techniques share identical ETW event signatures. SHAP analysis confirms `Is_Tainted_Past` dominates all three at 63-76%, providing no technique-discriminating information. This is an architectural ceiling, not a modelling limitation.

**3 — ML decisively outperforms MITRE ATT&CK rules**  
2.56× higher F1, 50× lower FPR. Rules fail because they are stateless — they cannot propagate detection context across time windows.

**4 — SMOTE degrades performance at 16.8:1 imbalance**  
No correction: F1=0.640, FPR=0.34%. SMOTE: F1=0.578, FPR=2.90%.

**5 — Reflective injection is the most evasive zero-day technique**  
LOTO detection rate: 29.5%. Section mapping (NtMapViewOfSection) predicted near-zero.

---

## Attack Techniques Covered

| Technique | MITRE | Tools Used |
|---|---|---|
| Classic CreateRemoteThread | T1055.002 | Stephen Fewer's framework, Pinjectra |
| Classic Hook Injection | T1055.004 | PSInject |
| Reflective DLL Injection | T1055.001 | ReflectiveDLLInjection, Donut |
| DLL Sideloading | T1574.001 | Atomic Red Team |
| Mixed Multi-Stage | T1055 | Atomic Red Team |

---

## Requirements

- Python 3.9+
- 8 GB RAM recommended
- See `requirements.txt` for full dependencies

---

