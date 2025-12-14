import pandas as pd
import joblib
import numpy as np
import sys
import os

# -------- HOW TO RUN ----------
# python src/predict2.py full_network_log.csv
# ------------------------------

# Get base directory (project root)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Load trained model + metadata
model = joblib.load(os.path.join(BASE_DIR, 'models', 'rf_cicids2017_model.pkl'))
metadata = joblib.load(os.path.join(BASE_DIR, 'models', 'rf_feature_metadata.pkl'))

selected_features = metadata["selected_features"]
selected_mask = metadata["selected_mask"]

# TAKE BASE FEATURES FROM MODEL METADATA
base_features_raw = metadata["base_features"]  # list saved at training time
# normalize names (strip spaces) but KEEP ORDER
base_features = [c.strip() for c in base_features_raw]

print("Loaded base feature count from metadata:", len(base_features))
print("Example base features:", base_features[:10])
print("Selected feature count:", len(selected_features))

if len(sys.argv) < 2:
    print("Usage: python predict.py <csv_file>")
    sys.exit(1)

csv_path = sys.argv[1]

# Load new data
df = pd.read_csv(csv_path)

# Normalize column names in the CSV (strip leading/trailing spaces)
df.columns = df.columns.str.strip()

# Check that all required base features exist in the CSV
missing = [col for col in base_features if col not in df.columns]
if missing:
    print("ERROR: The following required base features are missing from your CSV:")
    for col in missing:
        print(f"  - {col}")
    print("\nFirst few columns in your CSV are:")
    print(list(df.columns)[:30])
    sys.exit(1)

# Keep ONLY the base features (drops Label and any other extra columns)
df_base = df[base_features].copy()

# Clean infinities / NaNs
df_base = df_base.replace([np.inf, -np.inf], np.nan).dropna()

if df_base.empty:
    print("ERROR: After cleaning NaN/inf, no rows remain to predict on.")
    sys.exit(1)

# Convert to numpy and apply same feature-selection mask used at training
X_base = df_base.values              # shape: (n_samples, len(base_features))
X_sel = X_base[:, selected_mask]     # shape: (n_samples, n_selected_features)

# Predict
preds = model.predict(X_sel)
proba = model.predict_proba(X_sel)

# Optional: true label if present (for checking accuracy)
true_labels = None
for cand in ["Label", "label"]:
    if cand in df.columns:
        true_labels = df[cand]
        break

# Output predictions
for i in range(len(preds)):
    pred_label = "ATTACK" if preds[i] == 1 else "BENIGN"
    if true_labels is not None and i < len(true_labels):
        print(
            f"Row {i}: PRED={pred_label} | TRUE={true_labels.iloc[i]} "
            f"| Prob(BENIGN)={proba[i][0]:.4f}, Prob(ATTACK)={proba[i][1]:.4f}"
        )
    else:
        print(
            f"Row {i}: {pred_label}  |  Prob(BENIGN)={proba[i][0]:.4f}, "
            f"Prob(ATTACK)={proba[i][1]:.4f}"
        )
