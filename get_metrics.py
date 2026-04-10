import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import timm
from torchvision import transforms
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, balanced_accuracy_score, roc_auc_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from tabular_model import TabularModel

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def compute_metrics(y_true, y_prob, threshold=0.5):
    preds = (y_prob > threshold).astype(int)
    return {
        "Accuracy": accuracy_score(y_true, preds),
        "F1": f1_score(y_true, preds, zero_division=0),
        "Bal_Acc": balanced_accuracy_score(y_true, preds),
        "AUC": roc_auc_score(y_true, y_prob)
    }

# 1. Load Data
df = pd.read_csv("meta/meta.csv").dropna(subset=["diagnosis"])
malignant_classes = ["melanoma", "basal cell carcinoma"]
df["label"] = df["diagnosis"].apply(lambda x: 1 if x.lower() in malignant_classes else 0)

_, val_df = train_test_split(df, test_size=0.2, stratify=df["label"], random_state=42)

# 2. Setup Data Loader
val_tf = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

class MultimodalDataset(Dataset):
    def __init__(self, df, tf):
        self.df = df.reset_index(drop=True)
        self.tf = tf
    def __len__(self): return len(self.df)
    def __getitem__(self, i):
        img_file = self.df.loc[i, "derm"]
        image = Image.open(os.path.join("images", img_file)).convert("RGB")
        return self.tf(image), int(self.df.loc[i, "label"])

val_loader = DataLoader(MultimodalDataset(val_df, val_tf), batch_size=16)

# 3. Load Models
print("Loading trained models...")
model = timm.create_model("efficientnet_b2", pretrained=False, num_classes=2).to(DEVICE)
model.load_state_dict(torch.load("best_image_model_opt.pth", map_location=DEVICE))
model.eval()

tab_model = TabularModel() # Loads latest pkl automatically if exists in some setups, but we'll ensure it works

# 4. Run Evaluation
print("Evaluating multimodal pipeline...")
all_img_probs = []
y_true = []
with torch.no_grad():
    for x, y in val_loader:
        out = model(x.to(DEVICE))
        probs = torch.softmax(out, dim=1)[:, 1].cpu().numpy()
        all_img_probs.extend(probs)
        y_true.extend(y.numpy())

img_probs = np.array(all_img_probs)
y_true = np.array(y_true)
tab_probs = tab_model.predict_proba(val_df)

# Load saved fusion params
with open("fusion_params.txt", "r") as f:
    params = f.read().splitlines()
    alpha = float(params[0].split(": ")[1])
    threshold = float(params[1].split(": ")[1])

fused_probs = alpha * img_probs + (1 - alpha) * tab_probs

# 5. Print Results
res_img = compute_metrics(y_true, img_probs, threshold=0.5)
res_tab = compute_metrics(y_true, tab_probs, threshold=0.5)
res_fusion = compute_metrics(y_true, fused_probs, threshold=threshold)

comparison_df = pd.DataFrame({
    "Metric": ["Accuracy", "F1 Score", "Bal. Acc", "ROC-AUC"],
    "Image-Only": [res_img["Accuracy"], res_img["F1"], res_img["Bal_Acc"], res_img["AUC"]],
    "Tabular-Only": [res_tab["Accuracy"], res_tab["F1"], res_tab["Bal_Acc"], res_tab["AUC"]],
    "Optimized Fusion": [res_fusion["Accuracy"], res_fusion["F1"], res_fusion["Bal_Acc"], res_fusion["AUC"]]
})

print("\n" + "="*60)
print("FINAL PERFORMANCE METRICS")
print("="*60)
print(comparison_df.to_string(index=False))
print("="*60)
print(f"Fusion Configuration: Alpha={alpha}, Threshold={threshold}")

# Confusion Matrix
cm = confusion_matrix(y_true, (fused_probs > threshold).astype(int))
plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title(f"Confusion Matrix (F1: {res_fusion['F1']:.4f})")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.savefig("evaluation_results_cm.png")
print("Confusion Matrix saved to: evaluation_results_cm.png")
