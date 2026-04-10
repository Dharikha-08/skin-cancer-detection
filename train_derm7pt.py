import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import pickle

# Import TabularModel from modular file
from tabular_model import TabularModel

torch.manual_seed(42)
np.random.seed(42)

from torchvision import transforms
import timm

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, balanced_accuracy_score, confusion_matrix, roc_auc_score

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

############################################
# 1. LOAD DATA + IMBALANCE ANALYSIS
############################################
print("\n--- OPTIMIZED DATA ANALYSIS ---")

df = pd.read_csv("meta/meta.csv")
df = df.dropna(subset=["diagnosis"])

malignant_classes = ["melanoma", "basal cell carcinoma"]

def map_label(x):
    return 1 if x.lower() in malignant_classes else 0

df["label"] = df["diagnosis"].apply(map_label)

counts = df["label"].value_counts().sort_index()
imbalance_ratio = counts[0] / counts[1]
print(f"Benign (0): {counts[0]}")
print(f"Malignant (1): {counts[1]}")
print(f"Imbalance Ratio: {imbalance_ratio:.2f}")

############################################
# 2. SPLIT
############################################
train_df, val_df = train_test_split(
    df,
    test_size=0.2,
    stratify=df["label"],
    random_state=42
)

############################################
# 3. OPTIMIZED TABULAR MODEL (XGBOOST)
############################################
print("\n--- TRAINING OPTIMIZED TABULAR MODEL ---")
tab_model = TabularModel(imbalance_ratio=imbalance_ratio)
tab_model.train(train_df, train_df["label"].values)

############################################
# 4. IMAGE TRANSFORMS
############################################
train_tf = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(0.2,0.2,0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

val_tf = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

############################################
# 5. MULTIMODAL DATASET
############################################
class MultimodalDataset(Dataset):
    def __init__(self, df, tf):
        self.df = df.reset_index(drop=True)
        self.tf = tf

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        img_file = self.df.loc[i, "derm"]
        path = os.path.join("images", img_file)
        image = Image.open(path).convert("RGB")
        image = self.tf(image)
        label = int(self.df.loc[i, "label"])
        return image, label, i

train_ds = MultimodalDataset(train_df, train_tf)
val_ds   = MultimodalDataset(val_df, val_tf)

weights = 1.0 / train_df["label"].value_counts().sort_index()
sample_weights = train_df["label"].map(weights)
sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)

train_loader = DataLoader(train_ds, batch_size=16, sampler=sampler)
val_loader   = DataLoader(val_ds, batch_size=16)

############################################
# 6. OPTIMIZED IMAGE MODEL(S)
############################################
def create_image_model(model_name="efficientnet_b2"):
    print(f"Creating Image Model: {model_name}...")
    model = timm.create_model(model_name, pretrained=True, num_classes=2)
    return model.to(DEVICE)

# Primary Model
model_primary = create_image_model("efficientnet_b2")

# Optimized Loss: Strongly weighted for minority (1:20+)
# Fix: Explicitly set dtype to float32 to avoid "expected scalar type Float but found Double" error
class_weights = torch.tensor([1.0, imbalance_ratio], dtype=torch.float32).to(DEVICE)
criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = torch.optim.AdamW(model_primary.parameters(), lr=1e-4)

############################################
# 7. REUSABLE FUNCTIONS (Tuning, Evaluation)
############################################
def find_best_fusion_params(img_probs, tab_probs, y_true):
    """ Dynamically searches for optimal image weight (alpha) and threshold (T). """
    print("\n--- OPTIMIZING FUSION PARAMETERS ---")
    best_f1 = 0
    best_alpha = 0.6
    best_threshold = 0.5
    
    # Grid search for weight alpha (image dominance)
    for alpha in np.linspace(0.5, 0.95, 10):
        fused_probs = alpha * img_probs + (1 - alpha) * tab_probs
        
        # Grid search for threshold T
        for t in np.linspace(0.05, 0.95, 19):
            preds = (fused_probs > t).astype(int)
            score = f1_score(y_true, preds, zero_division=0)
            if score > best_f1:
                best_f1 = score
                best_alpha = alpha
                best_threshold = t
                
    print(f"Optimal Fusion: alpha={best_alpha:.2f}, threshold={best_threshold:.2f} (F1: {best_f1:.4f})")
    return best_alpha, best_threshold

def compute_metrics(y_true, y_prob, threshold=0.5):
    preds = (y_prob > threshold).astype(int)
    return {
        "Accuracy": accuracy_score(y_true, preds),
        "F1": f1_score(y_true, preds, zero_division=0),
        "Bal_Acc": balanced_accuracy_score(y_true, preds),
        "AUC": roc_auc_score(y_true, y_prob)
    }

############################################
# 8. TRAIN LOOP (Optimized)
############################################
EPOCHS = 20 # Increased for better convergence
best_f1 = 0
patience = 5
counter = 0

print(f"\nTraining Primary Image Model (EfficientNet)...")

for e in range(EPOCHS):
    model_primary.train()
    running_loss = 0
    for x, y, _ in tqdm(train_loader, desc=f"Epoch {e+1}/{EPOCHS}"):
        x, y = x.to(DEVICE), y.to(DEVICE)
        optimizer.zero_grad()
        loss = criterion(model_primary(x), y)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    # Quick eval for saving
    model_primary.eval()
    val_p = []
    val_l = []
    with torch.no_grad():
        for x, y, _ in val_loader:
            x = x.to(DEVICE)
            probs = torch.softmax(model_primary(x), dim=1)[:, 1].cpu().numpy()
            val_p.extend(probs)
            val_l.extend(y.numpy())
    
    cur_f1 = f1_score(val_l, (np.array(val_p) > 0.5).astype(int), zero_division=0)
    print(f"Loss: {running_loss/len(train_loader):.4f} | Val F1: {cur_f1:.4f}")

    if cur_f1 > best_f1:
        best_f1 = cur_f1
        torch.save(model_primary.state_dict(), "best_image_model_opt.pth")
        counter = 0
        print("✅ New Best Model Saved")
    else:
        counter += 1
        if counter >= patience:
            print("Early Stopping.")
            break

############################################
# 9. FINAL MULTIMODAL OPTIMIZATION
############################################
print("\n" + "="*50)
print("FINAL OPTIMIZED EVALUATION PHASE")
print("="*50)

# Load best image model
model_primary.load_state_dict(torch.load("best_image_model_opt.pth"))
model_primary.eval()

# Gather predictions
all_img_probs = []
y_val_true = []
with torch.no_grad():
    for x, y, _ in val_loader:
        out = model_primary(x.to(DEVICE))
        probs = torch.softmax(out, dim=1)[:, 1].cpu().numpy()
        all_img_probs.extend(probs)
        y_val_true.extend(y.numpy())

img_probs_arr = np.array(all_img_probs)
y_val_true_arr = np.array(y_val_true)
tab_probs_arr = tab_model.predict_proba(val_df)

# Optimal Multi-modal Tuning
alpha, best_t = find_best_fusion_params(img_probs_arr, tab_probs_arr, y_val_true_arr)
fused_probs = alpha * img_probs_arr + (1 - alpha) * tab_probs_arr

# Results Collection
res_img = compute_metrics(y_val_true_arr, img_probs_arr, threshold=0.5)
res_tab = compute_metrics(y_val_true_arr, tab_probs_arr, threshold=0.5)
res_fusion = compute_metrics(y_val_true_arr, fused_probs, threshold=best_t)

comparison_df = pd.DataFrame({
    "Metric": ["Accuracy", "F1 Score", "Bal. Acc", "ROC-AUC"],
    "Image-Only": [res_img["Accuracy"], res_img["F1"], res_img["Bal_Acc"], res_img["AUC"]],
    "Tabular-Only": [res_tab["Accuracy"], res_tab["F1"], res_tab["Bal_Acc"], res_tab["AUC"]],
    "Optimized Fusion": [res_fusion["Accuracy"], res_fusion["F1"], res_fusion["Bal_Acc"], res_fusion["AUC"]]
})

print("\nFinal Comparison Results:")
print(comparison_df.to_string(index=False))

# Confusion Matrix for Optimized Fusion
cm = confusion_matrix(y_val_true_arr, (fused_probs > best_t).astype(int))
plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Oranges')
plt.title(f"Optimized Fusion CM (Alpha={alpha:.2f}, T={best_t:.2f})")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.savefig("optimized_fusion_cm.png")
plt.close()

# Save parameters
with open("fusion_params.txt", "w") as f:
    f.write(f"alpha: {alpha}\nthreshold: {best_t}")

print("\n" + "="*50)
print("SUCCESS: Optimized multimodal pipeline completed.")
print(f"F1 Score Improved to: {res_fusion['F1']:.4f}")
print("Check optimized_fusion_cm.png for confusion matrix.")
print("="*50)