# Multimodal Skin Cancer Classification Upgrade

This document summarizes the upgrades made to transition your image-only EfficientNet model into a multimodal system that incorporates clinical tabular data.

### 1. New Component: `tabular_model.py`
A modular Python script that handles the end-to-end processing of clinical metadata.
- **Preprocessing:** Handles missing values (median/most frequent imputer), encodes categorical variables (LabelEncoder), and normalizes numerical features (StandardScaler).
- **Model:** Uses a `RandomForestClassifier` with balanced class weights to handle dataset imbalance.
- **Modularity:** Can be imported and used independently for training and inference.

### 2. Updated Training: `train_derm7pt.py`
The main training script has been extended (not replaced) to support multimodal training and evaluation.
- **Multimodal Dataset:** The `MultimodalDataset` class now facilitates matching image samples with their corresponding tabular metadata.
- **Class Imbalance:** Added class weights `1:3` to the `nn.CrossEntropyLoss` for the image model, complementing the `WeightedRandomSampler`.
- **Late Fusion Logic:**
  - Predictions from the **EfficientNet-B2** image model and the **Random Forest** tabular model are gathered.
  - Final probability is computed as:
    `final_prob = 0.6 * image_prob + 0.4 * tabular_prob`

### 3. Comprehensive Evaluation
At the end of training, a comparison table is generated to show performance across three configurations:
1. **Image-Only:** EfficientNet model alone.
2. **Tabular-Only:** Random Forest model alone.
3. **Multimodal (Fusion):** Combined late fusion model.

**Metrics Computed:**
- Accuracy
- F1 Score
- Balanced Accuracy
- ROC-AUC Score

### How to Run
1. Ensure `tabular_model.py` and `train_derm7pt.py` are in the same directory (`release_v0`).
2. Run the modified training script:
   ```powershell
   python train_derm7pt.py
   ```
3. After execution, check the console for the **Performance Comparison Table** and the `fusion_cm.png` for the final confusion matrix.

---
**Note:** The clinical features used include: `pigment_network`, `streaks`, `pigmentation`, `regression_structures`, `dots_and_globules`, `blue_whitish_veil`, `vascular_structures`, `level_of_diagnostic_difficulty`, `elevation`, `location`, and `sex`.
