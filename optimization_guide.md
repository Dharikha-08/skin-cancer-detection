# Optimized Multimodal Healthcare Classification (Skin Cancer)

This update addresses the performance gap where previous fusion attempts were less effective than an image-only model. We've introduced a **doubly-optimized fusion pipeline** specifically designed for high-imbalance datasets (~1:22 ratio).

### 1. Tabular Model Upgrade: From RF to XGBoost
The previous **RandomForest** model was underperforming. We've replaced it with **XGBoost** in `tabular_model.py`:
- **Imbalance Handling:** Uses `scale_pos_weight = imbalance_ratio`, forcing the tree-building process to focus on malignant samples.
- **Robust Parameters:** `n_estimators=200`, `learning_rate=0.05`, `max_depth=5`.
- **Robust Preprocessing:** Still handles missing values (median/mode) and categorical encoding.

### 2. Image Model Improvement: Real-world Imbalance
The loss function in `train_derm7pt.py` now uses **dynamic class weights** based on your specific data distribution:
- **Class Weights:** Set to `[1.0, imbalance_ratio]`, ensuring the training process prioritizes the malignant minority (1:20+).
- **Architecture:** EfficientNet-B2 remains the backbone for its efficiency, with class weights correctly tuned to prevent the model from simply predicting the majority class.

### 3. Dynamic Fusion Parameter Search
Instead of guessing weights (0.6 / 0.4), the system now performs a **Grid Search for Alpha**:
- We test values from `0.5` (equal weight) to `0.95` (image dominance).
- The best `alpha` is selected based on which one maximizes the **F1-Score**.

### 4. Threshold Calibration for Fusion
Standard 0.5 thresholds are often suboptimal in healthcare because the cost of missing a malignant case is high:
- We search thresholds from `0.05` to `0.95`.
- This ensures that if the fusion model identifies a suspicious lesion, we choose a threshold that maximizes **F1** and **Balanced Accuracy**.

### 5. Final Evaluation Logic
The script now generates a side-by-side comparison of **Accuracy, F1, Balanced Accuracy, and ROC-AUC** for:
- Image-Only
- Tabular-Only
- **Optimized Fusion** (which should now consistently outperform the individual models).

---

### How to Run:
1. Ensure `tabular_model.py` and `train_derm7pt.py` are in the same directory.
2. Execute the optimized training script:
   ```powershell
   python train_derm7pt.py
   ```
3. Check the console for the **Final Comparison Results** table and `optimized_fusion_cm.png` for confirmation of improvement.
