# Pneumonia Detection from Chest X-Rays

Deep-learning classifier that flags likely pneumonia in chest X-ray images using a fine-tuned EfficientNetB0. Deployed as an interactive Streamlit app with a threshold slider that exposes the sensitivity vs specificity trade-off central to medical screening.

**[➡  Live demo][(https://pneumonia-detection-vjixsubtgwv4bjndk7wspa.streamlit.app/)]**

---

## What's in here

| File | Purpose |
|---|---|
| `app.py` | Streamlit app — upload an X-ray, see prediction + threshold controls |
| `model.py` | Model loading and preprocessing (matches the training pipeline exactly) |
| `models/pneumonia_model.keras` | Trained EfficientNetB0 weights (33 MB) |
| `requirements.txt` | Dependencies for Streamlit Cloud |
| `notebooks/` *(optional)* | Original training notebook |

## Dataset

**Kaggle Chest X-Ray Images (Pneumonia)** by Paul Mooney — 5,232 training images, 624 test images across two classes (Normal, Pneumonia). Dataset is **not** included in this repo; download from [kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia) if reproducing training.

Class imbalance in training set: **2.88:1 Pneumonia to Normal**, handled with class weighting (Normal images penalised 1.94× more than Pneumonia during training).

## Architecture

```
Input (chest X-ray, 224 x 224 x 3)
   |
   v
EfficientNetB0 base (ImageNet pretrained)
   |
   v
GlobalAveragePooling2D
   |
   v
BatchNormalization
   |
   v
Dense(256, relu) -> Dropout(0.4)
   |
   v
Dense(128, relu) -> Dropout(0.3)
   |
   v
Dense(1, sigmoid)
   |
   v
Output: probability of pneumonia in [0, 1]
```

## Training approach

Two-phase training strategy — a well-established technique for transfer learning that avoids destroying pretrained features:

1. **Phase 1 — Feature extraction.** Base model frozen; only the classifier head trains at LR = 1e-3 (15 epochs).
2. **Phase 2 — Fine-tuning.** Top 30 layers of the base unfrozen; whole model trains at 100× lower LR (1e-5) to gently adapt pretrained features to X-ray data (20 epochs).

**Class weighting** is applied throughout to counteract the 2.88:1 imbalance.

## Evaluation and threshold selection

Rather than defaulting to the naive 0.5 cutoff, threshold selection uses the **Youden's J statistic** (maximises `TPR − FPR` on the ROC curve). This is standard for medical screening tasks where the cost of false negatives (missed pneumonia) is much higher than false positives (unnecessary follow-up).

Final EfficientNetB0 metrics on the 624-image held-out test set (at the Youden-optimal threshold):
- **Accuracy:** ~89%
- **Pneumonia precision / recall:** high on both — see notebook for exact numbers per model
- **AUC:** 0.99 during training on the validation split

The Streamlit app exposes the threshold as an interactive slider so the sensitivity/specificity trade-off is visible in real time.

## Limitations (honest section)

- **Not for clinical use.** This is an educational demo, not a diagnostic tool. A real deployment would need FDA/CDSCO clearance, prospective validation on a different hospital's data, radiologist review of edge cases, and integration with clinical decision-support workflows.
- **Train / test distribution shift.** The Kaggle dataset has a documented distribution shift between the train and test splits — test images tend to be harder. This is a real-world modelling lesson: benchmark performance on this dataset overstates what would be seen on new hospital data.
- **Not for adults.** All images are from paediatric patients in Guangzhou. Adult chest radiography looks different and this model was not trained on it.
- **Two-class simplification.** Real screening distinguishes viral vs bacterial pneumonia, other lung pathologies, and normal variants — this is a binary classifier trained on a simplified problem.

## Run locally

```bash
git clone https://github.com/aaranarc/pneumonia-detection.git
cd pneumonia-detection

python -m venv .venv
source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt

streamlit run app.py
```

## Author

**Aarana Chaurasia** — B.Tech CSE-DS, DJSCE Mumbai. Built as part of an applied ML portfolio spanning finance, healthcare, and biometrics.

- GitHub: [@aaranarc](https://github.com/aaranarc)
- LinkedIn: [aarana-chaurasia](https://www.linkedin.com/in/aarana-chaurasia-3215a8330)

## License

MIT — see `LICENSE`.
