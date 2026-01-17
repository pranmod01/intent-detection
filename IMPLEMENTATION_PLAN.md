# Implementation Plan: Motor Intent Detection in Hybrid BCIs

## Project Overview

**Goal**: Build a multi-condition EEG classifier that distinguishes volitional motor intent from externally-evoked neural responses (EMS, passive movement).

**Current Assets**:
- 2 subjects with 6 conditions each (voluntary L/R, imagery, EMS short/long, passive x2)
- 4 EEG channels (C3, C4, Cz, CP3) at 200 Hz
- Existing preprocessing pipeline (`eeg_analysis.py`, `enhanced_analysis.py`)
- Cleaned data organized by session

---

## Phase 1: Problem Demonstration

### 1.1 Setup and PhysioNet Data Integration

**Files to Create**: `phase1/physionet_loader.py`

Tasks:
- [ ] Download PhysioNet Motor Movement/Imagery Dataset (EEGMMIDB)
- [ ] Create data loader that:
  - Selects 20 subjects (S001-S020)
  - Filters to C3, C4, Cz channels (CP3 not standard in PhysioNet)
  - Extracts "motor imagery" (T1, T2 tasks) and "rest" conditions
  - Resamples to 200 Hz (matches your data)
  - Applies same preprocessing (bandpass 1-50 Hz, notch 60 Hz)
  - Creates 2-second epochs

**Key Code Components**:
```python
# Use MNE's built-in PhysioNet fetcher
from mne.datasets import eegbci
from mne.io import concatenate_raws, read_raw_edf

# Download runs: 4,8,12 (imagery hands), 1 (baseline rest)
raw_fnames = eegbci.load_data(subject, runs=[1, 4, 8, 12])
```

### 1.2 Train Standard MI-BCI (CSP + LDA)

**Files to Create**: `phase1/csp_lda_baseline.py`

Tasks:
- [ ] Implement CSP feature extraction
  - Use `mne.decoding.CSP` with n_components=6
  - Filter to 8-30 Hz (mu + beta bands)
- [ ] Train LDA classifier on PhysioNet data
  - Binary: motor imagery vs. rest
  - 5-fold cross-validation within PhysioNet
  - Expected accuracy: 70-80%
- [ ] Save trained model (CSP filters + LDA weights)

**Key Code Components**:
```python
from mne.decoding import CSP
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.pipeline import Pipeline

clf = Pipeline([
    ('csp', CSP(n_components=6, reg=None, log=True)),
    ('lda', LinearDiscriminantAnalysis())
])
```

### 1.3 Test on Your Intent Conditions

**Files to Create**: `phase1/cross_dataset_evaluation.py`

Tasks:
- [ ] Prepare your intent data (voluntary L/R + imagery)
  - Combine into "intent" class
  - Match channel subset (C3, C4, Cz - exclude CP3 for PhysioNet compatibility)
  - Same 2-second epochs
- [ ] Apply PhysioNet-trained classifier
- [ ] Record accuracy, sensitivity, specificity
- [ ] Expected: ~60-70% accuracy (reasonable generalization)

### 1.4 Test on Your Non-Intent Conditions (Critical Test)

**File**: Same as 1.3, extended

Tasks:
- [ ] Prepare non-intent data (EMS + passive conditions)
- [ ] Apply PhysioNet-trained classifier
- [ ] Record FALSE POSITIVE RATE
  - How often EMS is classified as "intent"
  - How often passive is classified as "intent"
- [ ] Expected: 40-60% specificity (HIGH false positives)

### 1.5 Deliverable: Figure 1 - The Intent Detection Problem

**Files to Create**: `phase1/generate_figures.py`

Tasks:
- [ ] Create bar plot showing:
  - PhysioNet test accuracy: ~75%
  - Your intent conditions accuracy: ~65%
  - Your EMS specificity: ~45% (problem!)
  - Your passive specificity: ~55% (problem!)
- [ ] Add error bars, significance markers
- [ ] Save as high-resolution PNG for paper

---

## Phase 2: Multi-Condition Training Solution

### 2.1 Data Organization for Classification

**Files to Create**: `phase2/data_organizer.py`

Tasks:
- [ ] Create unified data structure:
  ```
  Intent class:
    - voluntary_left (all epochs, both subjects)
    - voluntary_right (all epochs, both subjects)
    - motor_imagery (all epochs, both subjects)

  Non-Intent class:
    - ems_short (all epochs, both subjects)
    - ems_long (all epochs, both subjects)
    - passive_1 (all epochs, both subjects)
    - passive_2 (all epochs, both subjects)
  ```
- [ ] Balance classes if needed (undersampling/oversampling)
- [ ] Create train/test splits (leave-one-subject-out)

### 2.2 CSP + LDA on Multi-Condition Data

**Files to Create**: `phase2/multicond_csp_lda.py`

Tasks:
- [ ] Train CSP + LDA on your 4-condition data
  - Binary: intent vs. non-intent
  - CSP: n_components=4 (reduced due to small N)
  - Frequency band: 8-30 Hz
- [ ] Cross-validation: Leave-one-subject-out (2-fold)
- [ ] Metrics:
  - Accuracy
  - Sensitivity (detecting intent)
  - Specificity (rejecting non-intent)
  - F1 score
- [ ] Expected: 75-85% accuracy

### 2.3 EEGNet Deep Learning Classifier

**Files to Create**: `phase2/eegnet_classifier.py`

Tasks:
- [ ] Implement EEGNet architecture (Lawhern et al. 2018):
  ```python
  # Input shape: (batch, 1, 4 channels, 400 timepoints)
  # Temporal conv: 8 filters, kernel (1, 64)
  # Depthwise spatial: 2x filters per channel
  # Separable conv: 16 filters
  # Output: softmax(2) for intent/non-intent
  ```
- [ ] Training setup:
  - Optimizer: Adam, lr=0.001
  - Epochs: 100 with early stopping (patience=10)
  - Batch size: 16-32
- [ ] Data augmentation (critical for small N):
  - Gaussian noise injection
  - Time jittering (small shifts)
  - Optional: mixup
- [ ] Cross-validation: Leave-one-subject-out
- [ ] Expected: 78-88% accuracy

**Dependencies to Add**:
```
torch or tensorflow
braindecode (optional, has EEGNet implementation)
```

### 2.4 Detailed Condition Analysis

**Files to Create**: `phase2/confusion_analysis.py`

Tasks:
- [ ] Train 4-class classifier (vol, img, ems, passive)
- [ ] Generate 4x4 confusion matrix
- [ ] Analyze:
  - Which conditions are most confusable?
  - Does EMS look more like passive or imagery?
  - Does imagery look more like voluntary or rest?
- [ ] Create confusion matrix heatmap visualization

### 2.5 Channel-Specific Analysis

**Files to Create**: `phase2/spatial_analysis.py`

Tasks:
- [ ] Extract and visualize CSP filters
- [ ] Plot topographic maps showing discriminative patterns
- [ ] Analyze which channels distinguish intent best
- [ ] Compare EMS vs. voluntary topographies

### 2.6 Deliverable: Figure 2 & Table 1

**Files to Create**: `phase2/generate_figures.py`

Tasks:
- [ ] Figure 2 Panel A: Comparison plot
  - CSP+LDA (PhysioNet): 51% specificity
  - CSP+LDA (your data): 82% specificity
  - EEGNet (your data): 87% specificity
- [ ] Figure 2 Panel B: Confusion matrix heatmap
- [ ] Figure 2 Panel C: CSP spatial patterns
- [ ] Table 1: Full metrics comparison

---

## Phase 3: Transfer Learning Enhancement

### 3.1 Feature-Level Transfer with Autoencoder

**Files to Create**: `phase3/autoencoder_transfer.py`

Tasks:
- [ ] Pre-train convolutional autoencoder on PhysioNet:
  - Encoder: EEG epochs → latent space (128-256 dims)
  - Decoder: reconstruct EEG
  - Loss: MSE reconstruction
- [ ] Transfer to your data:
  - Freeze encoder
  - Add classification head
  - Fine-tune on your labeled data
- [ ] Compare: from-scratch vs. transfer learning accuracy

### 3.2 Data Augmentation from Public Data

**Files to Create**: `phase3/augmented_training.py`

Tasks:
- [ ] Create augmented training set:
  - Your intent data: 100%
  - PhysioNet imagery: supplement as additional "intent" examples
  - Your non-intent data: 100%
- [ ] Train EEGNet on augmented set
- [ ] Measure generalization improvement

### 3.3 Domain Adaptation Analysis

**Files to Create**: `phase3/domain_analysis.py`

Tasks:
- [ ] Extract features from both datasets (EEGNet embeddings or CSP features)
- [ ] Plot t-SNE/UMAP visualization:
  - Show PhysioNet imagery ≈ your imagery/voluntary (overlapping)
  - Show your EMS/passive = unique region
- [ ] Quantify domain gap

### 3.4 Deliverable: Figure 3

Tasks:
- [ ] Panel A: Learning curves (from-scratch vs. transfer)
- [ ] Panel B: t-SNE visualization
- [ ] Panel C: Sample efficiency curves

---

## Phase 4: Real-Time Demo

### 4.1 Model Optimization

**Files to Create**: `phase4/model_optimization.py`

Tasks:
- [ ] Quantize EEGNet (float32 → int8 if using PyTorch/ONNX)
- [ ] Benchmark inference time:
  - Target: <50ms per epoch
  - Test on laptop CPU
- [ ] Create streaming simulation:
  - Sliding window (50% overlap)
  - Real-time predictions

### 4.2 Interactive Demo App

**Files to Create**: `phase4/demo_app.py`

Tasks:
- [ ] Build Streamlit/Gradio app with:
  - File upload for EEG data
  - Real-time EEG plot
  - Intent/Non-Intent prediction display
  - Confidence score visualization
  - Color coding: Green (intent), Red (non-intent), Gray (uncertain)
- [ ] Load pre-trained model
- [ ] Demo mode with sample data

---

## Project Structure (Proposed)

```
intent-detection/
├── analysis/                    # Existing analysis code
│   ├── eeg_analysis.py
│   └── enhanced_analysis.py
├── data/                        # Existing data
│   ├── cleaned/
│   ├── raw/
│   ├── physionet/              # NEW: PhysioNet data
│   └── experiment_metadata.json
├── phase1/                      # NEW: Problem demonstration
│   ├── physionet_loader.py
│   ├── csp_lda_baseline.py
│   ├── cross_dataset_evaluation.py
│   └── generate_figures.py
├── phase2/                      # NEW: Multi-condition solution
│   ├── data_organizer.py
│   ├── multicond_csp_lda.py
│   ├── eegnet_classifier.py
│   ├── confusion_analysis.py
│   ├── spatial_analysis.py
│   └── generate_figures.py
├── phase3/                      # NEW: Transfer learning
│   ├── autoencoder_transfer.py
│   ├── augmented_training.py
│   ├── domain_analysis.py
│   └── generate_figures.py
├── phase4/                      # NEW: Real-time demo
│   ├── model_optimization.py
│   └── demo_app.py
├── models/                      # NEW: Saved models
│   ├── physionet_csp_lda.pkl
│   ├── multicond_csp_lda.pkl
│   └── eegnet_best.pt
├── results/                     # Existing + new figures
│   ├── figures/
│   └── tables/
├── notebooks/
│   └── eeg_analysis_report.ipynb
├── requirements.txt             # NEW: Dependencies
├── README.md                    # NEW: Project documentation
└── IMPLEMENTATION_PLAN.md       # This file
```

---

## Dependencies to Add

```txt
# requirements.txt
mne>=1.0.0
numpy>=1.21.0
pandas>=1.3.0
scipy>=1.7.0
matplotlib>=3.4.0
seaborn>=0.11.0
scikit-learn>=1.0.0

# Deep Learning (choose one framework)
torch>=2.0.0
# OR tensorflow>=2.10.0

# Optional but recommended
braindecode>=0.7.0  # EEGNet implementation
moabb>=0.4.0        # BCI benchmarking tools

# Demo app
streamlit>=1.20.0
# OR gradio>=3.0.0

# Visualization
plotly>=5.0.0       # Interactive plots
umap-learn>=0.5.0   # For domain analysis

# Model optimization
onnx>=1.12.0        # Model export
onnxruntime>=1.12.0 # Fast inference
```

---

## Critical Success Metrics

| Metric | Phase 1 (Baseline) | Phase 2 (Goal) | Phase 3 (Enhanced) |
|--------|-------------------|----------------|-------------------|
| Accuracy | 61% (PhysioNet→yours) | 81% | 85% |
| Sensitivity | 72% | 79% | 83% |
| Specificity | 51% (problem!) | 83% | 87% |
| F1 Score | 0.64 | 0.80 | 0.85 |

---

## Execution Checklist

### Phase 1 (Problem Demonstration)
- [ ] Set up PhysioNet data loader
- [ ] Train CSP+LDA on PhysioNet
- [ ] Test on your intent conditions
- [ ] Test on your non-intent conditions
- [ ] Generate Figure 1

### Phase 2 (Multi-Condition Training)
- [ ] Organize data for classification
- [ ] Implement CSP+LDA multi-condition
- [ ] Implement EEGNet classifier
- [ ] Run confusion analysis
- [ ] Run spatial analysis
- [ ] Generate Figure 2 and Table 1

### Phase 3 (Transfer Learning)
- [ ] Implement autoencoder transfer
- [ ] Implement augmented training
- [ ] Run domain analysis
- [ ] Generate Figure 3

### Phase 4 (Demo)
- [ ] Optimize model for inference
- [ ] Build interactive demo app
- [ ] Create documentation

### Final Deliverables
- [ ] GitHub repository (clean, documented)
- [ ] Technical report/paper
- [ ] Interactive demo
- [ ] Dataset release (anonymized)

---

## Notes on Your Data

**Strengths**:
- Perfect experimental design with all key conditions
- Clean separation: intent (voluntary/imagery) vs. non-intent (EMS/passive)
- Two subjects allows leave-one-out validation
- Consistent preprocessing already done

**Limitations to Address**:
- N=2 subjects is small → heavy data augmentation needed
- CP3 channel not in PhysioNet → will use C3, C4, Cz for cross-dataset
- May need to combine EMS short/long and passive trials for power

**Recommendations**:
1. Use aggressive data augmentation (noise, time shifts)
2. Consider 4-class then collapse to binary for more insight
3. The "proof of concept" claim is valid even with N=2 if results are clean
4. Emphasize the METHODOLOGY contribution, not just results
