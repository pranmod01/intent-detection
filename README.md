# Motor Intent Detection in Hybrid BCIs

Distinguishing volitional motor intent from externally-evoked neural responses in EEG signals.

## Problem

Standard motor imagery BCIs trained on public datasets (PhysioNet) fail to distinguish between intentional and non-intentional movements, producing 40-60% false positive rates on EMS-evoked and passive movements.

## Solution

Train classifiers on multi-condition data that explicitly includes non-intent examples, improving specificity from ~51% to 83%+.

## Project Structure

```
├── phase1/          # Problem demonstration (PhysioNet baseline)
├── phase2/          # Multi-condition training (CSP+LDA, EEGNet)
├── phase3/          # Transfer learning enhancement
├── phase4/          # Streamlit demo application
├── data/            # Raw, cleaned, and PhysioNet data
├── models/          # Trained model checkpoints
├── results/         # Evaluation metrics and figures
└── analysis/        # EEG processing utilities
```

## Conditions

**Intent:** Voluntary movement (left/right hand), Motor imagery
**Non-Intent:** EMS-evoked (short/long pulses), Passive movement

## Quick Start

```bash
pip install -r requirements.txt

python run_pipeline.py --phase 1    # Problem demonstration
python run_pipeline.py --phase 2    # Multi-condition training
python run_pipeline.py --phase 3    # Transfer learning
python run_pipeline.py --demo       # Launch interactive demo
python run_pipeline.py --all        # Run all phases
```

## Methods

- **CSP + LDA**: Classical spatial filtering with linear discriminant analysis
- **EEGNet**: Compact CNN architecture for EEG classification
- **Transfer Learning**: Autoencoder pre-trained on PhysioNet, fine-tuned on local data

## Data

- **Local**: 2 subjects, 6 conditions, 4 channels (C3, C4, Cz, CP3), 200 Hz
- **PhysioNet**: EEGMMIDB motor imagery dataset for cross-validation

## Requirements

Core: mne, numpy, scipy, scikit-learn, torch
Demo: streamlit, plotly
