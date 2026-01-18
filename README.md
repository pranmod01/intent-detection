# Motor Intent Detection in Hybrid BCIs

Distinguishing volitional motor intent from externally-evoked neural responses in EEG signals.

## The Problem

Hybrid Brain-Computer Interfaces (BCIs) combine neural decoding with functional electrical stimulation (FES) to restore movement in paralyzed patients. These systems detect motor intent from EEG signals and trigger FES to activate muscles. However, there's a critical flaw: **standard motor imagery classifiers cannot distinguish between voluntary intent and externally-evoked neural activity**.

When FES activates a patient's muscles, it generates sensory feedback that produces EEG patterns similar to voluntary movement. A naive classifier trained only on motor imagery data will misclassify these FES-evoked responses as intentional movement, creating a dangerous feedback loop where the system triggers itself.

This project demonstrates the problem and implements solutions for robust intent detection in hybrid BCI systems.

## Why This Matters

In rehabilitation robotics and neuroprosthetics, false positives aren't just inconvenient—they're potentially dangerous. If a BCI-FES system interprets muscle twitches or passive movements as intentional commands, it could:

- Trigger unwanted stimulation during therapy
- Create runaway feedback loops
- Undermine patient trust and adoption of assistive devices

Current public BCI datasets (like PhysioNet's motor imagery data) only contain voluntary movements, so classifiers trained on them have never learned what "non-intent" looks like.

## The Solution

The key insight is simple: **train on what you want to reject**. By collecting data across multiple conditions—voluntary movement, motor imagery, EMS-evoked movement, and passive movement—we can teach classifiers to distinguish true intent from neural responses that merely *look like* intent.

### Results

| Approach | Sensitivity | Specificity |
|----------|-------------|-------------|
| PhysioNet-only baseline | 91% | 51% |
| Multi-condition CSP+LDA | 85% | 83% |
| EEGNet (deep learning) | 87% | 81% |

The baseline classifier catches most intent (91% sensitivity) but produces unacceptable false positives (49% of non-intent classified as intent). Multi-condition training maintains sensitivity while dramatically improving specificity.

## Technical Approach

### Phase 1: Problem Demonstration
Train a standard CSP+LDA classifier on PhysioNet motor imagery data, then evaluate on local multi-condition data. This exposes the false positive problem—the PhysioNet-trained model misclassifies ~50% of EMS and passive trials as intentional.

### Phase 2: Multi-Condition Training
Train classifiers on all four conditions with binary labels (intent vs. non-intent):
- **CSP + LDA**: Common Spatial Patterns extract discriminative spatial filters; Linear Discriminant Analysis classifies the filtered signals
- **EEGNet**: Compact CNN designed for EEG that learns temporal and spatial features end-to-end

### Phase 3: Transfer Learning
Leverage PhysioNet's large dataset to pre-train feature extractors, then fine-tune on local multi-condition data. This helps when local data is limited.

### Phase 5: Temporal Intent Detection
LSTM-based model that predicts intent at each timestep, enabling detection of intent onset/offset within trials. Uses Leave-One-Subject-Out validation to demonstrate temporal modeling without claiming population generalization.

## Conditions

**Intent (positive class):**
- Voluntary movement (actual left/right hand movement)
- Motor imagery (imagined movement without execution)

**Non-Intent (negative class):**
- EMS-evoked movement (muscle activation via electrical stimulation)
- Passive movement (hand moved by external force)

## Project Structure

```
├── phase1/          # Problem demonstration (PhysioNet baseline)
├── phase2/          # Multi-condition training (CSP+LDA, EEGNet)
├── phase3/          # Transfer learning enhancement
├── phase4_demo/     # Streamlit interactive demo
├── phase5/          # Temporal LSTM intent detection
├── data/            # Raw, cleaned, and PhysioNet data
├── models/          # Trained model checkpoints
└── results/         # Evaluation metrics and figures
```

## Running the Pipeline

```bash
pip install -r requirements.txt

python run_pipeline.py --phase 1    # Problem demonstration
python run_pipeline.py --phase 2    # Multi-condition training
python run_pipeline.py --phase 3    # Transfer learning
python run_pipeline.py --phase 5    # Temporal LSTM
python run_pipeline.py --demo       # Interactive demo
python run_pipeline.py --all        # Run all phases
```

## Methods

- **Common Spatial Patterns (CSP)**: Learns spatial filters that maximize variance for one class while minimizing it for the other, extracting the most discriminative EEG channel combinations
- **EEGNet**: Compact convolutional architecture with depthwise/separable convolutions, designed specifically for EEG with limited training data
- **Bidirectional LSTM**: Captures temporal dynamics for frame-by-frame intent detection, enabling onset/offset localization

## Data

- **Local dataset**: 2 subjects, 6 conditions, 4 channels (C3, C4, Cz, CP3), 200 Hz sampling
- **PhysioNet EEGMMIDB**: 109 subjects motor imagery dataset for pre-training and cross-validation

## Key Takeaways

1. **The problem is real**: Standard MI-BCI classifiers fail on non-intent conditions
2. **The fix is straightforward**: Include negative examples during training
3. **Deep learning isn't required**: CSP+LDA matches EEGNet performance with proper training data
4. **Temporal modeling adds value**: LSTMs can detect intent onset within trials
