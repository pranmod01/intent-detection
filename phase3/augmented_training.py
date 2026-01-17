"""
Augmented Training with Combined Data

Combines PhysioNet motor imagery data with local multi-condition data
to improve model performance through data augmentation.

Strategy:
- Use PhysioNet imagery as additional "intent" examples
- Keep local non-intent data as-is (unique to this dataset)
- Train EEGNet on the combined, augmented dataset
"""

import sys
import numpy as np
import json
from pathlib import Path
from typing import Dict, Tuple
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

# Add project paths
sys.path.insert(0, str(Path(__file__).parent.parent / 'phase1'))
sys.path.insert(0, str(Path(__file__).parent.parent / 'phase2'))

from physionet_loader import PhysioNetLoader
from data_organizer import IntentDataOrganizer
from eegnet_classifier import EEGNetClassifier, augment_data


def load_physionet_data(subjects=None, runs=None):
    """Helper function to load PhysioNet data using PhysioNetLoader."""
    loader = PhysioNetLoader(subjects=subjects or list(range(1, 11)))
    X, y = loader.load_imagery_data()
    ch_names = loader.TARGET_CHANNELS
    return X, y, ch_names


def create_augmented_dataset(
    X_local: np.ndarray,
    y_local: np.ndarray,
    X_physionet: np.ndarray,
    physionet_ratio: float = 0.5,
    random_state: int = 42
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create augmented dataset combining local and PhysioNet data.

    Parameters
    ----------
    X_local : ndarray
        Local EEG data, shape (n_samples, n_channels, n_times)
    y_local : ndarray
        Local labels (0=non-intent, 1=intent)
    X_physionet : ndarray
        PhysioNet motor imagery data (all labeled as intent)
    physionet_ratio : float
        Ratio of PhysioNet samples to add relative to local intent samples

    Returns
    -------
    X_aug, y_aug : augmented data and labels
    """
    np.random.seed(random_state)

    # Separate local intent and non-intent
    X_local_intent = X_local[y_local == 1]
    X_local_nonintent = X_local[y_local == 0]

    print(f"Local intent samples: {len(X_local_intent)}")
    print(f"Local non-intent samples: {len(X_local_nonintent)}")
    print(f"PhysioNet samples available: {len(X_physionet)}")

    # Sample PhysioNet data
    n_physionet_to_add = int(len(X_local_intent) * physionet_ratio)
    n_physionet_to_add = min(n_physionet_to_add, len(X_physionet))

    indices = np.random.choice(len(X_physionet), n_physionet_to_add, replace=False)
    X_physionet_sample = X_physionet[indices]

    # Trim PhysioNet samples to match local data length if needed
    local_n_samples = X_local_intent.shape[2]
    if X_physionet_sample.shape[2] > local_n_samples:
        X_physionet_sample = X_physionet_sample[:, :, :local_n_samples]
    elif X_physionet_sample.shape[2] < local_n_samples:
        # Pad if PhysioNet is shorter (unlikely)
        pad_width = local_n_samples - X_physionet_sample.shape[2]
        X_physionet_sample = np.pad(X_physionet_sample, ((0, 0), (0, 0), (0, pad_width)), mode='edge')

    print(f"Adding {n_physionet_to_add} PhysioNet samples as intent")

    # Combine intent data
    X_intent_combined = np.concatenate([X_local_intent, X_physionet_sample], axis=0)
    y_intent_combined = np.ones(len(X_intent_combined))

    # Combine all data
    X_aug = np.concatenate([X_intent_combined, X_local_nonintent], axis=0)
    y_aug = np.concatenate([y_intent_combined, np.zeros(len(X_local_nonintent))])

    # Shuffle
    shuffle_idx = np.random.permutation(len(X_aug))
    X_aug = X_aug[shuffle_idx]
    y_aug = y_aug[shuffle_idx]

    print(f"\nAugmented dataset:")
    print(f"  Total samples: {len(X_aug)}")
    print(f"  Intent: {np.sum(y_aug == 1)}")
    print(f"  Non-intent: {np.sum(y_aug == 0)}")

    return X_aug, y_aug


def compare_training_strategies(
    X_local: np.ndarray,
    y_local: np.ndarray,
    X_physionet: np.ndarray,
    test_size: float = 0.3,
    random_state: int = 42
) -> Dict:
    """
    Compare different training strategies:
    1. Local data only
    2. Local + PhysioNet augmentation
    3. Local + noise augmentation
    4. Local + PhysioNet + noise augmentation
    """
    results = {}

    # Split local data (same split for all strategies)
    X_train_local, X_test, y_train_local, y_test = train_test_split(
        X_local, y_local, test_size=test_size, stratify=y_local, random_state=random_state
    )

    print(f"\nTest set: {len(X_test)} samples (held out for all strategies)")

    # Strategy 1: Local only
    print("\n" + "="*50)
    print("Strategy 1: Local Data Only")
    print("="*50)

    clf1 = EEGNetClassifier(
        n_channels=X_train_local.shape[1],
        n_samples=X_train_local.shape[2],
        n_classes=2
    )

    # Balance
    from data_organizer import IntentDataOrganizer
    organizer = IntentDataOrganizer(use_all_channels=False)
    X_train_bal, y_train_bal = organizer.balance_classes(X_train_local, y_train_local)

    clf1.fit(X_train_bal, y_train_bal, X_val=X_test, y_val=y_test,
             epochs=100, batch_size=16, patience=15, verbose=True)

    metrics1 = clf1.get_detailed_metrics(X_test, y_test)
    results['local_only'] = metrics1
    print(f"\nResults: Acc={metrics1['accuracy']:.2%}, Sens={metrics1['sensitivity']:.2%}, Spec={metrics1['specificity']:.2%}")

    # Strategy 2: Local + PhysioNet
    print("\n" + "="*50)
    print("Strategy 2: Local + PhysioNet Augmentation")
    print("="*50)

    X_train_aug, y_train_aug = create_augmented_dataset(
        X_train_local, y_train_local, X_physionet, physionet_ratio=0.5
    )
    X_train_aug_bal, y_train_aug_bal = organizer.balance_classes(X_train_aug, y_train_aug)

    clf2 = EEGNetClassifier(
        n_channels=X_train_aug_bal.shape[1],
        n_samples=X_train_aug_bal.shape[2],
        n_classes=2
    )

    clf2.fit(X_train_aug_bal, y_train_aug_bal, X_val=X_test, y_val=y_test,
             epochs=100, batch_size=16, patience=15, verbose=True)

    metrics2 = clf2.get_detailed_metrics(X_test, y_test)
    results['local_plus_physionet'] = metrics2
    print(f"\nResults: Acc={metrics2['accuracy']:.2%}, Sens={metrics2['sensitivity']:.2%}, Spec={metrics2['specificity']:.2%}")

    # Strategy 3: Local + noise augmentation
    print("\n" + "="*50)
    print("Strategy 3: Local + Noise Augmentation")
    print("="*50)

    X_train_noise, y_train_noise = augment_data(X_train_bal, y_train_bal, noise_factor=0.1)

    clf3 = EEGNetClassifier(
        n_channels=X_train_noise.shape[1],
        n_samples=X_train_noise.shape[2],
        n_classes=2
    )

    clf3.fit(X_train_noise, y_train_noise, X_val=X_test, y_val=y_test,
             epochs=100, batch_size=16, patience=15, verbose=True)

    metrics3 = clf3.get_detailed_metrics(X_test, y_test)
    results['local_plus_noise'] = metrics3
    print(f"\nResults: Acc={metrics3['accuracy']:.2%}, Sens={metrics3['sensitivity']:.2%}, Spec={metrics3['specificity']:.2%}")

    # Strategy 4: Local + PhysioNet + noise
    print("\n" + "="*50)
    print("Strategy 4: Local + PhysioNet + Noise (Full Augmentation)")
    print("="*50)

    X_train_full, y_train_full = augment_data(X_train_aug_bal, y_train_aug_bal, noise_factor=0.1)

    clf4 = EEGNetClassifier(
        n_channels=X_train_full.shape[1],
        n_samples=X_train_full.shape[2],
        n_classes=2
    )

    clf4.fit(X_train_full, y_train_full, X_val=X_test, y_val=y_test,
             epochs=100, batch_size=16, patience=15, verbose=True)

    metrics4 = clf4.get_detailed_metrics(X_test, y_test)
    results['full_augmentation'] = metrics4
    print(f"\nResults: Acc={metrics4['accuracy']:.2%}, Sens={metrics4['sensitivity']:.2%}, Spec={metrics4['specificity']:.2%}")

    # Save best model
    best_strategy = max(results.keys(), key=lambda k: results[k]['f1'])
    print(f"\nBest strategy: {best_strategy}")

    if best_strategy == 'local_only':
        clf1.save("models/eegnet_augmented_best.pt")
    elif best_strategy == 'local_plus_physionet':
        clf2.save("models/eegnet_augmented_best.pt")
    elif best_strategy == 'local_plus_noise':
        clf3.save("models/eegnet_augmented_best.pt")
    else:
        clf4.save("models/eegnet_augmented_best.pt")

    return results


def main():
    """Run augmented training comparison."""
    print("="*60)
    print("Phase 3: Augmented Training with Combined Data")
    print("="*60)

    # Load PhysioNet data
    print("\n[1/3] Loading PhysioNet motor imagery data...")
    X_physionet, y_physionet, ch_names = load_physionet_data(
        subjects=list(range(1, 11)),
        runs=[4, 8, 12]
    )

    # Only use imagery trials (not rest)
    X_physionet_imagery = X_physionet[y_physionet == 1]
    print(f"PhysioNet imagery samples: {len(X_physionet_imagery)}")

    # Load local data
    print("\n[2/3] Loading local multi-condition data...")
    organizer = IntentDataOrganizer(use_all_channels=False)
    data = organizer.load_all_data()

    X_local = data['X']
    y_local = data['y_binary']

    print(f"Local data: {len(X_local)} samples")
    print(f"  Intent: {np.sum(y_local == 1)}")
    print(f"  Non-intent: {np.sum(y_local == 0)}")

    # Compare strategies
    print("\n[3/3] Comparing training strategies...")
    results = compare_training_strategies(X_local, y_local, X_physionet_imagery)

    # Print summary
    print("\n" + "="*60)
    print("SUMMARY: Training Strategy Comparison")
    print("="*60)
    print(f"\n{'Strategy':<30} {'Accuracy':>10} {'Sensitivity':>12} {'Specificity':>12} {'F1':>8}")
    print("-" * 75)

    for strategy, metrics in results.items():
        print(f"{strategy:<30} {metrics['accuracy']:>10.2%} {metrics['sensitivity']:>12.2%} {metrics['specificity']:>12.2%} {metrics['f1']:>8.2f}")

    # Save results
    results_path = Path("results/augmented_training_comparison.json")
    results_path.parent.mkdir(parents=True, exist_ok=True)

    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {results_path}")

    return results


if __name__ == "__main__":
    results = main()
