"""
Cross-Dataset Evaluation

Tests the PhysioNet-trained CSP+LDA classifier on your local data.
This demonstrates the critical intent detection problem:
- Standard MI-BCIs work on intent conditions
- But FAIL on non-intent conditions (EMS, passive) with high false positives

This is the key result for Phase 1.
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Tuple, List
import json
import mne
from scipy import signal

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'analysis'))

from csp_lda_baseline import CSPLDAClassifier, evaluate_classifier


class LocalDataLoader:
    """Load and prepare your local EEG data for cross-dataset evaluation."""

    def __init__(
        self,
        data_dir: str = None,
        metadata_path: str = None
    ):
        # Get the project root directory (parent of phase1/)
        project_root = Path(__file__).parent.parent

        # Use project root-relative paths by default
        self.data_dir = Path(data_dir) if data_dir else project_root / 'data' / 'cleaned'
        self.metadata_path = Path(metadata_path) if metadata_path else project_root / 'data' / 'experiment_metadata.json'

        # Must match PhysioNet preprocessing
        self.sfreq = 200
        self.target_channels = ['C3', 'C4', 'Cz']  # Exclude CP3 for compatibility
        self.ch_names_local = ['C3', 'C4', 'Cz', 'CP3']

        # Load metadata
        with open(self.metadata_path, 'r') as f:
            self.metadata = json.load(f)

    def load_csv_to_epochs(
        self,
        filepath: Path,
        epoch_duration: float = 2.0,
        overlap: float = 0.5
    ) -> np.ndarray:
        """
        Load CSV file and create epochs matching PhysioNet format.

        Returns
        -------
        epochs : ndarray, shape (n_epochs, n_channels, n_times)
        """
        df = pd.read_csv(filepath, comment='#')

        # Extract EEG channels
        eeg_data = df[['eeg_ch0', 'eeg_ch1', 'eeg_ch2', 'eeg_ch3']].values.T

        # Convert microvolts to volts
        eeg_data = eeg_data * 1e-6

        # Select only C3, C4, Cz (indices 0, 1, 2) - exclude CP3
        eeg_data = eeg_data[:3, :]

        # Create MNE Raw object
        info = mne.create_info(
            ch_names=self.target_channels,
            sfreq=self.sfreq,
            ch_types=['eeg'] * 3
        )
        raw = mne.io.RawArray(eeg_data, info, verbose=False)

        # Apply same preprocessing as PhysioNet
        raw.filter(l_freq=1.0, h_freq=50.0, fir_design='firwin', verbose=False)
        raw.notch_filter(freqs=60.0, fir_design='firwin', verbose=False)

        # Create fixed-length epochs
        n_samples = eeg_data.shape[1]
        epoch_samples = int(epoch_duration * self.sfreq)
        step_samples = int(epoch_samples * (1 - overlap))

        epochs_list = []
        start = 0
        while start + epoch_samples <= n_samples:
            epoch = raw.get_data()[:, start:start + epoch_samples]
            epochs_list.append(epoch)
            start += step_samples

        if len(epochs_list) == 0:
            return np.array([])

        return np.array(epochs_list)

    def load_condition_data(self, condition_type: str) -> Tuple[np.ndarray, List[str]]:
        """
        Load all data for a specific condition type.

        Parameters
        ----------
        condition_type : str
            One of: 'intent', 'non_intent', 'voluntary', 'imagery', 'ems', 'passive'

        Returns
        -------
        X : ndarray, shape (n_epochs, n_channels, n_times)
        files_loaded : list of str
        """
        condition_patterns = {
            'intent': ['voluntary', 'Motor imagery'],
            'non_intent': ['EMS', 'Experimenter'],
            'voluntary': ['voluntary'],
            'imagery': ['Motor imagery'],
            'ems': ['EMS'],
            'passive': ['Experimenter']
        }

        patterns = condition_patterns.get(condition_type, [condition_type])

        all_epochs = []
        files_loaded = []

        for subject_id, subject_info in self.metadata['subjects'].items():
            session_dir = self.data_dir / subject_info['session_id']

            for filename, condition in subject_info['conditions'].items():
                # Check if this condition matches any pattern
                if any(p.lower() in condition.lower() for p in patterns):
                    filepath = session_dir / filename

                    if filepath.exists():
                        epochs = self.load_csv_to_epochs(filepath)
                        if len(epochs) > 0:
                            all_epochs.append(epochs)
                            files_loaded.append(f"{subject_id}: {condition}")
                            print(f"  Loaded {len(epochs)} epochs from {filename}")

        if all_epochs:
            X = np.concatenate(all_epochs, axis=0)
            return X, files_loaded
        else:
            return np.array([]), []

    def get_all_conditions_data(self) -> Dict[str, np.ndarray]:
        """Load data for all condition types."""
        conditions = ['voluntary', 'imagery', 'ems', 'passive']
        data = {}

        for cond in conditions:
            print(f"\nLoading {cond} condition...")
            X, files = self.load_condition_data(cond)
            data[cond] = {
                'X': X,
                'files': files,
                'n_epochs': len(X)
            }
            print(f"  Total: {len(X)} epochs")

        return data


def evaluate_on_local_data(
    clf: CSPLDAClassifier,
    loader: LocalDataLoader
) -> Dict[str, Dict]:
    """
    Evaluate PhysioNet-trained classifier on local data.
    """
    print("\n" + "="*60)
    print("CROSS-DATASET EVALUATION")
    print("Testing PhysioNet-trained classifier on local data")
    print("="*60)

    results = {}

    # Load all conditions
    all_data = loader.get_all_conditions_data()

    # Test on INTENT conditions (should work reasonably well)
    print("\n" + "-"*60)
    print("INTENT CONDITIONS (Expected: reasonable accuracy)")
    print("-"*60)

    intent_data = []
    for cond in ['voluntary', 'imagery']:
        if all_data[cond]['n_epochs'] > 0:
            intent_data.append(all_data[cond]['X'])

    if intent_data:
        X_intent = np.concatenate(intent_data, axis=0)
        # Label as "imagery" (class 1) since these represent intent
        y_intent = np.ones(len(X_intent))

        y_pred = clf.predict(X_intent)

        # How many are correctly identified as "intent" (class 1)?
        accuracy = np.mean(y_pred == 1)
        results['intent'] = {
            'n_samples': len(X_intent),
            'predicted_as_intent': np.sum(y_pred == 1),
            'predicted_as_rest': np.sum(y_pred == 0),
            'intent_rate': accuracy  # Should be high
        }

        print(f"\nIntent conditions ({len(X_intent)} epochs):")
        print(f"  Predicted as INTENT: {results['intent']['predicted_as_intent']} ({accuracy:.1%})")
        print(f"  Predicted as REST:   {results['intent']['predicted_as_rest']} ({1-accuracy:.1%})")

    # Test on NON-INTENT conditions (THE CRITICAL TEST)
    print("\n" + "-"*60)
    print("NON-INTENT CONDITIONS (Expected: HIGH FALSE POSITIVES)")
    print("-"*60)

    for cond in ['ems', 'passive']:
        if all_data[cond]['n_epochs'] > 0:
            X_cond = all_data[cond]['X']
            # These should be classified as "rest" (class 0) - NOT intent
            y_true = np.zeros(len(X_cond))

            y_pred = clf.predict(X_cond)

            # Specificity = correctly classified as non-intent
            specificity = np.mean(y_pred == 0)
            false_positive_rate = np.mean(y_pred == 1)  # Incorrectly called "intent"

            results[cond] = {
                'n_samples': len(X_cond),
                'predicted_as_intent': np.sum(y_pred == 1),
                'predicted_as_rest': np.sum(y_pred == 0),
                'specificity': specificity,
                'false_positive_rate': false_positive_rate
            }

            print(f"\n{cond.upper()} condition ({len(X_cond)} epochs):")
            print(f"  Predicted as INTENT (FALSE POSITIVE): {results[cond]['predicted_as_intent']} ({false_positive_rate:.1%})")
            print(f"  Predicted as REST (correct):          {results[cond]['predicted_as_rest']} ({specificity:.1%})")
            print(f"  >>> SPECIFICITY: {specificity:.1%} {'(PROBLEM!)' if specificity < 0.7 else ''}")

    # Summary
    print("\n" + "="*60)
    print("SUMMARY: THE INTENT DETECTION PROBLEM")
    print("="*60)

    if 'intent' in results and 'ems' in results:
        print(f"""
Standard MI-BCI trained on PhysioNet:
  - Intent detection rate:  {results['intent']['intent_rate']:.1%} (voluntary + imagery)
  - EMS specificity:        {results['ems']['specificity']:.1%} (should be >90%)
  - EMS false positive rate: {results['ems']['false_positive_rate']:.1%}
  - Passive specificity:    {results.get('passive', {}).get('specificity', 0):.1%}

PROBLEM: The classifier incorrectly identifies {results['ems']['false_positive_rate']:.0%} of
EMS-evoked movements as volitional intent!

This demonstrates why standard MI-BCIs cannot be used in hybrid BCI-FES systems.
""")

    return results


def main():
    """Run cross-dataset evaluation."""
    print("="*60)
    print("Phase 1: Cross-Dataset Evaluation")
    print("Demonstrating the Intent Detection Problem")
    print("="*60)

    # Load trained classifier
    model_path = Path("models/physionet_csp_lda.pkl")

    if not model_path.exists():
        print(f"\nModel not found at {model_path}")
        print("Please run csp_lda_baseline.py first to train the model.")
        return None

    clf = CSPLDAClassifier.load(str(model_path))

    # Load local data (uses project-root-relative paths by default)
    loader = LocalDataLoader()

    # Evaluate
    results = evaluate_on_local_data(clf, loader)

    # Save results
    results_path = Path("results/cross_dataset_evaluation.json")
    results_path.parent.mkdir(parents=True, exist_ok=True)

    # Convert numpy types for JSON serialization
    results_serializable = {}
    for k, v in results.items():
        results_serializable[k] = {
            kk: float(vv) if isinstance(vv, (np.floating, np.integer)) else vv
            for kk, vv in v.items()
        }

    with open(results_path, 'w') as f:
        json.dump(results_serializable, f, indent=2)

    print(f"\nResults saved to {results_path}")

    return results


if __name__ == "__main__":
    results = main()
