"""
Data Organizer for Multi-Condition Classification

Prepares your local data for training intent vs. non-intent classifiers.
Handles:
- Loading all conditions from both subjects
- Labeling for binary (intent/non-intent) or multi-class classification
- Train/test splits (leave-one-subject-out)
- Class balancing
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Tuple, List, Optional
import json
import mne
from sklearn.model_selection import LeaveOneGroupOut
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent.parent / 'analysis'))


class IntentDataOrganizer:
    """Organize EEG data for intent classification."""

    # Condition mappings
    INTENT_CONDITIONS = ['voluntary', 'imagery']
    NON_INTENT_CONDITIONS = ['ems', 'passive']

    CONDITION_LABELS = {
        'voluntary_left': 0,
        'voluntary_right': 1,
        'imagery': 2,
        'ems_short': 3,
        'ems_long': 4,
        'passive': 5
    }

    def __init__(
        self,
        data_dir: str = None,
        metadata_path: str = None,
        use_all_channels: bool = True
    ):
        """
        Initialize data organizer.

        Parameters
        ----------
        data_dir : str
            Path to cleaned data directory
        metadata_path : str
            Path to experiment metadata JSON
        use_all_channels : bool
            If True, use all 4 channels (C3, C4, Cz, CP3)
            If False, use only 3 channels for PhysioNet compatibility
        """
        # Get project root directory (parent of phase2/)
        project_root = Path(__file__).parent.parent

        self.data_dir = Path(data_dir) if data_dir else project_root / 'data' / 'cleaned'
        self.metadata_path = Path(metadata_path) if metadata_path else project_root / 'data' / 'experiment_metadata.json'

        self.sfreq = 200
        self.use_all_channels = use_all_channels

        if use_all_channels:
            self.ch_names = ['C3', 'C4', 'Cz', 'CP3']
            self.n_channels = 4
        else:
            self.ch_names = ['C3', 'C4', 'Cz']
            self.n_channels = 3

        # Load metadata
        with open(self.metadata_path, 'r') as f:
            self.metadata = json.load(f)

        self.data_cache = {}

    def _load_single_file(
        self,
        filepath: Path,
        epoch_duration: float = 2.0,
        overlap: float = 0.5
    ) -> np.ndarray:
        """Load and epoch a single CSV file."""
        df = pd.read_csv(filepath, comment='#')

        # Extract EEG channels
        eeg_data = df[['eeg_ch0', 'eeg_ch1', 'eeg_ch2', 'eeg_ch3']].values.T
        eeg_data = eeg_data * 1e-6  # Convert to volts

        # Select channels
        if not self.use_all_channels:
            eeg_data = eeg_data[:3, :]

        # Create MNE Raw object
        info = mne.create_info(
            ch_names=self.ch_names,
            sfreq=self.sfreq,
            ch_types=['eeg'] * self.n_channels
        )
        raw = mne.io.RawArray(eeg_data, info, verbose=False)

        # Preprocess
        raw.filter(l_freq=1.0, h_freq=50.0, fir_design='firwin', verbose=False)
        raw.notch_filter(freqs=60.0, fir_design='firwin', verbose=False)

        # Create epochs
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

    def _categorize_condition(self, condition_str: str) -> Tuple[str, str]:
        """
        Categorize a condition string into type and subtype.

        Returns
        -------
        condition_type : str
            'intent' or 'non_intent'
        condition_subtype : str
            e.g., 'voluntary_left', 'ems_short', etc.
        """
        condition_lower = condition_str.lower()

        if 'left' in condition_lower and 'voluntary' in condition_lower:
            return 'intent', 'voluntary_left'
        elif 'right' in condition_lower and 'voluntary' in condition_lower:
            return 'intent', 'voluntary_right'
        elif 'imagery' in condition_lower:
            return 'intent', 'imagery'
        elif 'ems' in condition_lower and 'long' in condition_lower:
            return 'non_intent', 'ems_long'
        elif 'ems' in condition_lower:
            return 'non_intent', 'ems_short'
        elif 'experimenter' in condition_lower or 'passive' in condition_lower:
            return 'non_intent', 'passive'
        else:
            return 'unknown', 'unknown'

    def load_all_data(
        self,
        epoch_duration: float = 2.0,
        overlap: float = 0.5
    ) -> Dict:
        """
        Load all data from all subjects and conditions.

        Returns
        -------
        data : dict
            {
                'X': ndarray (n_epochs, n_channels, n_times),
                'y_binary': ndarray (n_epochs,) - 0=non_intent, 1=intent,
                'y_multiclass': ndarray (n_epochs,) - condition labels,
                'subjects': ndarray (n_epochs,) - subject IDs,
                'conditions': list - condition names per epoch
            }
        """
        print("Loading all data...")

        all_epochs = []
        all_binary_labels = []
        all_multi_labels = []
        all_subjects = []
        all_conditions = []

        for subject_idx, (subject_id, subject_info) in enumerate(self.metadata['subjects'].items()):
            print(f"\n{subject_id}:")
            session_dir = self.data_dir / subject_info['session_id']

            for filename, condition in subject_info['conditions'].items():
                filepath = session_dir / filename

                if not filepath.exists():
                    print(f"  Warning: {filename} not found")
                    continue

                epochs = self._load_single_file(filepath, epoch_duration, overlap)

                if len(epochs) == 0:
                    continue

                cond_type, cond_subtype = self._categorize_condition(condition)

                if cond_type == 'unknown':
                    print(f"  Skipping unknown condition: {condition}")
                    continue

                # Binary label: 0 = non_intent, 1 = intent
                binary_label = 1 if cond_type == 'intent' else 0

                # Multi-class label
                multi_label = self.CONDITION_LABELS.get(cond_subtype, -1)

                all_epochs.append(epochs)
                all_binary_labels.extend([binary_label] * len(epochs))
                all_multi_labels.extend([multi_label] * len(epochs))
                all_subjects.extend([subject_idx] * len(epochs))
                all_conditions.extend([cond_subtype] * len(epochs))

                print(f"  {cond_subtype}: {len(epochs)} epochs (binary={binary_label})")

        # Concatenate
        X = np.concatenate(all_epochs, axis=0)
        y_binary = np.array(all_binary_labels)
        y_multi = np.array(all_multi_labels)
        subjects = np.array(all_subjects)

        data = {
            'X': X,
            'y_binary': y_binary,
            'y_multiclass': y_multi,
            'subjects': subjects,
            'conditions': all_conditions
        }

        self.data_cache = data

        # Print summary
        print("\n" + "="*50)
        print("DATA SUMMARY")
        print("="*50)
        print(f"Total epochs: {len(X)}")
        print(f"Shape: {X.shape}")
        print(f"Subjects: {len(np.unique(subjects))}")
        print(f"\nBinary class distribution:")
        print(f"  Intent (1):     {np.sum(y_binary == 1)} epochs")
        print(f"  Non-Intent (0): {np.sum(y_binary == 0)} epochs")
        print(f"\nMulti-class distribution:")
        for cond, label in self.CONDITION_LABELS.items():
            count = np.sum(y_multi == label)
            print(f"  {cond}: {count} epochs")

        return data

    def get_train_test_split(
        self,
        test_subject: int = 1
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Get leave-one-subject-out train/test split.

        Parameters
        ----------
        test_subject : int
            Subject index to use as test set (0 or 1)

        Returns
        -------
        X_train, X_test, y_train, y_test : ndarrays
        """
        if not self.data_cache:
            self.load_all_data()

        X = self.data_cache['X']
        y = self.data_cache['y_binary']
        subjects = self.data_cache['subjects']

        train_mask = subjects != test_subject
        test_mask = subjects == test_subject

        return (
            X[train_mask],
            X[test_mask],
            y[train_mask],
            y[test_mask]
        )

    def get_cross_validation_splits(self) -> List[Tuple]:
        """
        Get leave-one-subject-out cross-validation splits.

        Returns
        -------
        splits : list of (train_idx, test_idx) tuples
        """
        if not self.data_cache:
            self.load_all_data()

        subjects = self.data_cache['subjects']
        logo = LeaveOneGroupOut()

        return list(logo.split(self.data_cache['X'], self.data_cache['y_binary'], subjects))

    def balance_classes(
        self,
        X: np.ndarray,
        y: np.ndarray,
        method: str = 'undersample'
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Balance classes by undersampling majority or oversampling minority.

        Parameters
        ----------
        X : ndarray
        y : ndarray
        method : str
            'undersample' or 'oversample'

        Returns
        -------
        X_balanced, y_balanced : ndarrays
        """
        classes, counts = np.unique(y, return_counts=True)
        min_count = min(counts)
        max_count = max(counts)

        if method == 'undersample':
            target_count = min_count
        else:
            target_count = max_count

        balanced_X = []
        balanced_y = []

        for cls in classes:
            cls_mask = y == cls
            cls_X = X[cls_mask]
            cls_count = len(cls_X)

            if method == 'undersample':
                # Randomly select target_count samples
                indices = np.random.choice(cls_count, target_count, replace=False)
            else:
                # Oversample with replacement
                indices = np.random.choice(cls_count, target_count, replace=True)

            balanced_X.append(cls_X[indices])
            balanced_y.extend([cls] * target_count)

        return np.concatenate(balanced_X, axis=0), np.array(balanced_y)

    def get_filtered_data(
        self,
        freq_band: Tuple[float, float] = (8, 30)
    ) -> np.ndarray:
        """
        Apply additional bandpass filter for mu+beta band.

        Parameters
        ----------
        freq_band : tuple
            (low_freq, high_freq) in Hz

        Returns
        -------
        X_filtered : ndarray
        """
        if not self.data_cache:
            self.load_all_data()

        from scipy.signal import butter, filtfilt

        X = self.data_cache['X'].copy()
        low, high = freq_band

        # Design Butterworth filter
        nyq = self.sfreq / 2
        b, a = butter(4, [low / nyq, high / nyq], btype='band')

        # Apply filter to each epoch and channel
        for i in range(len(X)):
            for ch in range(X.shape[1]):
                X[i, ch, :] = filtfilt(b, a, X[i, ch, :])

        return X


def main():
    """Test the data organizer."""
    print("="*60)
    print("Data Organizer Test")
    print("="*60)

    organizer = IntentDataOrganizer(use_all_channels=True)

    # Load all data
    data = organizer.load_all_data()

    # Test train/test split
    print("\n" + "="*50)
    print("Leave-One-Subject-Out Split (Subject 1 as test)")
    print("="*50)

    X_train, X_test, y_train, y_test = organizer.get_train_test_split(test_subject=1)
    print(f"Train: {X_train.shape}, Test: {X_test.shape}")
    print(f"Train class balance: {np.sum(y_train==0)} non-intent, {np.sum(y_train==1)} intent")
    print(f"Test class balance: {np.sum(y_test==0)} non-intent, {np.sum(y_test==1)} intent")

    # Test balancing
    print("\n" + "="*50)
    print("Class Balancing (Undersample)")
    print("="*50)

    X_bal, y_bal = organizer.balance_classes(X_train, y_train, method='undersample')
    print(f"Balanced: {X_bal.shape}")
    print(f"Class distribution: {np.sum(y_bal==0)} non-intent, {np.sum(y_bal==1)} intent")

    return organizer, data


if __name__ == "__main__":
    organizer, data = main()
