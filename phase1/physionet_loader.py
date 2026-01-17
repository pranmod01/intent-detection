"""
PhysioNet Motor Movement/Imagery Dataset Loader

Downloads and prepares the EEGMMIDB dataset for training a standard MI-BCI.
Matches preprocessing to your local data for fair comparison.

Dataset: https://physionet.org/content/eegmmidb/1.0.0/
"""

import os
import numpy as np
import mne
from mne.datasets import eegbci
from mne.io import concatenate_raws, read_raw_edf
from pathlib import Path
from typing import Tuple, List, Dict
import warnings

warnings.filterwarnings('ignore')


class PhysioNetLoader:
    """Load and preprocess PhysioNet Motor Movement/Imagery data."""

    # PhysioNet task runs
    RUNS_REST = [1]  # Baseline rest
    RUNS_IMAGERY_HANDS = [4, 8, 12]  # Imagine hands (open/close fists)
    RUNS_IMAGERY_FEET = [6, 10, 14]  # Imagine feet

    # Standard 10-20 channels to match your setup
    TARGET_CHANNELS = ['C3', 'C4', 'Cz']  # CP3 not in PhysioNet

    def __init__(
        self,
        data_dir: str = 'data/physionet',
        subjects: List[int] = None,
        sfreq: int = 200,
        l_freq: float = 1.0,
        h_freq: float = 50.0,
        notch_freq: float = 60.0
    ):
        """
        Initialize PhysioNet loader.

        Parameters
        ----------
        data_dir : str
            Directory to store downloaded data
        subjects : list of int
            Subject IDs to load (1-109). Default: [1-20]
        sfreq : int
            Target sampling frequency (Hz). Default: 200 to match your data
        l_freq, h_freq : float
            Bandpass filter frequencies
        notch_freq : float
            Notch filter frequency for line noise
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

        self.subjects = subjects if subjects else list(range(1, 21))  # 20 subjects
        self.sfreq = sfreq
        self.l_freq = l_freq
        self.h_freq = h_freq
        self.notch_freq = notch_freq

        self.raw_data = {}
        self.epochs_data = {}

    def download_subject(self, subject: int, runs: List[int]) -> mne.io.Raw:
        """
        Download and load raw data for a single subject.

        Parameters
        ----------
        subject : int
            Subject ID (1-109)
        runs : list of int
            Run IDs to load

        Returns
        -------
        raw : mne.io.Raw
            Concatenated raw data
        """
        print(f"  Downloading subject {subject}, runs {runs}...")

        try:
            raw_fnames = eegbci.load_data(
                subject,
                runs,
                path=str(self.data_dir)
            )
            raws = [read_raw_edf(f, preload=True, verbose=False) for f in raw_fnames]
            raw = concatenate_raws(raws)

            # Standardize channel names
            eegbci.standardize(raw)

            # Set montage
            raw.set_montage('standard_1020', on_missing='ignore')

            return raw

        except Exception as e:
            print(f"  Error loading subject {subject}: {e}")
            return None

    def preprocess_raw(self, raw: mne.io.Raw) -> mne.io.Raw:
        """
        Apply preprocessing to match your local data pipeline.

        Parameters
        ----------
        raw : mne.io.Raw
            Raw EEG data

        Returns
        -------
        raw_proc : mne.io.Raw
            Preprocessed data
        """
        raw_proc = raw.copy()

        # Pick only EEG channels
        raw_proc.pick_types(eeg=True, exclude='bads')

        # Pick target channels (C3, C4, Cz)
        available_channels = [ch for ch in self.TARGET_CHANNELS if ch in raw_proc.ch_names]
        if len(available_channels) < len(self.TARGET_CHANNELS):
            print(f"  Warning: Only found {available_channels}")

        raw_proc.pick_channels(available_channels)

        # Resample to target frequency
        if raw_proc.info['sfreq'] != self.sfreq:
            raw_proc.resample(self.sfreq)

        # Bandpass filter
        raw_proc.filter(l_freq=self.l_freq, h_freq=self.h_freq, fir_design='firwin')

        # Notch filter for line noise
        raw_proc.notch_filter(freqs=self.notch_freq, fir_design='firwin')

        return raw_proc

    def create_epochs(
        self,
        raw: mne.io.Raw,
        event_id: Dict[str, int],
        tmin: float = 0.0,
        tmax: float = 2.0
    ) -> mne.Epochs:
        """
        Create epochs from raw data using annotations.

        Parameters
        ----------
        raw : mne.io.Raw
            Preprocessed raw data
        event_id : dict
            Mapping of event names to IDs
        tmin, tmax : float
            Epoch time window

        Returns
        -------
        epochs : mne.Epochs
            Epoched data
        """
        events, _ = mne.events_from_annotations(raw, verbose=False)

        epochs = mne.Epochs(
            raw,
            events,
            event_id=event_id,
            tmin=tmin,
            tmax=tmax,
            baseline=None,
            preload=True,
            verbose=False
        )

        return epochs

    def load_imagery_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load motor imagery data for classification.

        Returns
        -------
        X : ndarray, shape (n_epochs, n_channels, n_times)
            EEG data
        y : ndarray, shape (n_epochs,)
            Labels: 0 = rest, 1 = motor imagery
        """
        all_epochs_imagery = []
        all_epochs_rest = []

        print("Loading PhysioNet Motor Imagery data...")

        for subject in self.subjects:
            print(f"\nSubject {subject}/{max(self.subjects)}:")

            # Load imagery runs
            raw_imagery = self.download_subject(subject, self.RUNS_IMAGERY_HANDS)
            if raw_imagery is None:
                continue

            raw_imagery_proc = self.preprocess_raw(raw_imagery)

            # PhysioNet event annotations
            # T0 = rest, T1 = left fist, T2 = right fist
            events_imagery, event_dict = mne.events_from_annotations(
                raw_imagery_proc, verbose=False
            )

            # Create epochs for imagery (T1 and T2 combined as "imagery")
            imagery_ids = {k: v for k, v in event_dict.items() if 'T1' in k or 'T2' in k}
            if imagery_ids:
                epochs_img = mne.Epochs(
                    raw_imagery_proc,
                    events_imagery,
                    event_id=imagery_ids,
                    tmin=0.0,
                    tmax=2.0,
                    baseline=None,
                    preload=True,
                    verbose=False
                )
                all_epochs_imagery.append(epochs_img.get_data())

            # Create epochs for rest (T0)
            rest_ids = {k: v for k, v in event_dict.items() if 'T0' in k}
            if rest_ids:
                epochs_rest = mne.Epochs(
                    raw_imagery_proc,
                    events_imagery,
                    event_id=rest_ids,
                    tmin=0.0,
                    tmax=2.0,
                    baseline=None,
                    preload=True,
                    verbose=False
                )
                all_epochs_rest.append(epochs_rest.get_data())

        # Combine all subjects
        if all_epochs_imagery and all_epochs_rest:
            X_imagery = np.concatenate(all_epochs_imagery, axis=0)
            X_rest = np.concatenate(all_epochs_rest, axis=0)

            # Balance classes
            n_samples = min(len(X_imagery), len(X_rest))
            X_imagery = X_imagery[:n_samples]
            X_rest = X_rest[:n_samples]

            X = np.concatenate([X_rest, X_imagery], axis=0)
            y = np.concatenate([
                np.zeros(len(X_rest)),
                np.ones(len(X_imagery))
            ])

            print(f"\nLoaded data:")
            print(f"  Rest epochs: {len(X_rest)}")
            print(f"  Imagery epochs: {len(X_imagery)}")
            print(f"  Total: {len(X)} epochs")
            print(f"  Shape: {X.shape}")

            return X, y

        else:
            raise ValueError("Failed to load sufficient data")

    def get_channel_info(self) -> Dict:
        """Return channel information for compatibility checking."""
        return {
            'channels': self.TARGET_CHANNELS,
            'sfreq': self.sfreq,
            'n_channels': len(self.TARGET_CHANNELS)
        }


def main():
    """Test the PhysioNet loader."""
    print("="*60)
    print("PhysioNet Motor Imagery Data Loader")
    print("="*60)

    # Initialize with 5 subjects for testing
    loader = PhysioNetLoader(subjects=list(range(1, 6)))

    # Load data
    X, y = loader.load_imagery_data()

    print(f"\nFinal data shape: {X.shape}")
    print(f"Labels shape: {y.shape}")
    print(f"Class distribution: rest={np.sum(y==0)}, imagery={np.sum(y==1)}")

    return loader, X, y


if __name__ == "__main__":
    loader, X, y = main()
