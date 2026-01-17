"""
EEG/EMG Analysis Pipeline for Voluntary vs. Externally-Evoked Motor Activation Study

This script analyzes EEG data to compare cortical signatures across different conditions:
- Voluntary motor actions (left/right hand)
- Motor imagery
- EMS-evoked movements
- Passive movements

Analysis focuses on mu (8-13 Hz) and beta (13-30 Hz) band power changes.
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import signal, stats
from scipy.stats import ttest_rel, f_oneway
import mne
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Set plotting style
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 8)


class EEGAnalyzer:
    """Class for EEG data analysis and visualization"""

    def __init__(self, data_dir='data/cleaned', metadata_path='data/experiment_metadata.json'):
        self.data_dir = Path(data_dir)
        self.metadata_path = Path(metadata_path)
        self.sfreq = 200  # Hz
        self.ch_names = ['C3', 'C4', 'Cz', 'CP3']
        self.ch_types = ['eeg', 'eeg', 'eeg', 'eeg']

        # Frequency bands of interest
        self.mu_band = (8, 13)
        self.beta_band = (13, 30)

        # Load metadata
        with open(self.metadata_path, 'r') as f:
            self.metadata = json.load(f)

        self.results = {}

    def load_csv_data(self, filepath):
        """Load cleaned CSV data and convert to MNE format"""
        df = pd.read_csv(filepath, comment='#')

        # Extract EEG channels (assuming columns: sample_index, eeg_ch0-3, timestamp)
        eeg_data = df[['eeg_ch0', 'eeg_ch1', 'eeg_ch2', 'eeg_ch3']].values.T

        # Convert from microvolts to volts for MNE
        eeg_data = eeg_data * 1e-6

        # Create MNE info structure
        info = mne.create_info(
            ch_names=self.ch_names,
            sfreq=self.sfreq,
            ch_types=self.ch_types
        )

        # Set montage for standard 10-20 positions
        info.set_montage('standard_1020')

        # Create Raw object
        raw = mne.io.RawArray(eeg_data, info)

        return raw

    def preprocess_raw(self, raw, l_freq=1.0, h_freq=50.0, notch_freq=60.0):
        """Apply preprocessing: filtering and artifact removal"""
        # Make a copy to avoid modifying original
        raw_filt = raw.copy()

        # Apply bandpass filter
        raw_filt.filter(l_freq=l_freq, h_freq=h_freq, fir_design='firwin')

        # Apply notch filter for line noise
        raw_filt.notch_filter(freqs=notch_freq, fir_design='firwin')

        return raw_filt

    def create_epochs(self, raw, epoch_duration=2.0, overlap=0.5):
        """
        Create fixed-length epochs from continuous data

        Parameters:
        -----------
        raw : mne.io.Raw
            Preprocessed raw data
        epoch_duration : float
            Duration of each epoch in seconds
        overlap : float
            Overlap between epochs (0-1)
        """
        # Create fixed-length epochs
        events = mne.make_fixed_length_events(
            raw,
            duration=epoch_duration,
            overlap=overlap
        )

        epochs = mne.Epochs(
            raw,
            events,
            tmin=0,
            tmax=epoch_duration,
            baseline=None,
            preload=True,
            verbose=False
        )

        return epochs

    def compute_band_power(self, epochs, fmin, fmax):
        """
        Compute average power in a frequency band using Welch's method

        Returns:
        --------
        power : ndarray, shape (n_epochs, n_channels)
        """
        # Get data: (n_epochs, n_channels, n_times)
        data = epochs.get_data()

        n_epochs, n_channels, n_times = data.shape
        power = np.zeros((n_epochs, n_channels))

        for epoch_idx in range(n_epochs):
            for ch_idx in range(n_channels):
                # Compute power spectral density using Welch's method
                freqs, psd = signal.welch(
                    data[epoch_idx, ch_idx, :],
                    fs=self.sfreq,
                    nperseg=min(256, n_times)
                )

                # Extract power in frequency band
                freq_mask = (freqs >= fmin) & (freqs <= fmax)
                power[epoch_idx, ch_idx] = np.mean(psd[freq_mask])

        return power

    def compute_time_frequency(self, epochs, freqs=None):
        """
        Compute time-frequency representation using morlet wavelets

        Returns:
        --------
        power : mne.time_frequency.AverageTFR
        """
        if freqs is None:
            freqs = np.arange(4, 40, 1)

        n_cycles = freqs / 2.0  # Number of cycles for each frequency

        # Compute power
        power = mne.time_frequency.tfr_morlet(
            epochs,
            freqs=freqs,
            n_cycles=n_cycles,
            return_itc=False,
            average=True,
            verbose=False
        )

        return power

    def analyze_condition(self, filepath, condition_name):
        """Analyze a single condition file"""
        print(f"Analyzing: {condition_name}")

        # Load and preprocess
        raw = self.load_csv_data(filepath)
        raw_filt = self.preprocess_raw(raw)

        # Create epochs
        epochs = self.create_epochs(raw_filt, epoch_duration=2.0, overlap=0.5)

        # Compute band powers
        mu_power = self.compute_band_power(epochs, *self.mu_band)
        beta_power = self.compute_band_power(epochs, *self.beta_band)

        # Compute time-frequency
        tfr = self.compute_time_frequency(epochs)

        return {
            'condition': condition_name,
            'epochs': epochs,
            'mu_power': mu_power,
            'beta_power': beta_power,
            'tfr': tfr,
            'n_epochs': len(epochs)
        }

    def analyze_all_subjects(self):
        """Analyze all subjects and conditions"""
        all_results = {}

        for subject_id, subject_info in self.metadata['subjects'].items():
            print(f"\n{'='*60}")
            print(f"Processing {subject_id}")
            print(f"{'='*60}")

            subject_results = {}
            session_dir = self.data_dir / subject_info['session_id']

            for filename, condition in subject_info['conditions'].items():
                filepath = session_dir / filename

                if filepath.exists():
                    result = self.analyze_condition(filepath, condition)
                    subject_results[condition] = result
                else:
                    print(f"Warning: File not found: {filepath}")

            all_results[subject_id] = subject_results

        self.results = all_results
        return all_results

    def aggregate_by_category(self):
        """
        Aggregate results by condition category across subjects

        Returns:
        --------
        aggregated : dict
            Dictionary with keys as categories and values containing:
            - mu_power: list of power arrays
            - beta_power: list of power arrays
            - subjects: list of subject IDs
        """
        categories = {
            'voluntary_left': [],
            'voluntary_right': [],
            'motor_imagery': [],
            'ems': [],
            'passive': []
        }

        for subject_id, subject_data in self.results.items():
            for condition, data in subject_data.items():
                if 'Left hand voluntary' in condition:
                    categories['voluntary_left'].append({
                        'subject': subject_id,
                        'mu_power': data['mu_power'],
                        'beta_power': data['beta_power']
                    })
                elif 'Right hand voluntary' in condition:
                    categories['voluntary_right'].append({
                        'subject': subject_id,
                        'mu_power': data['mu_power'],
                        'beta_power': data['beta_power']
                    })
                elif 'Motor imagery' in condition:
                    categories['motor_imagery'].append({
                        'subject': subject_id,
                        'mu_power': data['mu_power'],
                        'beta_power': data['beta_power']
                    })
                elif 'EMS' in condition:
                    categories['ems'].append({
                        'subject': subject_id,
                        'mu_power': data['mu_power'],
                        'beta_power': data['beta_power']
                    })
                elif 'Experimenter' in condition:
                    categories['passive'].append({
                        'subject': subject_id,
                        'mu_power': data['mu_power'],
                        'beta_power': data['beta_power']
                    })

        return categories

    def statistical_analysis(self):
        """
        Perform statistical tests across conditions

        Tests:
        - Paired t-tests between voluntary and EMS conditions
        - ANOVA across all conditions
        - Channel-specific analysis
        """
        print("\n" + "="*60)
        print("STATISTICAL ANALYSIS")
        print("="*60)

        categories = self.aggregate_by_category()
        stats_results = {}

        # Analyze mu and beta separately
        for band_name in ['mu_power', 'beta_power']:
            print(f"\n{band_name.upper()} ANALYSIS:")
            print("-" * 60)

            # Extract mean power across epochs for each category
            category_powers = {}
            for cat_name, cat_data in categories.items():
                if len(cat_data) > 0:
                    # Average across epochs and channels for each subject
                    powers = [np.mean(item[band_name]) for item in cat_data]
                    category_powers[cat_name] = powers
                    print(f"{cat_name:20s}: mean={np.mean(powers):.2e}, std={np.std(powers):.2e}, n={len(powers)}")

            # Paired t-tests
            print("\nPaired t-tests:")
            if len(category_powers.get('voluntary_right', [])) > 0 and len(category_powers.get('ems', [])) > 0:
                # For paired test, we need same number of observations
                vol = category_powers['voluntary_right'][:min(len(category_powers['voluntary_right']), len(category_powers['ems']))]
                ems = category_powers['ems'][:len(vol)]

                if len(vol) > 1:
                    t_stat, p_val = ttest_rel(vol, ems)
                    print(f"  Voluntary Right vs EMS: t={t_stat:.3f}, p={p_val:.4f}")
                    stats_results[f'{band_name}_voluntary_vs_ems'] = {'t': t_stat, 'p': p_val}

            # ANOVA across multiple conditions
            groups = [v for v in category_powers.values() if len(v) > 0]
            if len(groups) >= 3:
                f_stat, p_val = f_oneway(*groups)
                print(f"\nANOVA across conditions: F={f_stat:.3f}, p={p_val:.4f}")
                stats_results[f'{band_name}_anova'] = {'F': f_stat, 'p': p_val}

        return stats_results

    def plot_summary(self, output_dir='results'):
        """Generate summary visualizations"""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)

        categories = self.aggregate_by_category()

        # Plot 1: Band power comparison across conditions
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))

        for band_idx, (band_name, band_label) in enumerate([('mu_power', 'Mu (8-13 Hz)'),
                                                              ('beta_power', 'Beta (13-30 Hz)')]):
            for ch_idx, ch_name in enumerate(self.ch_names[:2]):  # Plot C3 and C4
                ax = axes[band_idx, ch_idx]

                data_to_plot = []
                labels = []

                for cat_name, cat_data in categories.items():
                    if len(cat_data) > 0:
                        # Extract power for specific channel across all epochs and subjects
                        powers = []
                        for item in cat_data:
                            powers.extend(item[band_name][:, self.ch_names.index(ch_name)])

                        data_to_plot.append(powers)
                        labels.append(cat_name.replace('_', ' ').title())

                bp = ax.boxplot(data_to_plot, labels=labels, patch_artist=True)

                # Color the boxes
                colors = plt.cm.Set3(np.linspace(0, 1, len(data_to_plot)))
                for patch, color in zip(bp['boxes'], colors):
                    patch.set_facecolor(color)

                ax.set_ylabel('Power (V²/Hz)')
                ax.set_title(f'{band_label} - {ch_name}')
                ax.tick_params(axis='x', rotation=45)
                plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

        plt.tight_layout()
        plt.savefig(output_path / 'band_power_comparison.png', dpi=300, bbox_inches='tight')
        print(f"\nSaved: {output_path / 'band_power_comparison.png'}")

        # Plot 2: Time-frequency plots for representative conditions
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes = axes.flatten()

        plot_idx = 0
        for subject_id, subject_data in self.results.items():
            for condition, data in subject_data.items():
                if plot_idx >= 4:
                    break

                # Plot average TFR for motor cortex channel (C3 or C4)
                tfr = data['tfr']
                ch_idx = 0  # C3

                # Plot
                im = axes[plot_idx].imshow(
                    tfr.data[ch_idx, :, :],
                    aspect='auto',
                    origin='lower',
                    extent=[tfr.times[0], tfr.times[-1], tfr.freqs[0], tfr.freqs[-1]],
                    cmap='RdBu_r'
                )
                axes[plot_idx].set_xlabel('Time (s)')
                axes[plot_idx].set_ylabel('Frequency (Hz)')
                axes[plot_idx].set_title(f'{condition[:30]}...\n{self.ch_names[ch_idx]}', fontsize=9)
                plt.colorbar(im, ax=axes[plot_idx], label='Power')

                # Mark mu and beta bands
                axes[plot_idx].axhline(y=self.mu_band[0], color='white', linestyle='--', alpha=0.5)
                axes[plot_idx].axhline(y=self.mu_band[1], color='white', linestyle='--', alpha=0.5)
                axes[plot_idx].axhline(y=self.beta_band[0], color='yellow', linestyle='--', alpha=0.5)
                axes[plot_idx].axhline(y=self.beta_band[1], color='yellow', linestyle='--', alpha=0.5)

                plot_idx += 1

        plt.tight_layout()
        plt.savefig(output_path / 'time_frequency_plots.png', dpi=300, bbox_inches='tight')
        print(f"Saved: {output_path / 'time_frequency_plots.png'}")

        plt.close('all')


def main():
    """Main analysis pipeline"""
    print("="*60)
    print("EEG ANALYSIS PIPELINE")
    print("Voluntary vs. Externally-Evoked Motor Activation")
    print("="*60)

    # Initialize analyzer
    analyzer = EEGAnalyzer()

    # Analyze all data
    results = analyzer.analyze_all_subjects()

    # Perform statistical analysis
    stats = analyzer.statistical_analysis()

    # Generate visualizations
    print("\nGenerating visualizations...")
    analyzer.plot_summary()

    print("\n" + "="*60)
    print("ANALYSIS COMPLETE")
    print("="*60)
    print("\nResults saved to 'results/' directory")
    print("- band_power_comparison.png: Power comparison across conditions")
    print("- time_frequency_plots.png: Time-frequency representations")

    return analyzer, results, stats


if __name__ == "__main__":
    analyzer, results, stats = main()
