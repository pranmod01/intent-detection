"""
Enhanced EEG Analysis with ERD/ERS Calculation

This script extends the basic analysis with:
- Baseline normalization
- Event-related desynchronization (ERD) / synchronization (ERS) calculation
- Channel-specific comparisons
- Enhanced statistical reporting
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import signal, stats
from pathlib import Path
from eeg_analysis import EEGAnalyzer


class EnhancedEEGAnalyzer(EEGAnalyzer):
    """Extended analyzer with ERD/ERS calculation"""

    def compute_erd_ers(self, task_power, baseline_power):
        """
        Compute Event-Related Desynchronization/Synchronization

        ERD/ERS (%) = ((task_power - baseline_power) / baseline_power) * 100

        Negative values = ERD (desynchronization)
        Positive values = ERS (synchronization)

        Parameters:
        -----------
        task_power : ndarray
            Power during task condition
        baseline_power : ndarray or float
            Power during baseline/rest

        Returns:
        --------
        erd_ers : ndarray
            ERD/ERS values in percentage
        """
        erd_ers = ((task_power - baseline_power) / baseline_power) * 100
        return erd_ers

    def compare_conditions_detailed(self):
        """
        Detailed comparison across conditions with channel-specific analysis
        """
        print("\n" + "="*70)
        print("DETAILED CONDITION COMPARISON")
        print("="*70)

        categories = self.aggregate_by_category()

        # For each frequency band
        for band_name, band_label in [('mu_power', 'Mu (8-13 Hz)'),
                                       ('beta_power', 'Beta (13-30 Hz)')]:
            print(f"\n{band_label}")
            print("-"*70)

            # For each channel
            for ch_idx, ch_name in enumerate(self.ch_names):
                print(f"\n  {ch_name}:")

                channel_data = {}
                for cat_name, cat_data in categories.items():
                    if len(cat_data) > 0:
                        # Extract power for this channel
                        powers = []
                        for item in cat_data:
                            powers.extend(item[band_name][:, ch_idx])

                        channel_data[cat_name] = powers

                        mean_power = np.mean(powers)
                        std_power = np.std(powers)
                        print(f"    {cat_name:20s}: {mean_power:.2e} ± {std_power:.2e} (n={len(powers)})")

                # Statistical test across conditions for this channel
                if len(channel_data) >= 2:
                    groups = list(channel_data.values())
                    if all(len(g) > 0 for g in groups):
                        f_stat, p_val = stats.f_oneway(*groups)
                        sig_marker = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "ns"
                        print(f"    ANOVA: F={f_stat:.3f}, p={p_val:.4f} {sig_marker}")

    def plot_channel_comparison(self, output_dir='results'):
        """
        Create detailed channel-by-channel comparison plots
        """
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)

        categories = self.aggregate_by_category()

        # Separate plot for each frequency band
        for band_name, band_label in [('mu_power', 'Mu Band (8-13 Hz)'),
                                       ('beta_power', 'Beta Band (13-30 Hz)')]:

            fig, axes = plt.subplots(2, 2, figsize=(14, 10))
            axes = axes.flatten()

            for ch_idx, ch_name in enumerate(self.ch_names):
                ax = axes[ch_idx]

                data_to_plot = []
                labels = []
                colors_list = []

                color_map = {
                    'voluntary_left': '#FF6B6B',
                    'voluntary_right': '#4ECDC4',
                    'motor_imagery': '#45B7D1',
                    'ems': '#FFA07A',
                    'passive': '#98D8C8'
                }

                for cat_name, cat_data in categories.items():
                    if len(cat_data) > 0:
                        powers = []
                        for item in cat_data:
                            powers.extend(item[band_name][:, ch_idx])

                        data_to_plot.append(powers)
                        labels.append(cat_name.replace('_', '\n').title())
                        colors_list.append(color_map.get(cat_name, '#999999'))

                bp = ax.boxplot(data_to_plot, labels=labels, patch_artist=True,
                               showfliers=True, notch=True)

                for patch, color in zip(bp['boxes'], colors_list):
                    patch.set_facecolor(color)
                    patch.set_alpha(0.7)

                ax.set_ylabel('Power (V²/Hz)', fontsize=10)
                ax.set_title(f'{ch_name}', fontsize=12, fontweight='bold')
                ax.tick_params(axis='x', rotation=0, labelsize=8)
                ax.grid(axis='y', alpha=0.3)

                # Add sample sizes
                for i, n in enumerate([len(d) for d in data_to_plot]):
                    ax.text(i+1, ax.get_ylim()[0], f'n={n}',
                           ha='center', va='top', fontsize=7)

            fig.suptitle(f'{band_label} - Power Comparison Across Conditions',
                        fontsize=14, fontweight='bold', y=0.995)
            plt.tight_layout()

            filename = f"{band_name.split('_')[0]}_channel_comparison.png"
            plt.savefig(output_path / filename, dpi=300, bbox_inches='tight')
            print(f"Saved: {output_path / filename}")

        plt.close('all')

    def create_summary_table(self, output_dir='results'):
        """
        Create a comprehensive summary table
        """
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)

        categories = self.aggregate_by_category()

        summary_rows = []

        for cat_name, cat_data in categories.items():
            if len(cat_data) > 0:
                # Overall statistics
                mu_powers = [np.mean(item['mu_power']) for item in cat_data]
                beta_powers = [np.mean(item['beta_power']) for item in cat_data]

                # Channel-specific statistics
                for ch_idx, ch_name in enumerate(self.ch_names):
                    mu_ch = [np.mean(item['mu_power'][:, ch_idx]) for item in cat_data]
                    beta_ch = [np.mean(item['beta_power'][:, ch_idx]) for item in cat_data]

                    summary_rows.append({
                        'Condition': cat_name.replace('_', ' ').title(),
                        'Channel': ch_name,
                        'N': len(cat_data),
                        'Mu Mean': f"{np.mean(mu_ch):.2e}",
                        'Mu Std': f"{np.std(mu_ch):.2e}",
                        'Beta Mean': f"{np.mean(beta_ch):.2e}",
                        'Beta Std': f"{np.std(beta_ch):.2e}"
                    })

        df = pd.DataFrame(summary_rows)
        csv_path = output_path / 'summary_statistics.csv'
        df.to_csv(csv_path, index=False)
        print(f"\nSaved: {csv_path}")

        return df

    def plot_topographic_comparison(self, output_dir='results'):
        """
        Create simple topographic-style visualization
        """
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)

        categories = self.aggregate_by_category()

        # Define approximate channel positions (simplified)
        positions = {
            'C3': (-1, 0),
            'C4': (1, 0),
            'Cz': (0, 0),
            'CP3': (-0.7, -1)
        }

        for band_name, band_label in [('mu_power', 'Mu'), ('beta_power', 'Beta')]:
            fig, axes = plt.subplots(1, len(categories), figsize=(16, 4))

            for ax, (cat_name, cat_data) in zip(axes, categories.items()):
                if len(cat_data) == 0:
                    ax.axis('off')
                    continue

                # Compute mean power for each channel
                channel_means = []
                for ch_idx in range(len(self.ch_names)):
                    powers = []
                    for item in cat_data:
                        powers.extend(item[band_name][:, ch_idx])
                    channel_means.append(np.mean(powers))

                # Normalize for visualization
                channel_means = np.array(channel_means)
                if np.max(channel_means) > 0:
                    channel_means_norm = channel_means / np.max(channel_means)
                else:
                    channel_means_norm = channel_means

                # Plot
                for ch_idx, ch_name in enumerate(self.ch_names):
                    x, y = positions[ch_name]
                    size = max(100, channel_means_norm[ch_idx] * 1000)
                    ax.scatter(x, y, s=size, c=[channel_means[ch_idx]],
                             cmap='RdYlBu_r', alpha=0.7, edgecolors='black')
                    ax.text(x, y-0.3, ch_name, ha='center', fontsize=10)

                ax.set_xlim(-2, 2)
                ax.set_ylim(-2, 1)
                ax.set_aspect('equal')
                ax.axis('off')
                ax.set_title(cat_name.replace('_', ' ').title(), fontsize=11)

            fig.suptitle(f'{band_label} Band Power - Topographic View',
                        fontsize=13, fontweight='bold')
            plt.tight_layout()

            filename = f"{band_name.split('_')[0]}_topography.png"
            plt.savefig(output_path / filename, dpi=300, bbox_inches='tight')
            print(f"Saved: {output_path / filename}")

        plt.close('all')


def main():
    """Run enhanced analysis"""
    print("="*70)
    print("ENHANCED EEG ANALYSIS")
    print("="*70)

    # Initialize enhanced analyzer
    analyzer = EnhancedEEGAnalyzer(
        data_dir='../data/cleaned',
        metadata_path='../data/experiment_metadata.json'
    )

    # Run basic analysis
    print("\nRunning basic analysis pipeline...")
    results = analyzer.analyze_all_subjects()

    # Run enhanced analyses
    print("\n" + "="*70)
    print("ENHANCED ANALYSES")
    print("="*70)

    # Detailed condition comparison
    analyzer.compare_conditions_detailed()

    # Generate enhanced visualizations
    print("\n" + "="*70)
    print("GENERATING VISUALIZATIONS")
    print("="*70)

    analyzer.plot_channel_comparison()
    analyzer.plot_topographic_comparison()

    # Create summary table
    df = analyzer.create_summary_table()
    print("\nSummary Statistics Table:")
    print(df.to_string(index=False))

    print("\n" + "="*70)
    print("ENHANCED ANALYSIS COMPLETE")
    print("="*70)

    return analyzer


if __name__ == "__main__":
    analyzer = main()
