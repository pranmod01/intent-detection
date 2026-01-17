"""
Domain Analysis and Visualization

Analyzes the feature space relationships between:
- PhysioNet motor imagery data
- Local intent conditions (voluntary, imagery)
- Local non-intent conditions (EMS, passive)

Uses t-SNE/UMAP to visualize how conditions cluster in feature space.
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, Tuple, List
import json

# Add project paths
sys.path.insert(0, str(Path(__file__).parent.parent / 'phase1'))
sys.path.insert(0, str(Path(__file__).parent.parent / 'phase2'))

from physionet_loader import PhysioNetLoader
from csp_lda_baseline import CSPLDAClassifier
from data_organizer import IntentDataOrganizer


def load_physionet_data(subjects=None, runs=None):
    """Helper function to load PhysioNet data using PhysioNetLoader."""
    loader = PhysioNetLoader(subjects=subjects or list(range(1, 11)))
    X, y = loader.load_imagery_data()
    ch_names = loader.TARGET_CHANNELS
    return X, y, ch_names

# Dimensionality reduction
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
try:
    import umap
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False


def extract_csp_features(
    X: np.ndarray,
    clf: CSPLDAClassifier = None,
    n_components: int = 4
) -> np.ndarray:
    """
    Extract CSP features from EEG data.

    Parameters
    ----------
    X : ndarray, shape (n_samples, n_channels, n_times)
    clf : CSPLDAClassifier, optional
        If provided, use its CSP filters. Otherwise fit new.

    Returns
    -------
    features : ndarray, shape (n_samples, n_components)
    """
    if clf is not None and clf.is_fitted:
        return clf.csp.transform(X)
    else:
        from mne.decoding import CSP
        from scipy.signal import butter, filtfilt

        # Bandpass filter to mu/beta
        b, a = butter(4, [8, 30], btype='band', fs=200)
        X_filt = np.array([filtfilt(b, a, x, axis=-1) for x in X])

        # Need labels for CSP fitting - use dummy binary labels
        # This is a simplified version for visualization
        csp = CSP(n_components=n_components, reg=None, log=True)
        y_dummy = np.array([0, 1] * (len(X) // 2 + 1))[:len(X)]
        csp.fit(X_filt, y_dummy)
        return csp.transform(X_filt)


def extract_frequency_features(X: np.ndarray, sfreq: int = 200) -> np.ndarray:
    """
    Extract frequency band power features.

    Returns power in delta, theta, alpha, mu, beta, gamma bands.
    """
    from scipy.signal import welch

    bands = {
        'delta': (1, 4),
        'theta': (4, 8),
        'alpha': (8, 13),
        'mu': (8, 13),
        'beta': (13, 30),
        'gamma': (30, 50)
    }

    features = []

    for epoch in X:
        epoch_features = []
        for ch_idx in range(epoch.shape[0]):
            freqs, psd = welch(epoch[ch_idx], fs=sfreq, nperseg=min(256, epoch.shape[1]))

            for band_name, (fmin, fmax) in bands.items():
                band_mask = (freqs >= fmin) & (freqs <= fmax)
                band_power = np.mean(psd[band_mask])
                epoch_features.append(band_power)

        features.append(epoch_features)

    return np.array(features)


def create_domain_visualization(
    X_physionet: np.ndarray,
    X_local: np.ndarray,
    y_local_conditions: np.ndarray,
    condition_names: Dict[int, str],
    output_path: str = None,
    method: str = 'tsne'
):
    """
    Create t-SNE/UMAP visualization of domain relationships.

    Parameters
    ----------
    X_physionet : ndarray
        PhysioNet EEG data
    X_local : ndarray
        Local EEG data
    y_local_conditions : ndarray
        Condition labels for local data (0-5 for different conditions)
    condition_names : dict
        Mapping from condition index to name
    method : str
        'tsne', 'umap', or 'pca'
    """
    print(f"\nExtracting frequency features...")

    # Extract features
    feat_physionet = extract_frequency_features(X_physionet)
    feat_local = extract_frequency_features(X_local)

    print(f"PhysioNet features: {feat_physionet.shape}")
    print(f"Local features: {feat_local.shape}")

    # Combine for dimensionality reduction
    X_combined = np.vstack([feat_physionet, feat_local])

    # Create labels
    n_physionet = len(feat_physionet)
    labels = ['PhysioNet'] * n_physionet

    for i, cond_idx in enumerate(y_local_conditions):
        labels.append(condition_names.get(cond_idx, f'Condition {cond_idx}'))

    # Normalize features
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_combined)

    # Dimensionality reduction
    print(f"\nRunning {method.upper()}...")

    if method == 'tsne':
        reducer = TSNE(n_components=2, perplexity=30, random_state=42, max_iter=1000)
        X_embedded = reducer.fit_transform(X_scaled)
    elif method == 'umap' and UMAP_AVAILABLE:
        reducer = umap.UMAP(n_components=2, random_state=42)
        X_embedded = reducer.fit_transform(X_scaled)
    else:
        reducer = PCA(n_components=2)
        X_embedded = reducer.fit_transform(X_scaled)

    # Plotting
    fig, ax = plt.subplots(figsize=(12, 10))

    # Color scheme
    colors = {
        'PhysioNet': '#95a5a6',      # Gray
        'voluntary': '#27ae60',       # Green
        'imagery': '#2ecc71',         # Light green
        'ems_short': '#e74c3c',       # Red
        'ems_long': '#c0392b',        # Dark red
        'passive': '#e67e22'          # Orange
    }

    markers = {
        'PhysioNet': 'o',
        'voluntary': '^',
        'imagery': 's',
        'ems_short': 'v',
        'ems_long': 'd',
        'passive': 'p'
    }

    # Plot each condition
    unique_labels = list(set(labels))

    for label in unique_labels:
        mask = [l == label for l in labels]
        color = colors.get(label, '#333333')
        marker = markers.get(label, 'o')
        alpha = 0.3 if label == 'PhysioNet' else 0.7
        size = 30 if label == 'PhysioNet' else 60

        ax.scatter(
            X_embedded[mask, 0],
            X_embedded[mask, 1],
            c=color,
            marker=marker,
            s=size,
            alpha=alpha,
            label=label,
            edgecolors='white',
            linewidth=0.5
        )

    ax.set_xlabel(f'{method.upper()} Dimension 1', fontsize=12)
    ax.set_ylabel(f'{method.upper()} Dimension 2', fontsize=12)
    ax.set_title('Domain Analysis: EEG Feature Space\n'
                 'Showing relationships between PhysioNet and local conditions',
                 fontsize=14)
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Figure saved to {output_path}")

    plt.show()

    return X_embedded, labels


def analyze_domain_distances(
    X_physionet: np.ndarray,
    X_local: np.ndarray,
    y_local_conditions: np.ndarray,
    condition_names: Dict[int, str]
) -> Dict:
    """
    Compute distance metrics between domains.
    """
    from scipy.spatial.distance import cdist

    # Extract features
    feat_physionet = extract_frequency_features(X_physionet)
    feat_local = extract_frequency_features(X_local)

    # Normalize
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    scaler.fit(np.vstack([feat_physionet, feat_local]))
    feat_physionet_norm = scaler.transform(feat_physionet)
    feat_local_norm = scaler.transform(feat_local)

    # Compute centroid of PhysioNet data
    centroid_physionet = np.mean(feat_physionet_norm, axis=0)

    # Compute distances for each condition
    distances = {}

    for cond_idx in np.unique(y_local_conditions):
        cond_name = condition_names.get(cond_idx, f'Condition {cond_idx}')
        mask = y_local_conditions == cond_idx
        feat_cond = feat_local_norm[mask]

        # Centroid of this condition
        centroid_cond = np.mean(feat_cond, axis=0)

        # Distance from PhysioNet centroid
        dist_to_physionet = np.linalg.norm(centroid_cond - centroid_physionet)

        # Average within-condition variance
        within_var = np.mean(np.var(feat_cond, axis=0))

        distances[cond_name] = {
            'distance_to_physionet': float(dist_to_physionet),
            'within_variance': float(within_var),
            'n_samples': int(np.sum(mask))
        }

    return distances


def main():
    """Run domain analysis."""
    print("="*60)
    print("Phase 3: Domain Analysis and Visualization")
    print("="*60)

    # Load PhysioNet data
    print("\n[1/4] Loading PhysioNet data...")
    X_physionet, y_physionet, ch_names = load_physionet_data(
        subjects=list(range(1, 11)),
        runs=[4, 8, 12]
    )

    # Use only imagery samples for clearer visualization
    X_physionet_imagery = X_physionet[y_physionet == 1]
    # Subsample if too many
    if len(X_physionet_imagery) > 500:
        idx = np.random.choice(len(X_physionet_imagery), 500, replace=False)
        X_physionet_imagery = X_physionet_imagery[idx]

    print(f"PhysioNet imagery samples: {len(X_physionet_imagery)}")

    # Load local data with condition labels
    print("\n[2/4] Loading local multi-condition data...")
    organizer = IntentDataOrganizer(use_all_channels=False)
    data = organizer.load_all_data()

    X_local = data['X']
    y_conditions = data['y_multiclass']

    # Align sample counts - trim PhysioNet to match local data
    target_samples = X_local.shape[2]
    if X_physionet_imagery.shape[2] > target_samples:
        X_physionet_imagery = X_physionet_imagery[:, :, :target_samples]
        print(f"Trimmed PhysioNet to {target_samples} samples")

    condition_names = {
        0: 'voluntary',
        1: 'imagery',
        2: 'ems_short',
        3: 'ems_long',
        4: 'passive'
    }

    print(f"Local samples: {len(X_local)}")
    for idx, name in condition_names.items():
        count = np.sum(y_conditions == idx)
        print(f"  {name}: {count}")

    # Create visualizations
    print("\n[3/4] Creating domain visualizations...")

    # t-SNE visualization
    X_embedded, labels = create_domain_visualization(
        X_physionet_imagery,
        X_local,
        y_conditions,
        condition_names,
        output_path="results/figures/domain_tsne.png",
        method='tsne'
    )

    # PCA visualization
    create_domain_visualization(
        X_physionet_imagery,
        X_local,
        y_conditions,
        condition_names,
        output_path="results/figures/domain_pca.png",
        method='pca'
    )

    # Analyze distances
    print("\n[4/4] Computing domain distances...")
    distances = analyze_domain_distances(
        X_physionet_imagery,
        X_local,
        y_conditions,
        condition_names
    )

    print("\nDomain Distance Analysis:")
    print("-" * 60)
    print(f"{'Condition':<15} {'Dist to PhysioNet':>18} {'Within Variance':>16} {'N':>6}")
    print("-" * 60)

    for cond_name, metrics in distances.items():
        print(f"{cond_name:<15} {metrics['distance_to_physionet']:>18.3f} {metrics['within_variance']:>16.4f} {metrics['n_samples']:>6}")

    # Analysis insight
    intent_conds = ['voluntary', 'imagery']
    non_intent_conds = ['ems_short', 'ems_long', 'passive']

    avg_intent_dist = np.mean([distances[c]['distance_to_physionet'] for c in intent_conds if c in distances])
    avg_nonintent_dist = np.mean([distances[c]['distance_to_physionet'] for c in non_intent_conds if c in distances])

    print("\n" + "="*60)
    print("KEY INSIGHT")
    print("="*60)
    print(f"""
Average distance from PhysioNet:
  Intent conditions (voluntary, imagery): {avg_intent_dist:.3f}
  Non-intent conditions (EMS, passive):   {avg_nonintent_dist:.3f}

{'Intent conditions are CLOSER to PhysioNet' if avg_intent_dist < avg_nonintent_dist else 'Non-intent conditions are CLOSER to PhysioNet'}
This explains why PhysioNet-trained classifiers {'correctly' if avg_intent_dist < avg_nonintent_dist else 'incorrectly'}
identify intent conditions but fail on non-intent conditions.
""")

    # Save results
    results_path = Path("results/domain_analysis.json")
    results_path.parent.mkdir(parents=True, exist_ok=True)

    with open(results_path, 'w') as f:
        json.dump({
            'distances': distances,
            'avg_intent_distance': float(avg_intent_dist),
            'avg_nonintent_distance': float(avg_nonintent_dist)
        }, f, indent=2)

    print(f"\nResults saved to {results_path}")

    return distances


if __name__ == "__main__":
    distances = main()
