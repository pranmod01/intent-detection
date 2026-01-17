"""
CSP + LDA Baseline Classifier

Trains a standard Motor Imagery BCI using Common Spatial Patterns (CSP)
and Linear Discriminant Analysis (LDA) on PhysioNet data.

This serves as our baseline to demonstrate the intent detection problem.
"""

import numpy as np
import pickle
from pathlib import Path
from typing import Tuple, Dict, Any
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

from mne.decoding import CSP

from physionet_loader import PhysioNetLoader


class CSPLDAClassifier:
    """CSP + LDA classifier for motor imagery BCI."""

    def __init__(
        self,
        n_components: int = 6,
        reg: str = None,
        log: bool = True,
        freq_band: Tuple[float, float] = (8, 30)
    ):
        """
        Initialize CSP + LDA classifier.

        Parameters
        ----------
        n_components : int
            Number of CSP components (spatial filters)
        reg : str or None
            Regularization for CSP. None, 'ledoit_wolf', 'oas', or float
        log : bool
            Whether to log-transform CSP features
        freq_band : tuple
            Frequency band for filtering (mu + beta: 8-30 Hz)
        """
        self.n_components = n_components
        self.freq_band = freq_band

        # Build pipeline
        self.csp = CSP(n_components=n_components, reg=reg, log=log)
        self.lda = LinearDiscriminantAnalysis()

        self.pipeline = Pipeline([
            ('csp', self.csp),
            ('lda', self.lda)
        ])

        self.is_fitted = False
        self.training_metrics = {}

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'CSPLDAClassifier':
        """
        Fit the CSP + LDA classifier.

        Parameters
        ----------
        X : ndarray, shape (n_epochs, n_channels, n_times)
            Training EEG data
        y : ndarray, shape (n_epochs,)
            Training labels

        Returns
        -------
        self : CSPLDAClassifier
            Fitted classifier
        """
        print(f"Training CSP+LDA classifier...")
        print(f"  Input shape: {X.shape}")
        print(f"  Classes: {np.unique(y)}")

        self.pipeline.fit(X, y)
        self.is_fitted = True

        # Store training accuracy
        y_pred = self.pipeline.predict(X)
        self.training_metrics['train_accuracy'] = accuracy_score(y, y_pred)
        print(f"  Training accuracy: {self.training_metrics['train_accuracy']:.2%}")

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels."""
        if not self.is_fitted:
            raise RuntimeError("Classifier must be fitted before predicting")
        return self.pipeline.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities."""
        if not self.is_fitted:
            raise RuntimeError("Classifier must be fitted before predicting")
        return self.pipeline.predict_proba(X)

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Return accuracy score."""
        return self.pipeline.score(X, y)

    def cross_validate(
        self,
        X: np.ndarray,
        y: np.ndarray,
        cv: int = 5
    ) -> Dict[str, Any]:
        """
        Perform cross-validation.

        Parameters
        ----------
        X : ndarray
            EEG data
        y : ndarray
            Labels
        cv : int
            Number of folds

        Returns
        -------
        results : dict
            Cross-validation results
        """
        print(f"\nPerforming {cv}-fold cross-validation...")

        skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
        scores = cross_val_score(self.pipeline, X, y, cv=skf, scoring='accuracy')

        results = {
            'cv_scores': scores,
            'cv_mean': np.mean(scores),
            'cv_std': np.std(scores)
        }

        print(f"  CV Accuracy: {results['cv_mean']:.2%} (+/- {results['cv_std']:.2%})")
        print(f"  Fold scores: {[f'{s:.2%}' for s in scores]}")

        return results

    def get_csp_patterns(self) -> np.ndarray:
        """
        Get CSP spatial patterns for visualization.

        Returns
        -------
        patterns : ndarray, shape (n_components, n_channels)
            CSP patterns (inverse of filters)
        """
        if not self.is_fitted:
            raise RuntimeError("Classifier must be fitted first")
        return self.csp.patterns_

    def get_csp_filters(self) -> np.ndarray:
        """
        Get CSP spatial filters.

        Returns
        -------
        filters : ndarray, shape (n_components, n_channels)
            CSP filters
        """
        if not self.is_fitted:
            raise RuntimeError("Classifier must be fitted first")
        return self.csp.filters_

    def save(self, filepath: str):
        """Save the trained model."""
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump({
                'pipeline': self.pipeline,
                'csp': self.csp,
                'lda': self.lda,
                'n_components': self.n_components,
                'freq_band': self.freq_band,
                'training_metrics': self.training_metrics,
                'is_fitted': self.is_fitted
            }, f)
        print(f"Model saved to {filepath}")

    @classmethod
    def load(cls, filepath: str) -> 'CSPLDAClassifier':
        """Load a trained model."""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)

        model = cls(
            n_components=data['n_components'],
            freq_band=data['freq_band']
        )
        model.pipeline = data['pipeline']
        model.csp = data['csp']
        model.lda = data['lda']
        model.training_metrics = data['training_metrics']
        model.is_fitted = data['is_fitted']

        print(f"Model loaded from {filepath}")
        return model


def evaluate_classifier(
    clf: CSPLDAClassifier,
    X: np.ndarray,
    y: np.ndarray,
    dataset_name: str = "Test"
) -> Dict[str, float]:
    """
    Evaluate classifier and return detailed metrics.

    Parameters
    ----------
    clf : CSPLDAClassifier
        Trained classifier
    X : ndarray
        Test data
    y : ndarray
        True labels
    dataset_name : str
        Name for printing

    Returns
    -------
    metrics : dict
        Dictionary with accuracy, sensitivity, specificity, etc.
    """
    y_pred = clf.predict(X)

    # Compute confusion matrix
    tn, fp, fn, tp = confusion_matrix(y, y_pred).ravel()

    metrics = {
        'accuracy': (tp + tn) / (tp + tn + fp + fn),
        'sensitivity': tp / (tp + fn) if (tp + fn) > 0 else 0,  # True positive rate
        'specificity': tn / (tn + fp) if (tn + fp) > 0 else 0,  # True negative rate
        'precision': tp / (tp + fp) if (tp + fp) > 0 else 0,
        'f1': 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0,
        'false_positive_rate': fp / (fp + tn) if (fp + tn) > 0 else 0,
        'n_samples': len(y)
    }

    print(f"\n{dataset_name} Evaluation:")
    print(f"  Accuracy:    {metrics['accuracy']:.2%}")
    print(f"  Sensitivity: {metrics['sensitivity']:.2%} (detecting positive class)")
    print(f"  Specificity: {metrics['specificity']:.2%} (rejecting negative class)")
    print(f"  F1 Score:    {metrics['f1']:.2f}")
    print(f"  FPR:         {metrics['false_positive_rate']:.2%}")

    return metrics


def plot_csp_patterns(clf: CSPLDAClassifier, ch_names: list, output_path: str = None):
    """Plot CSP spatial patterns."""
    patterns = clf.get_csp_patterns()

    # Use actual number of patterns (limited by min of n_components and n_channels)
    n_patterns = patterns.shape[0]

    fig, axes = plt.subplots(1, n_patterns, figsize=(3 * n_patterns, 3))

    # Handle case when there's only one pattern
    if n_patterns == 1:
        axes = [axes]

    for i, ax in enumerate(axes):
        pattern = patterns[i]

        # Simple bar plot for patterns
        colors = ['red' if p < 0 else 'blue' for p in pattern]
        ax.bar(range(len(ch_names)), pattern, color=colors)
        ax.set_xticks(range(len(ch_names)))
        ax.set_xticklabels(ch_names)
        ax.set_title(f'CSP {i+1}')
        ax.set_ylabel('Weight')
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved CSP patterns to {output_path}")

    plt.close()


def main():
    """Train and evaluate CSP+LDA on PhysioNet data."""
    print("="*60)
    print("CSP + LDA Baseline Training on PhysioNet")
    print("="*60)

    # Load PhysioNet data (20 subjects)
    loader = PhysioNetLoader(subjects=list(range(1, 21)))
    X, y = loader.load_imagery_data()

    print(f"\nData loaded:")
    print(f"  Shape: {X.shape}")
    print(f"  Classes: rest={np.sum(y==0)}, imagery={np.sum(y==1)}")

    # Train-test split (80-20)
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"\nTrain set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")

    # Initialize classifier
    clf = CSPLDAClassifier(n_components=6)

    # Cross-validation on training data
    cv_results = clf.cross_validate(X_train, y_train, cv=5)

    # Fit on full training data
    clf.fit(X_train, y_train)

    # Evaluate on test set
    test_metrics = evaluate_classifier(clf, X_test, y_test, "PhysioNet Test Set")

    # Save model
    model_path = Path("models/physionet_csp_lda.pkl")
    clf.save(str(model_path))

    # Plot CSP patterns
    ch_names = loader.get_channel_info()['channels']
    plot_csp_patterns(clf, ch_names, "results/figures/csp_patterns_physionet.png")

    # Summary
    print("\n" + "="*60)
    print("TRAINING COMPLETE")
    print("="*60)
    print(f"\nPhysioNet MI-BCI Results:")
    print(f"  CV Accuracy: {cv_results['cv_mean']:.2%} (+/- {cv_results['cv_std']:.2%})")
    print(f"  Test Accuracy: {test_metrics['accuracy']:.2%}")
    print(f"\nModel saved to: {model_path}")

    return clf, cv_results, test_metrics


if __name__ == "__main__":
    clf, cv_results, test_metrics = main()
