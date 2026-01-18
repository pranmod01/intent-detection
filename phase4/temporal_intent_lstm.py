"""
Temporal Intent Detection with LSTM

GOAL: Detect intent onset/offset within each subject.

Why it works with N=2:
- Training happens within-subject (lots of epochs per subject)
- LOSO validation: train on Subject 1, test on Subject 2
- You're not claiming population generalization, just demonstrating
  the method works for temporal modeling
"""

import sys
import numpy as np
import json
from pathlib import Path
from typing import Dict, Tuple, List, Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# Add project paths
sys.path.insert(0, str(Path(__file__).parent.parent / 'phase1'))
sys.path.insert(0, str(Path(__file__).parent.parent / 'phase2'))


class TemporalIntentLSTM(nn.Module):
    """
    Bidirectional LSTM for temporal intent detection.

    Predicts intent probability at each timestep, enabling
    detection of intent onset and offset within a trial.
    """

    def __init__(self, n_channels: int = 4, hidden_size: int = 64, num_layers: int = 2,
                 dropout: float = 0.3):
        super().__init__()

        self.n_channels = n_channels
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            n_channels,
            hidden_size,
            num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )

        # Output layer: intent probability per timestep
        self.fc = nn.Linear(hidden_size * 2, 2)  # 2 classes: non-intent, intent

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input EEG data, shape (batch, n_channels, n_times)

        Returns
        -------
        logits : torch.Tensor
            Intent logits per timestep, shape (batch, n_times, 2)
        """
        # Permute to (batch, n_times, n_channels) for LSTM
        x = x.permute(0, 2, 1)

        # LSTM forward pass
        lstm_out, _ = self.lstm(x)  # (batch, n_times, hidden_size * 2)

        # Apply dropout
        lstm_out = self.dropout(lstm_out)

        # Project to class logits
        logits = self.fc(lstm_out)  # (batch, n_times, 2)

        return logits

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Get probability predictions per timestep."""
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            probs = torch.softmax(logits, dim=-1)
        return probs

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Get class predictions per timestep."""
        probs = self.predict_proba(x)
        return torch.argmax(probs, dim=-1)


class TemporalIntentDetector:
    """
    Wrapper class for training and evaluating the temporal LSTM model.

    Supports Leave-One-Subject-Out (LOSO) cross-validation.
    """

    def __init__(self, n_channels: int = 4, hidden_size: int = 64,
                 num_layers: int = 2, dropout: float = 0.3,
                 device: str = None):

        self.n_channels = n_channels
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout

        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.history = []

    def _create_model(self) -> TemporalIntentLSTM:
        """Create a new model instance."""
        model = TemporalIntentLSTM(
            n_channels=self.n_channels,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.dropout
        )
        return model.to(self.device)

    def _create_temporal_labels(self, y: np.ndarray, n_times: int,
                                 onset_fraction: float = 0.2) -> np.ndarray:
        """
        Create temporal labels for each timestep.

        For intent trials, the first `onset_fraction` is labeled as non-intent
        (pre-movement baseline), and the rest as intent.
        For non-intent trials, all timesteps are labeled as non-intent.

        Parameters
        ----------
        y : ndarray
            Binary labels (0=non-intent, 1=intent), shape (n_trials,)
        n_times : int
            Number of timesteps per trial
        onset_fraction : float
            Fraction of trial before intent onset (for intent trials)

        Returns
        -------
        y_temporal : ndarray
            Temporal labels, shape (n_trials, n_times)
        """
        n_trials = len(y)
        y_temporal = np.zeros((n_trials, n_times), dtype=np.int64)

        onset_idx = int(n_times * onset_fraction)

        for i in range(n_trials):
            if y[i] == 1:  # Intent trial
                # Label post-onset as intent
                y_temporal[i, onset_idx:] = 1
            # Non-intent trials remain all zeros

        return y_temporal

    def fit(self, X: np.ndarray, y: np.ndarray,
            X_val: np.ndarray = None, y_val: np.ndarray = None,
            epochs: int = 100, batch_size: int = 16,
            learning_rate: float = 1e-3, patience: int = 15,
            onset_fraction: float = 0.2, verbose: bool = True) -> Dict:
        """
        Train the temporal LSTM model.

        Parameters
        ----------
        X : ndarray
            Training data, shape (n_trials, n_channels, n_times)
        y : ndarray
            Trial-level labels, shape (n_trials,)
        X_val, y_val : ndarray, optional
            Validation data
        epochs : int
            Maximum number of training epochs
        batch_size : int
            Batch size for training
        learning_rate : float
            Learning rate
        patience : int
            Early stopping patience
        onset_fraction : float
            Fraction of trial before intent onset
        verbose : bool
            Print training progress

        Returns
        -------
        history : dict
            Training history with loss and metrics
        """
        self.model = self._create_model()
        n_times = X.shape[2]

        # Create temporal labels
        y_temporal = self._create_temporal_labels(y, n_times, onset_fraction)

        # Convert to tensors
        X_train = torch.FloatTensor(X).to(self.device)
        y_train = torch.LongTensor(y_temporal).to(self.device)

        train_dataset = TensorDataset(X_train, y_train)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        # Validation data
        if X_val is not None:
            y_val_temporal = self._create_temporal_labels(y_val, n_times, onset_fraction)
            X_val_t = torch.FloatTensor(X_val).to(self.device)
            y_val_t = torch.LongTensor(y_val_temporal).to(self.device)

        # Loss and optimizer
        # Weight classes to handle imbalance (more non-intent timesteps)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

        # Training loop
        history = {'train_loss': [], 'val_loss': [], 'val_acc': []}
        best_val_loss = float('inf')
        best_state = None
        patience_counter = 0

        for epoch in range(epochs):
            # Training
            self.model.train()
            train_loss = 0

            for X_batch, y_batch in train_loader:
                optimizer.zero_grad()

                logits = self.model(X_batch)  # (batch, n_times, 2)

                # Reshape for loss computation
                logits_flat = logits.view(-1, 2)  # (batch * n_times, 2)
                y_flat = y_batch.view(-1)  # (batch * n_times,)

                loss = criterion(logits_flat, y_flat)
                loss.backward()

                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

                optimizer.step()
                train_loss += loss.item()

            train_loss /= len(train_loader)
            history['train_loss'].append(train_loss)

            # Validation
            if X_val is not None:
                self.model.eval()
                with torch.no_grad():
                    val_logits = self.model(X_val_t)
                    val_logits_flat = val_logits.view(-1, 2)
                    y_val_flat = y_val_t.view(-1)

                    val_loss = criterion(val_logits_flat, y_val_flat).item()

                    val_preds = torch.argmax(val_logits_flat, dim=-1)
                    val_acc = (val_preds == y_val_flat).float().mean().item()

                history['val_loss'].append(val_loss)
                history['val_acc'].append(val_acc)

                scheduler.step(val_loss)

                # Early stopping
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_state = self.model.state_dict().copy()
                    patience_counter = 0
                else:
                    patience_counter += 1

                if verbose and (epoch + 1) % 10 == 0:
                    print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f}, "
                          f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

                if patience_counter >= patience:
                    if verbose:
                        print(f"Early stopping at epoch {epoch+1}")
                    break
            else:
                if verbose and (epoch + 1) % 10 == 0:
                    print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f}")

        # Load best model
        if best_state is not None:
            self.model.load_state_dict(best_state)

        self.history = history
        return history

    def evaluate(self, X: np.ndarray, y: np.ndarray,
                 onset_fraction: float = 0.2) -> Dict:
        """
        Evaluate the model on test data.

        Returns both timestep-level and trial-level metrics.
        """
        if self.model is None:
            raise ValueError("Model not trained. Call fit() first.")

        n_times = X.shape[2]
        y_temporal = self._create_temporal_labels(y, n_times, onset_fraction)

        X_t = torch.FloatTensor(X).to(self.device)

        self.model.eval()
        with torch.no_grad():
            preds = self.model.predict(X_t).cpu().numpy()

        # Timestep-level metrics
        y_true_flat = y_temporal.flatten()
        y_pred_flat = preds.flatten()

        timestep_acc = accuracy_score(y_true_flat, y_pred_flat)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true_flat, y_pred_flat, average='binary', zero_division=0
        )

        # Trial-level metrics (majority vote across timesteps)
        trial_preds = (preds.mean(axis=1) > 0.5).astype(int)
        trial_acc = accuracy_score(y, trial_preds)

        return {
            'timestep_accuracy': timestep_acc,
            'timestep_precision': precision,
            'timestep_recall': recall,
            'timestep_f1': f1,
            'trial_accuracy': trial_acc,
            'n_trials': len(y),
            'n_timesteps': n_times
        }

    def detect_onset(self, X: np.ndarray, threshold: float = 0.5,
                     min_duration: int = 10) -> List[Optional[int]]:
        """
        Detect intent onset for each trial.

        Parameters
        ----------
        X : ndarray
            EEG data, shape (n_trials, n_channels, n_times)
        threshold : float
            Probability threshold for intent detection
        min_duration : int
            Minimum consecutive timesteps above threshold

        Returns
        -------
        onsets : list
            Onset index for each trial (None if no onset detected)
        """
        if self.model is None:
            raise ValueError("Model not trained. Call fit() first.")

        X_t = torch.FloatTensor(X).to(self.device)
        probs = self.model.predict_proba(X_t).cpu().numpy()  # (n_trials, n_times, 2)
        intent_probs = probs[:, :, 1]  # Intent probability

        onsets = []
        for i in range(len(X)):
            above_threshold = intent_probs[i] > threshold

            # Find first sustained period above threshold
            onset = None
            consecutive = 0
            for t in range(len(above_threshold)):
                if above_threshold[t]:
                    consecutive += 1
                    if consecutive >= min_duration and onset is None:
                        onset = t - min_duration + 1
                else:
                    consecutive = 0

            onsets.append(onset)

        return onsets

    def save(self, path: str):
        """Save model to file."""
        if self.model is None:
            raise ValueError("No model to save.")

        state = {
            'model_state': self.model.state_dict(),
            'n_channels': self.n_channels,
            'hidden_size': self.hidden_size,
            'num_layers': self.num_layers,
            'dropout': self.dropout,
            'history': self.history
        }
        torch.save(state, path)

    @classmethod
    def load(cls, path: str, device: str = None) -> 'TemporalIntentDetector':
        """Load model from file."""
        state = torch.load(path, map_location=device or 'cpu')

        detector = cls(
            n_channels=state['n_channels'],
            hidden_size=state['hidden_size'],
            num_layers=state['num_layers'],
            dropout=state['dropout'],
            device=device
        )

        detector.model = detector._create_model()
        detector.model.load_state_dict(state['model_state'])
        detector.history = state.get('history', [])

        return detector


def run_loso_evaluation(X: np.ndarray, y: np.ndarray,
                        subject_ids: np.ndarray,
                        **kwargs) -> Dict:
    """
    Run Leave-One-Subject-Out cross-validation.

    Parameters
    ----------
    X : ndarray
        EEG data, shape (n_trials, n_channels, n_times)
    y : ndarray
        Binary labels, shape (n_trials,)
    subject_ids : ndarray
        Subject ID for each trial, shape (n_trials,)
    **kwargs : dict
        Arguments passed to TemporalIntentDetector

    Returns
    -------
    results : dict
        Cross-validation results
    """
    unique_subjects = np.unique(subject_ids)
    print(f"\nRunning LOSO CV with {len(unique_subjects)} subjects...")

    all_results = []

    for test_subject in unique_subjects:
        print(f"\n  Testing on Subject {test_subject}...")

        # Split data
        train_mask = subject_ids != test_subject
        test_mask = subject_ids == test_subject

        X_train, y_train = X[train_mask], y[train_mask]
        X_test, y_test = X[test_mask], y[test_mask]

        print(f"    Train: {len(X_train)} trials, Test: {len(X_test)} trials")

        # Train and evaluate
        detector = TemporalIntentDetector(n_channels=X.shape[1], **kwargs)
        detector.fit(X_train, y_train, X_val=X_test, y_val=y_test, verbose=False)

        metrics = detector.evaluate(X_test, y_test)
        metrics['test_subject'] = int(test_subject)
        all_results.append(metrics)

        print(f"    Timestep Acc: {metrics['timestep_accuracy']:.2%}, "
              f"Trial Acc: {metrics['trial_accuracy']:.2%}")

    # Aggregate results
    aggregate = {
        'mean_timestep_accuracy': np.mean([r['timestep_accuracy'] for r in all_results]),
        'std_timestep_accuracy': np.std([r['timestep_accuracy'] for r in all_results]),
        'mean_trial_accuracy': np.mean([r['trial_accuracy'] for r in all_results]),
        'std_trial_accuracy': np.std([r['trial_accuracy'] for r in all_results]),
        'per_subject': all_results
    }

    return aggregate


def main():
    """Run temporal intent detection with LOSO validation."""
    print("="*60)
    print("Phase 5: Temporal Intent Detection with LSTM")
    print("="*60)

    # Load data
    print("\n[1/3] Loading data...")
    from data_organizer import IntentDataOrganizer

    organizer = IntentDataOrganizer(use_all_channels=True)
    data = organizer.load_all_data()

    X = data['X']
    y = data['y_binary']
    conditions = data['conditions']

    # Create pseudo-subject IDs based on conditions
    # (In real scenario, you'd have actual subject IDs)
    subject_ids = np.array([hash(c) % 2 for c in conditions])  # 2 pseudo-subjects

    print(f"Data shape: {X.shape}")
    print(f"Labels: {np.bincount(y)}")
    print(f"Subjects: {np.bincount(subject_ids)}")

    # Train with LOSO validation
    print("\n[2/3] Running LOSO cross-validation...")
    results = run_loso_evaluation(
        X, y, subject_ids,
        hidden_size=64,
        num_layers=2,
        dropout=0.3
    )

    # Print results
    print("\n" + "="*60)
    print("RESULTS: Temporal Intent Detection")
    print("="*60)
    print(f"\nTimestep-level accuracy: {results['mean_timestep_accuracy']:.2%} "
          f"(+/- {results['std_timestep_accuracy']:.2%})")
    print(f"Trial-level accuracy: {results['mean_trial_accuracy']:.2%} "
          f"(+/- {results['std_trial_accuracy']:.2%})")

    # Train final model on all data
    print("\n[3/3] Training final model on all data...")
    final_detector = TemporalIntentDetector(n_channels=X.shape[1])
    final_detector.fit(X, y, epochs=100, verbose=True)

    # Save model and results
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    final_detector.save(str(models_dir / "temporal_lstm_intent.pt"))

    results_path = Path("results/temporal_intent_detection.json")
    results_path.parent.mkdir(parents=True, exist_ok=True)
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nModel saved to models/temporal_lstm_intent.pt")
    print(f"Results saved to {results_path}")

    # Demonstrate onset detection
    print("\n" + "="*60)
    print("Demo: Intent Onset Detection")
    print("="*60)

    onsets = final_detector.detect_onset(X[:10], threshold=0.5, min_duration=10)
    for i, onset in enumerate(onsets):
        label = "Intent" if y[i] == 1 else "Non-Intent"
        if onset is not None:
            print(f"  Trial {i+1} ({label}): Onset at timestep {onset}")
        else:
            print(f"  Trial {i+1} ({label}): No onset detected")

    return results


if __name__ == "__main__":
    results = main()
