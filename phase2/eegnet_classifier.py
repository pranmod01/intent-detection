"""
EEGNet Classifier for Intent Detection

Implements the EEGNet architecture (Lawhern et al., 2018) for
classifying intent vs. non-intent from EEG signals.

Reference: Lawhern et al., "EEGNet: A Compact Convolutional Network
for EEG-based Brain-Computer Interfaces", J Neural Eng, 2018.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from typing import Tuple, Dict, List, Optional
from pathlib import Path
import json
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from tqdm import tqdm


class EEGNet(nn.Module):
    """
    EEGNet architecture for EEG classification.

    Architecture:
    1. Temporal convolution (learn frequency filters)
    2. Depthwise spatial convolution (learn spatial filters per temporal filter)
    3. Separable convolution (combine features)
    4. Classification head
    """

    def __init__(
        self,
        n_channels: int = 4,
        n_samples: int = 400,  # 2 seconds at 200 Hz
        n_classes: int = 2,
        F1: int = 8,  # Number of temporal filters
        D: int = 2,   # Depth multiplier (spatial filters per temporal filter)
        F2: int = 16, # Number of pointwise filters
        kernel_length: int = 64,  # Temporal filter length (0.5s at 128Hz, scale for your sfreq)
        dropout_rate: float = 0.5
    ):
        super(EEGNet, self).__init__()

        self.n_channels = n_channels
        self.n_samples = n_samples
        self.n_classes = n_classes

        # Block 1: Temporal Convolution
        self.conv1 = nn.Conv2d(1, F1, (1, kernel_length), padding='same', bias=False)
        self.bn1 = nn.BatchNorm2d(F1)

        # Block 1: Depthwise Spatial Convolution
        self.conv2 = nn.Conv2d(F1, F1 * D, (n_channels, 1), groups=F1, bias=False)
        self.bn2 = nn.BatchNorm2d(F1 * D)
        self.pool1 = nn.AvgPool2d((1, 4))
        self.dropout1 = nn.Dropout(dropout_rate)

        # Block 2: Separable Convolution
        self.conv3 = nn.Conv2d(F1 * D, F1 * D, (1, 16), padding='same', groups=F1 * D, bias=False)
        self.conv4 = nn.Conv2d(F1 * D, F2, (1, 1), bias=False)
        self.bn3 = nn.BatchNorm2d(F2)
        self.pool2 = nn.AvgPool2d((1, 8))
        self.dropout2 = nn.Dropout(dropout_rate)

        # Calculate flatten size
        self._to_linear = None
        self._get_conv_output((1, n_channels, n_samples))

        # Classification head
        self.fc = nn.Linear(self._to_linear, n_classes)

    def _get_conv_output(self, shape):
        """Calculate output size after convolutions."""
        x = torch.zeros(1, *shape)
        x = self._forward_features(x)
        self._to_linear = x.view(1, -1).size(1)

    def _forward_features(self, x):
        """Forward through convolutional layers."""
        # Block 1
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.elu(x)
        x = self.pool1(x)
        x = self.dropout1(x)

        # Block 2
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.bn3(x)
        x = F.elu(x)
        x = self.pool2(x)
        x = self.dropout2(x)

        return x

    def forward(self, x):
        """
        Forward pass.

        Parameters
        ----------
        x : tensor, shape (batch, channels, samples)
            EEG data

        Returns
        -------
        out : tensor, shape (batch, n_classes)
            Class logits
        """
        # Add channel dimension for Conv2d: (batch, 1, channels, samples)
        if x.dim() == 3:
            x = x.unsqueeze(1)

        x = self._forward_features(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x

    def get_embeddings(self, x):
        """Extract feature embeddings before classification."""
        if x.dim() == 3:
            x = x.unsqueeze(1)

        x = self._forward_features(x)
        return x.view(x.size(0), -1)


class EEGNetClassifier:
    """Wrapper class for training and evaluation."""

    def __init__(
        self,
        n_channels: int = 4,
        n_samples: int = 400,
        n_classes: int = 2,
        lr: float = 0.001,
        device: str = None
    ):
        self.n_channels = n_channels
        self.n_samples = n_samples
        self.n_classes = n_classes
        self.lr = lr

        # Set device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        # Initialize model
        self.model = EEGNet(
            n_channels=n_channels,
            n_samples=n_samples,
            n_classes=n_classes
        ).to(self.device)

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.criterion = nn.CrossEntropyLoss()

        self.history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
        self.is_fitted = False

    def _prepare_data(self, X: np.ndarray, y: np.ndarray, batch_size: int = 32, shuffle: bool = True):
        """Convert numpy arrays to DataLoader."""
        X_tensor = torch.FloatTensor(X)
        y_tensor = torch.LongTensor(y)
        dataset = TensorDataset(X_tensor, y_tensor)
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray = None,
        y_val: np.ndarray = None,
        epochs: int = 100,
        batch_size: int = 32,
        patience: int = 10,
        verbose: bool = True
    ) -> Dict:
        """
        Train the EEGNet model.

        Parameters
        ----------
        X_train : ndarray, shape (n_samples, n_channels, n_times)
        y_train : ndarray, shape (n_samples,)
        X_val : ndarray, optional
        y_val : ndarray, optional
        epochs : int
        batch_size : int
        patience : int
            Early stopping patience
        verbose : bool

        Returns
        -------
        history : dict
            Training history
        """
        train_loader = self._prepare_data(X_train, y_train, batch_size, shuffle=True)

        if X_val is not None:
            val_loader = self._prepare_data(X_val, y_val, batch_size, shuffle=False)
        else:
            val_loader = None

        best_val_loss = float('inf')
        best_model_state = None
        patience_counter = 0

        iterator = range(epochs)
        if verbose:
            iterator = tqdm(iterator, desc="Training")

        for epoch in iterator:
            # Training
            self.model.train()
            train_loss = 0
            train_correct = 0
            train_total = 0

            for X_batch, y_batch in train_loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)

                self.optimizer.zero_grad()
                outputs = self.model(X_batch)
                loss = self.criterion(outputs, y_batch)
                loss.backward()
                self.optimizer.step()

                train_loss += loss.item()
                _, predicted = outputs.max(1)
                train_total += y_batch.size(0)
                train_correct += predicted.eq(y_batch).sum().item()

            train_loss /= len(train_loader)
            train_acc = train_correct / train_total

            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)

            # Validation
            if val_loader is not None:
                val_loss, val_acc = self._evaluate(val_loader)
                self.history['val_loss'].append(val_loss)
                self.history['val_acc'].append(val_acc)

                # Early stopping
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_model_state = self.model.state_dict().copy()
                    patience_counter = 0
                else:
                    patience_counter += 1

                if patience_counter >= patience:
                    if verbose:
                        print(f"\nEarly stopping at epoch {epoch+1}")
                    break

                if verbose:
                    iterator.set_postfix({
                        'train_loss': f'{train_loss:.4f}',
                        'train_acc': f'{train_acc:.2%}',
                        'val_loss': f'{val_loss:.4f}',
                        'val_acc': f'{val_acc:.2%}'
                    })
            else:
                if verbose:
                    iterator.set_postfix({
                        'train_loss': f'{train_loss:.4f}',
                        'train_acc': f'{train_acc:.2%}'
                    })

        # Restore best model
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)

        self.is_fitted = True
        return self.history

    def _evaluate(self, data_loader) -> Tuple[float, float]:
        """Evaluate on a data loader."""
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0

        with torch.no_grad():
            for X_batch, y_batch in data_loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)

                outputs = self.model(X_batch)
                loss = self.criterion(outputs, y_batch)

                total_loss += loss.item()
                _, predicted = outputs.max(1)
                total += y_batch.size(0)
                correct += predicted.eq(y_batch).sum().item()

        return total_loss / len(data_loader), correct / total

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels."""
        self.model.eval()
        X_tensor = torch.FloatTensor(X).to(self.device)

        with torch.no_grad():
            outputs = self.model(X_tensor)
            _, predicted = outputs.max(1)

        return predicted.cpu().numpy()

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities."""
        self.model.eval()
        X_tensor = torch.FloatTensor(X).to(self.device)

        with torch.no_grad():
            outputs = self.model(X_tensor)
            probs = F.softmax(outputs, dim=1)

        return probs.cpu().numpy()

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Return accuracy score."""
        y_pred = self.predict(X)
        return accuracy_score(y, y_pred)

    def get_detailed_metrics(self, X: np.ndarray, y: np.ndarray) -> Dict:
        """Get detailed classification metrics."""
        y_pred = self.predict(X)

        # Confusion matrix
        tn, fp, fn, tp = confusion_matrix(y, y_pred).ravel()

        metrics = {
            'accuracy': (tp + tn) / (tp + tn + fp + fn),
            'sensitivity': tp / (tp + fn) if (tp + fn) > 0 else 0,
            'specificity': tn / (tn + fp) if (tn + fp) > 0 else 0,
            'precision': tp / (tp + fp) if (tp + fp) > 0 else 0,
            'f1': 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0,
            'confusion_matrix': [[tn, fp], [fn, tp]]
        }

        return metrics

    def get_embeddings(self, X: np.ndarray) -> np.ndarray:
        """Extract feature embeddings."""
        self.model.eval()
        X_tensor = torch.FloatTensor(X).to(self.device)

        with torch.no_grad():
            embeddings = self.model.get_embeddings(X_tensor)

        return embeddings.cpu().numpy()

    def save(self, filepath: str):
        """Save model weights and config."""
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': {
                'n_channels': self.n_channels,
                'n_samples': self.n_samples,
                'n_classes': self.n_classes,
                'lr': self.lr
            },
            'history': self.history,
            'is_fitted': self.is_fitted
        }, filepath)
        print(f"Model saved to {filepath}")

    @classmethod
    def load(cls, filepath: str) -> 'EEGNetClassifier':
        """Load model from file."""
        checkpoint = torch.load(filepath, map_location='cpu')
        config = checkpoint['config']

        model = cls(
            n_channels=config['n_channels'],
            n_samples=config['n_samples'],
            n_classes=config['n_classes'],
            lr=config['lr']
        )

        model.model.load_state_dict(checkpoint['model_state_dict'])
        model.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        model.history = checkpoint['history']
        model.is_fitted = checkpoint['is_fitted']

        print(f"Model loaded from {filepath}")
        return model


def augment_data(
    X: np.ndarray,
    y: np.ndarray,
    noise_factor: float = 0.1,
    time_shift_max: int = 20
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply data augmentation for small datasets.

    Parameters
    ----------
    X : ndarray, shape (n_samples, n_channels, n_times)
    y : ndarray
    noise_factor : float
        Gaussian noise standard deviation as fraction of signal std
    time_shift_max : int
        Maximum time shift in samples

    Returns
    -------
    X_aug, y_aug : augmented data (2x original size)
    """
    n_samples = len(X)
    X_aug = []
    y_aug = []

    for i in range(n_samples):
        # Original
        X_aug.append(X[i])
        y_aug.append(y[i])

        # Noisy version
        noise = np.random.randn(*X[i].shape) * noise_factor * np.std(X[i])
        X_aug.append(X[i] + noise)
        y_aug.append(y[i])

        # Time-shifted version (optional, can add more augmentation)
        # shift = np.random.randint(-time_shift_max, time_shift_max)
        # X_shifted = np.roll(X[i], shift, axis=-1)
        # X_aug.append(X_shifted)
        # y_aug.append(y[i])

    return np.array(X_aug), np.array(y_aug)


def main():
    """Test EEGNet training."""
    print("="*60)
    print("EEGNet Classifier Test")
    print("="*60)

    # Load data (uses project-root-relative paths by default)
    from data_organizer import IntentDataOrganizer

    organizer = IntentDataOrganizer()
    data = organizer.load_all_data()

    # Get train/test split
    X_train, X_test, y_train, y_test = organizer.get_train_test_split(test_subject=1)

    print(f"\nTrain shape: {X_train.shape}")
    print(f"Test shape: {X_test.shape}")

    # Balance training data
    X_train_bal, y_train_bal = organizer.balance_classes(X_train, y_train)
    print(f"Balanced train shape: {X_train_bal.shape}")

    # Augment data
    X_train_aug, y_train_aug = augment_data(X_train_bal, y_train_bal)
    print(f"Augmented train shape: {X_train_aug.shape}")

    # Initialize classifier
    clf = EEGNetClassifier(
        n_channels=X_train.shape[1],
        n_samples=X_train.shape[2],
        n_classes=2
    )

    print(f"\nDevice: {clf.device}")

    # Train
    print("\nTraining EEGNet...")
    history = clf.fit(
        X_train_aug, y_train_aug,
        X_val=X_test, y_val=y_test,
        epochs=100,
        batch_size=16,
        patience=15
    )

    # Evaluate
    print("\n" + "="*50)
    print("EVALUATION")
    print("="*50)

    metrics = clf.get_detailed_metrics(X_test, y_test)

    print(f"\nTest Set Results:")
    print(f"  Accuracy:    {metrics['accuracy']:.2%}")
    print(f"  Sensitivity: {metrics['sensitivity']:.2%}")
    print(f"  Specificity: {metrics['specificity']:.2%}")
    print(f"  F1 Score:    {metrics['f1']:.2f}")

    # Save model
    clf.save("models/eegnet_intent.pt")

    return clf, metrics


if __name__ == "__main__":
    clf, metrics = main()
