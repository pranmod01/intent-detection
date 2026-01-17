"""
Autoencoder-based Transfer Learning

Pre-train a convolutional autoencoder on PhysioNet EEG data,
then transfer the encoder to classify intent vs. non-intent on local data.

This approach leverages the larger PhysioNet dataset to learn general EEG
representations, then fine-tunes on the smaller multi-condition local dataset.
"""

import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path
from typing import Dict, Tuple, Optional
import json
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from tqdm import tqdm

# Add project paths
sys.path.insert(0, str(Path(__file__).parent.parent / 'phase1'))
sys.path.insert(0, str(Path(__file__).parent.parent / 'phase2'))

from physionet_loader import PhysioNetLoader
from data_organizer import IntentDataOrganizer


def load_physionet_data(subjects=None, runs=None):
    """Helper function to load PhysioNet data using PhysioNetLoader."""
    loader = PhysioNetLoader(subjects=subjects or list(range(1, 11)))
    X, y = loader.load_imagery_data()
    ch_names = loader.TARGET_CHANNELS
    return X, y, ch_names


class EEGEncoder(nn.Module):
    """
    Convolutional encoder for EEG signals.

    Architecture designed to capture temporal and spatial patterns.
    """

    def __init__(
        self,
        n_channels: int = 3,
        n_samples: int = 400,
        latent_dim: int = 128
    ):
        super(EEGEncoder, self).__init__()

        self.n_channels = n_channels
        self.n_samples = n_samples
        self.latent_dim = latent_dim

        # Temporal convolutions
        self.conv1 = nn.Conv1d(n_channels, 32, kernel_size=25, padding=12)
        self.bn1 = nn.BatchNorm1d(32)
        self.pool1 = nn.MaxPool1d(4)

        self.conv2 = nn.Conv1d(32, 64, kernel_size=15, padding=7)
        self.bn2 = nn.BatchNorm1d(64)
        self.pool2 = nn.MaxPool1d(4)

        self.conv3 = nn.Conv1d(64, 128, kernel_size=9, padding=4)
        self.bn3 = nn.BatchNorm1d(128)
        self.pool3 = nn.MaxPool1d(5)

        # Calculate flattened size
        self._flat_size = self._get_flat_size()

        # Fully connected to latent space
        self.fc = nn.Linear(self._flat_size, latent_dim)

    def _get_flat_size(self):
        """Calculate size after convolutions."""
        x = torch.zeros(1, self.n_channels, self.n_samples)
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))
        return x.view(1, -1).size(1)

    def forward(self, x):
        """
        Encode EEG to latent representation.

        Parameters
        ----------
        x : tensor, shape (batch, channels, samples)

        Returns
        -------
        z : tensor, shape (batch, latent_dim)
        """
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))
        x = x.view(x.size(0), -1)
        z = self.fc(x)
        return z


class EEGDecoder(nn.Module):
    """
    Convolutional decoder to reconstruct EEG from latent space.
    """

    def __init__(
        self,
        n_channels: int = 3,
        n_samples: int = 400,
        latent_dim: int = 128
    ):
        super(EEGDecoder, self).__init__()

        self.n_channels = n_channels
        self.n_samples = n_samples
        self.latent_dim = latent_dim

        # Calculate intermediate size (after encoder pooling: 400 -> 100 -> 25 -> 5)
        self.intermediate_size = n_samples // 80  # 5 for 400 samples

        # Fully connected from latent space
        self.fc = nn.Linear(latent_dim, 128 * self.intermediate_size)

        # Transposed convolutions for upsampling
        self.deconv1 = nn.ConvTranspose1d(128, 64, kernel_size=9, stride=5, padding=2, output_padding=0)
        self.bn1 = nn.BatchNorm1d(64)

        self.deconv2 = nn.ConvTranspose1d(64, 32, kernel_size=15, stride=4, padding=4, output_padding=1)
        self.bn2 = nn.BatchNorm1d(32)

        self.deconv3 = nn.ConvTranspose1d(32, n_channels, kernel_size=25, stride=4, padding=9, output_padding=1)

    def forward(self, z):
        """
        Decode latent representation to EEG.

        Parameters
        ----------
        z : tensor, shape (batch, latent_dim)

        Returns
        -------
        x_recon : tensor, shape (batch, channels, samples)
        """
        x = self.fc(z)
        x = x.view(x.size(0), 128, self.intermediate_size)

        x = F.relu(self.bn1(self.deconv1(x)))
        x = F.relu(self.bn2(self.deconv2(x)))
        x_recon = self.deconv3(x)

        # Ensure correct output size
        if x_recon.size(2) != self.n_samples:
            x_recon = F.interpolate(x_recon, size=self.n_samples, mode='linear', align_corners=False)

        return x_recon


class EEGAutoencoder(nn.Module):
    """
    Full autoencoder combining encoder and decoder.
    """

    def __init__(
        self,
        n_channels: int = 3,
        n_samples: int = 400,
        latent_dim: int = 128
    ):
        super(EEGAutoencoder, self).__init__()

        self.encoder = EEGEncoder(n_channels, n_samples, latent_dim)
        self.decoder = EEGDecoder(n_channels, n_samples, latent_dim)

    def forward(self, x):
        z = self.encoder(x)
        x_recon = self.decoder(z)
        return x_recon, z

    def encode(self, x):
        return self.encoder(x)


class IntentClassifier(nn.Module):
    """
    Classification head for intent detection.
    Takes encoder output and classifies as intent/non-intent.
    """

    def __init__(self, latent_dim: int = 128, n_classes: int = 2):
        super(IntentClassifier, self).__init__()

        self.fc1 = nn.Linear(latent_dim, 64)
        self.bn1 = nn.BatchNorm1d(64)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(64, n_classes)

    def forward(self, z):
        x = F.relu(self.bn1(self.fc1(z)))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class TransferLearningPipeline:
    """
    Full transfer learning pipeline:
    1. Pretrain autoencoder on PhysioNet
    2. Freeze encoder, add classifier
    3. Fine-tune on local multi-condition data
    """

    def __init__(
        self,
        n_channels: int = 3,
        n_samples: int = 400,
        latent_dim: int = 128,
        device: str = None
    ):
        self.n_channels = n_channels
        self.n_samples = n_samples
        self.latent_dim = latent_dim

        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        # Initialize autoencoder
        self.autoencoder = EEGAutoencoder(n_channels, n_samples, latent_dim).to(self.device)
        self.classifier = None

        self.pretrain_history = {'loss': []}
        self.finetune_history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

    def pretrain(
        self,
        X: np.ndarray,
        epochs: int = 50,
        batch_size: int = 32,
        lr: float = 0.001,
        verbose: bool = True
    ) -> Dict:
        """
        Pretrain autoencoder on reconstruction task.

        Parameters
        ----------
        X : ndarray, shape (n_samples, n_channels, n_times)
            EEG data (e.g., from PhysioNet)
        """
        print(f"\nPretraining autoencoder on {len(X)} samples...")
        print(f"Device: {self.device}")

        # Prepare data
        X_tensor = torch.FloatTensor(X)
        dataset = TensorDataset(X_tensor, X_tensor)  # Input = Target for autoencoder
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        optimizer = torch.optim.Adam(self.autoencoder.parameters(), lr=lr)
        criterion = nn.MSELoss()

        iterator = range(epochs)
        if verbose:
            iterator = tqdm(iterator, desc="Pretraining")

        for epoch in iterator:
            self.autoencoder.train()
            epoch_loss = 0

            for X_batch, _ in dataloader:
                X_batch = X_batch.to(self.device)

                optimizer.zero_grad()
                X_recon, _ = self.autoencoder(X_batch)
                loss = criterion(X_recon, X_batch)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()

            epoch_loss /= len(dataloader)
            self.pretrain_history['loss'].append(epoch_loss)

            if verbose:
                iterator.set_postfix({'loss': f'{epoch_loss:.6f}'})

        print(f"Pretraining complete. Final loss: {self.pretrain_history['loss'][-1]:.6f}")
        return self.pretrain_history

    def finetune(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray = None,
        y_val: np.ndarray = None,
        epochs: int = 100,
        batch_size: int = 16,
        lr: float = 0.0005,
        freeze_encoder: bool = True,
        patience: int = 15,
        verbose: bool = True
    ) -> Dict:
        """
        Fine-tune classifier on local multi-condition data.

        Parameters
        ----------
        freeze_encoder : bool
            If True, freeze encoder weights during fine-tuning
        """
        print(f"\nFine-tuning on {len(X_train)} training samples...")

        # Initialize classifier
        self.classifier = IntentClassifier(self.latent_dim, n_classes=2).to(self.device)

        # Optionally freeze encoder
        if freeze_encoder:
            print("Freezing encoder weights...")
            for param in self.autoencoder.encoder.parameters():
                param.requires_grad = False

            # Only train classifier
            optimizer = torch.optim.Adam(self.classifier.parameters(), lr=lr)
        else:
            print("Fine-tuning entire model...")
            params = list(self.autoencoder.encoder.parameters()) + list(self.classifier.parameters())
            optimizer = torch.optim.Adam(params, lr=lr)

        criterion = nn.CrossEntropyLoss()

        # Prepare data
        train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.LongTensor(y_train))
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        if X_val is not None:
            val_dataset = TensorDataset(torch.FloatTensor(X_val), torch.LongTensor(y_val))
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        else:
            val_loader = None

        best_val_loss = float('inf')
        best_state = None
        patience_counter = 0

        iterator = range(epochs)
        if verbose:
            iterator = tqdm(iterator, desc="Fine-tuning")

        for epoch in iterator:
            # Training
            self.autoencoder.encoder.eval() if freeze_encoder else self.autoencoder.encoder.train()
            self.classifier.train()

            train_loss = 0
            train_correct = 0
            train_total = 0

            for X_batch, y_batch in train_loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)

                optimizer.zero_grad()

                # Get embeddings from encoder
                with torch.no_grad() if freeze_encoder else torch.enable_grad():
                    z = self.autoencoder.encode(X_batch)

                # Classify
                outputs = self.classifier(z)
                loss = criterion(outputs, y_batch)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                _, predicted = outputs.max(1)
                train_total += y_batch.size(0)
                train_correct += predicted.eq(y_batch).sum().item()

            train_loss /= len(train_loader)
            train_acc = train_correct / train_total

            self.finetune_history['train_loss'].append(train_loss)
            self.finetune_history['train_acc'].append(train_acc)

            # Validation
            if val_loader is not None:
                val_loss, val_acc = self._evaluate(val_loader, freeze_encoder)
                self.finetune_history['val_loss'].append(val_loss)
                self.finetune_history['val_acc'].append(val_acc)

                # Early stopping
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_state = {
                        'encoder': self.autoencoder.encoder.state_dict(),
                        'classifier': self.classifier.state_dict()
                    }
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
        if best_state is not None:
            self.autoencoder.encoder.load_state_dict(best_state['encoder'])
            self.classifier.load_state_dict(best_state['classifier'])

        return self.finetune_history

    def _evaluate(self, dataloader, freeze_encoder=True):
        """Evaluate on a dataloader."""
        self.autoencoder.encoder.eval()
        self.classifier.eval()

        total_loss = 0
        correct = 0
        total = 0
        criterion = nn.CrossEntropyLoss()

        with torch.no_grad():
            for X_batch, y_batch in dataloader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)

                z = self.autoencoder.encode(X_batch)
                outputs = self.classifier(z)
                loss = criterion(outputs, y_batch)

                total_loss += loss.item()
                _, predicted = outputs.max(1)
                total += y_batch.size(0)
                correct += predicted.eq(y_batch).sum().item()

        return total_loss / len(dataloader), correct / total

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels."""
        self.autoencoder.encoder.eval()
        self.classifier.eval()

        X_tensor = torch.FloatTensor(X).to(self.device)

        with torch.no_grad():
            z = self.autoencoder.encode(X_tensor)
            outputs = self.classifier(z)
            _, predicted = outputs.max(1)

        return predicted.cpu().numpy()

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities."""
        self.autoencoder.encoder.eval()
        self.classifier.eval()

        X_tensor = torch.FloatTensor(X).to(self.device)

        with torch.no_grad():
            z = self.autoencoder.encode(X_tensor)
            outputs = self.classifier(z)
            probs = F.softmax(outputs, dim=1)

        return probs.cpu().numpy()

    def get_detailed_metrics(self, X: np.ndarray, y: np.ndarray) -> Dict:
        """Get detailed classification metrics."""
        y_pred = self.predict(X)

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

    def save(self, filepath: str):
        """Save model."""
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            'autoencoder': self.autoencoder.state_dict(),
            'classifier': self.classifier.state_dict() if self.classifier else None,
            'config': {
                'n_channels': self.n_channels,
                'n_samples': self.n_samples,
                'latent_dim': self.latent_dim
            },
            'pretrain_history': self.pretrain_history,
            'finetune_history': self.finetune_history
        }, filepath)
        print(f"Model saved to {filepath}")

    @classmethod
    def load(cls, filepath: str) -> 'TransferLearningPipeline':
        """Load model."""
        checkpoint = torch.load(filepath, map_location='cpu')
        config = checkpoint['config']

        pipeline = cls(
            n_channels=config['n_channels'],
            n_samples=config['n_samples'],
            latent_dim=config['latent_dim']
        )

        pipeline.autoencoder.load_state_dict(checkpoint['autoencoder'])

        if checkpoint['classifier'] is not None:
            pipeline.classifier = IntentClassifier(config['latent_dim'], n_classes=2)
            pipeline.classifier.load_state_dict(checkpoint['classifier'])
            pipeline.classifier.to(pipeline.device)

        pipeline.pretrain_history = checkpoint['pretrain_history']
        pipeline.finetune_history = checkpoint['finetune_history']

        print(f"Model loaded from {filepath}")
        return pipeline


def main():
    """Run transfer learning pipeline."""
    print("="*60)
    print("Phase 3: Transfer Learning with Autoencoder")
    print("="*60)

    # Step 1: Load PhysioNet data for pretraining
    print("\n[1/4] Loading PhysioNet data for pretraining...")
    X_physionet, y_physionet, ch_names = load_physionet_data(
        subjects=list(range(1, 11)),  # Use 10 subjects
        runs=[4, 8, 12]  # Motor imagery runs
    )
    print(f"PhysioNet data shape: {X_physionet.shape}")

    # Step 2: Load local data for fine-tuning
    print("\n[2/4] Loading local multi-condition data...")
    organizer = IntentDataOrganizer(use_all_channels=False)  # Use 3 channels for compatibility
    data = organizer.load_all_data()

    X_local = data['X']
    y_local = data['y_binary']

    print(f"Local data shape: {X_local.shape}")

    # Align sample counts - trim PhysioNet to match local data
    target_samples = X_local.shape[2]
    if X_physionet.shape[2] > target_samples:
        X_physionet = X_physionet[:, :, :target_samples]
        print(f"Trimmed PhysioNet to {target_samples} samples to match local data")
    print(f"Intent samples: {np.sum(y_local == 1)}, Non-intent: {np.sum(y_local == 0)}")

    # Split local data
    X_train, X_test, y_train, y_test = train_test_split(
        X_local, y_local, test_size=0.3, stratify=y_local, random_state=42
    )

    # Balance training data
    X_train_bal, y_train_bal = organizer.balance_classes(X_train, y_train)
    print(f"Balanced training: {len(X_train_bal)} samples")

    # Step 3: Initialize and pretrain
    print("\n[3/4] Pretraining autoencoder on PhysioNet...")
    pipeline = TransferLearningPipeline(
        n_channels=3,
        n_samples=target_samples,  # Use aligned sample count
        latent_dim=128
    )

    pipeline.pretrain(X_physionet, epochs=30, batch_size=64)

    # Step 4: Fine-tune on local data
    print("\n[4/4] Fine-tuning on local multi-condition data...")
    pipeline.finetune(
        X_train_bal, y_train_bal,
        X_val=X_test, y_val=y_test,
        epochs=100,
        batch_size=16,
        freeze_encoder=True,
        patience=15
    )

    # Evaluate
    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)

    metrics = pipeline.get_detailed_metrics(X_test, y_test)

    print(f"\nTest Set ({len(X_test)} samples):")
    print(f"  Accuracy:    {metrics['accuracy']:.2%}")
    print(f"  Sensitivity: {metrics['sensitivity']:.2%} (intent detection)")
    print(f"  Specificity: {metrics['specificity']:.2%} (non-intent rejection)")
    print(f"  F1 Score:    {metrics['f1']:.2f}")

    # Save model
    pipeline.save("models/transfer_learning.pt")

    # Save results
    results_path = Path("results/transfer_learning_results.json")
    results_path.parent.mkdir(parents=True, exist_ok=True)

    # Convert numpy types for JSON serialization
    def convert_to_serializable(obj):
        if isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_serializable(v) for v in obj]
        elif isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    with open(results_path, 'w') as f:
        json.dump(convert_to_serializable({
            'metrics': metrics,
            'pretrain_epochs': len(pipeline.pretrain_history['loss']),
            'finetune_epochs': len(pipeline.finetune_history['train_loss'])
        }), f, indent=2)

    print(f"\nResults saved to {results_path}")

    return pipeline, metrics


if __name__ == "__main__":
    pipeline, metrics = main()
