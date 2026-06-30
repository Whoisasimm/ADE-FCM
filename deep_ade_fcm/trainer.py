"""
Training orchestrator for DeepADEFCM with curriculum learning and hyperparameter search.
"""
import torch
import numpy as np
from copy import deepcopy
from loguru import logger
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim
from itertools import product
from .deep_ade_fcm import DeepADEFCM


class DeepADEFCMTrainer:
    """
    Advanced trainer with curriculum learning, LR scheduling, and hyperparameter search.
    """

    def __init__(self, model, X, y=None):
        self.model = model
        self.X = X
        self.y = y
        self.best_model = None
        self.best_score = -np.inf

    def train_with_curriculum(self, warmup_epochs=30, lambda_final=1.0):
        """
        Curriculum training: gradually increase clustering loss weight.
        lambda(t) = lambda_final * (t / total_epochs)^2  (slow ramp-up)
        """
        original_lambda = self.model.lambda_cluster
        total_epochs = self.model.joint_epochs

        self.model.pretrain(self.X)

        X_tensor = torch.FloatTensor(self.X).to(self.model.device)
        dataset = TensorDataset(X_tensor)
        dataloader = DataLoader(dataset, batch_size=self.model.batch_size, shuffle=True)

        optimizer = optim.Adam(self.model.autoencoder.parameters(), lr=self.model.joint_lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=10
        )

        for epoch in range(total_epochs):
            progress = epoch / total_epochs
            if progress < warmup_epochs / total_epochs:
                self.model.lambda_cluster = 0.0
            else:
                ramp = (progress - warmup_epochs / total_epochs) / (1 - warmup_epochs / total_epochs)
                self.model.lambda_cluster = lambda_final * ramp ** 2

            epoch_loss = 0.0
            n_batches = 0
            for batch in dataloader:
                x_batch = batch[0].cpu().numpy()
                _, _, total_loss = self.model._joint_training_step(x_batch, optimizer)
                epoch_loss += total_loss
                n_batches += 1

            avg_loss = epoch_loss / n_batches
            scheduler.step(avg_loss)
            self.model.loss_history_['total'].append(avg_loss)

            if (epoch + 1) % 20 == 0:
                logger.info(f"Curriculum Epoch {epoch + 1}/{total_epochs}: "
                           f"lambda={self.model.lambda_cluster:.3f}, Loss={avg_loss:.4f}")

        self.model.lambda_cluster = original_lambda
        
        # Final clustering step to set labels_
        Z = self.model._get_latent(self.X)
        self.model.fcm_model.fit(Z)
        self.model.Z_ = Z
        self.model.labels_ = self.model.fcm_model.labels_
        return self.model

    def train_with_amp(self, scaler_init=2.0**16):
        """Train with automatic mixed precision."""
        self.model.pretrain(self.X)
        X_tensor = torch.FloatTensor(self.X).to(self.model.device)
        dataset = TensorDataset(X_tensor)
        dataloader = DataLoader(dataset, batch_size=self.model.batch_size, shuffle=True)
        optimizer = optim.Adam(self.model.autoencoder.parameters(), lr=self.model.joint_lr)
        scaler = torch.amp.GradScaler(init_scale=scaler_init)

        for epoch in range(self.model.joint_epochs):
            epoch_loss = 0.0
            n_batches = 0
            for batch in dataloader:
                x_batch = batch[0].cpu().numpy()
                X_t = torch.FloatTensor(x_batch).to(self.model.device)
                optimizer.zero_grad()
                with torch.amp.autocast(device_type=self.model.device):
                    output = self.model.autoencoder(X_t)
                    X_recon, Z = output[0], output[1]
                    recon_loss = torch.nn.MSELoss()(X_recon, X_t)
                    cluster_loss = torch.tensor(0.0)
                    total_loss = recon_loss
                scaler.scale(total_loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.autoencoder.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                epoch_loss += total_loss.item()
                n_batches += 1

            avg_loss = epoch_loss / n_batches
            self.model.loss_history_['total'].append(avg_loss)
            if (epoch + 1) % 25 == 0:
                logger.info(f"AMP Epoch {epoch + 1}/{self.model.joint_epochs}: Loss={avg_loss:.4f}")

        self.model.Z_ = self.model._get_latent(self.X)
        self.model.fcm_model.fit(self.model.Z_)
        self.model.labels_ = self.model.fcm_model.labels_
        return self.model

    def train_with_early_stopping(self, patience=15, min_delta=1e-4):
        """Train with early stopping based on total loss plateau."""
        self.model.pretrain(self.X)
        X_tensor = torch.FloatTensor(self.X).to(self.model.device)
        dataset = TensorDataset(X_tensor)
        dataloader = DataLoader(dataset, batch_size=self.model.batch_size, shuffle=True)
        optimizer = optim.Adam(self.model.autoencoder.parameters(), lr=self.model.joint_lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=10
        )

        best_loss = float('inf')
        patience_counter = 0

        for epoch in range(self.model.joint_epochs):
            epoch_loss = 0.0
            n_batches = 0
            for batch in dataloader:
                x_batch = batch[0].cpu().numpy()
                _, _, total_loss = self.model._joint_training_step(x_batch, optimizer)
                epoch_loss += total_loss
                n_batches += 1

            avg_loss = epoch_loss / n_batches
            scheduler.step(avg_loss)
            self.model.loss_history_['total'].append(avg_loss)

            if avg_loss < best_loss - min_delta:
                best_loss = avg_loss
                patience_counter = 0
                self.best_model = deepcopy(self.model.autoencoder.state_dict())
                self.best_score = -avg_loss
            else:
                patience_counter += 1

            if patience_counter >= patience:
                logger.info(f"Early stopping triggered at epoch {epoch + 1}")
                if self.best_model is not None:
                    self.model.autoencoder.load_state_dict(self.best_model)
                break

            if (epoch + 1) % 25 == 0:
                logger.info(f"ES Epoch {epoch + 1}: Loss={avg_loss:.4f}, patience={patience_counter}")

        self.model.Z_ = self.model._get_latent(self.X)
        self.model.fcm_model.fit(self.model.Z_)
        self.model.labels_ = self.model.fcm_model.labels_
        return self.model

    def save_checkpoint(self, path, epoch, optimizer, loss):
        """Save training checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.autoencoder.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
            'best_score': self.best_score,
        }
        torch.save(checkpoint, path)
        logger.info(f"Checkpoint saved to {path}")

    def load_checkpoint(self, path, optimizer=None):
        """Load training checkpoint."""
        checkpoint = torch.load(path, map_location=self.model.device)
        self.model.autoencoder.load_state_dict(checkpoint['model_state_dict'])
        if optimizer is not None:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.best_score = checkpoint.get('best_score', -np.inf)
        logger.info(f"Checkpoint loaded from {path} (epoch {checkpoint.get('epoch', '?')})")
        return checkpoint

    def cross_validate(self, n_folds=3, **train_kwargs):
        """K-fold cross-validation wrapper."""
        from sklearn.model_selection import KFold
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
        fold_scores = []

        for fold, (train_idx, val_idx) in enumerate(kf.split(self.X)):
            X_train, X_val = self.X[train_idx], self.X[val_idx]
            fold_model = DeepADEFCM(
                input_dim=self.X.shape[1],
                latent_dim=self.model.latent_dim,
                n_clusters=self.model.n_clusters if self.model.n_clusters != 'auto' else 2,
                lambda_cluster=self.model.lambda_cluster,
                ae_epochs=self.model.ae_epochs,
                joint_epochs=self.model.joint_epochs,
                batch_size=self.model.batch_size,
                use_vae=self.model.use_vae,
                random_state=42 + fold,
                verbose=False
            )
            fold_trainer = DeepADEFCMTrainer(fold_model, X_train, self.y[train_idx] if self.y is not None else None)
            fold_trainer.train_with_curriculum(**train_kwargs)
            val_Z = fold_model._get_latent(X_val)
            val_labels = fold_model.fcm_model.predict(val_Z)

            from sklearn.metrics import silhouette_score
            if len(set(val_labels)) > 1:
                sil = silhouette_score(val_Z, val_labels)
            else:
                sil = 0.0
            recon_err = fold_model._compute_reconstruction_error(X_val)
            fold_scores.append({'silhouette': sil, 'reconstruction_error': recon_err, 'fold': fold})
            logger.info(f"Fold {fold + 1}/{n_folds}: Silhouette={sil:.4f}, ReconErr={recon_err:.6f}")

        avg_sil = np.mean([s['silhouette'] for s in fold_scores])
        logger.info(f"Cross-validation complete. Avg Silhouette={avg_sil:.4f}")
        return fold_scores

    def grid_search(self, param_grid, cv_folds=3):
        """Hyperparameter grid search with cross-validation."""
        keys = list(param_grid.keys())
        values = list(param_grid.values())

        results = []
        for combination in product(*values):
            params = dict(zip(keys, combination))
            logger.info(f"Testing: {params}")

            fold_scores = []
            for fold in range(cv_folds):
                model = DeepADEFCM(
                    input_dim=self.X.shape[1],
                    latent_dim=params.get('latent_dim', 10),
                    n_clusters=params.get('n_clusters', 'auto'),
                    lambda_cluster=params.get('lambda_cluster', 0.5),
                    ae_epochs=params.get('ae_epochs', 50),
                    joint_epochs=params.get('joint_epochs', 50),
                    batch_size=params.get('batch_size', 256),
                    hidden_dims=params.get('hidden_dims', None),
                    use_vae=params.get('use_vae', False),
                    random_state=42 + fold,
                    verbose=False
                )
                model.fit(self.X)
                sil = model._silhouette_score()
                recon_err = model._compute_reconstruction_error(self.X)
                fold_scores.append({'silhouette': sil, 'reconstruction_error': recon_err})

            avg_sil = np.mean([s['silhouette'] for s in fold_scores])
            results.append({**params, 'silhouette': avg_sil})
            logger.info(f"  Avg Silhouette: {avg_sil:.4f}")

        best_params = max(results, key=lambda r: r['silhouette'])
        logger.info(f"Best params: {best_params}")
        return results, best_params
