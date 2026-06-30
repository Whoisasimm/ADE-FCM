"""
DeepADEFCM: Joint Deep Representation Learning and Fuzzy Clustering.

Key innovations:
1. Autoencoder learns latent representation optimized for clustering
2. Joint loss: L = reconstruction_loss + lambda * clustering_loss
3. Latent-space adaptive fuzzifier m(t)
4. End-to-end training with cluster-preserving gradients
5. Auto-discovery of K in latent space
6. Latent space XAI (explain clusters in original feature space)
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from copy import deepcopy
from loguru import logger

# Import autoencoder
from .autoencoder import Autoencoder, VAutoencoder, pretrain_autoencoder

# Import ADE-FCM components
from novel_algorithm.ade_fcm import ADEFCM
from novel_algorithm.adaptive_params import AdaptiveFuzzifier, DynamicThreshold, EarlyStopping
from novel_algorithm.auto_cluster import AutomaticClusterDiscovery


class DeepADEFCM:
    """
    Deep Adaptive Distributed Explainable Fuzzy C-Means.
    
    Jointly optimizes:
    - Autoencoder reconstruction loss (good latent representation)
    - ADE-FCM clustering loss (well-separated clusters in latent space)
    
    Args:
        input_dim: Input feature dimensionality
        latent_dim: Latent space dimensionality (default: 10)
        n_clusters: Number of clusters or 'auto' for automatic discovery
        lambda_cluster: Weight of clustering loss (0.1-1.0, default: 0.5)
        ae_epochs: Pretraining epochs for autoencoder
        joint_epochs: Joint training epochs
        ae_lr: Autoencoder learning rate
        joint_lr: Joint training learning rate
        batch_size: Training batch size
        hidden_dims: Encoder hidden layer sizes
        use_vae: Use variational autoencoder (default: False)
        device: 'cpu' or 'cuda' (auto-detected if None)
        random_state: Random seed
        verbose: Print progress
    """
    
    def __init__(self, input_dim=None, latent_dim=10, n_clusters='auto',
                 lambda_cluster=0.5, ae_epochs=100, joint_epochs=100,
                 ae_lr=1e-3, joint_lr=5e-4, batch_size=256,
                 hidden_dims=None, use_vae=False, device=None,
                 random_state=42, verbose=True):
        
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.n_clusters = n_clusters
        self.lambda_cluster = lambda_cluster
        self.ae_epochs = ae_epochs
        self.joint_epochs = joint_epochs
        self.ae_lr = ae_lr
        self.joint_lr = joint_lr
        self.batch_size = batch_size
        self.hidden_dims = hidden_dims
        self.use_vae = use_vae
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.random_state = random_state
        self.verbose = verbose
        
        self.autoencoder = None
        self.fcm_model = ADEFCM(
            n_clusters=n_clusters,
            init_method='kmeans++',
            m='adaptive',
            epsilon='dynamic',
            random_state=random_state,
            verbose=False
        )
        self.adaptive_m = AdaptiveFuzzifier()
        self.loss_history_ = {'reconstruction': [], 'clustering': [], 'total': []}
        self.Z_ = None  # Learned latent representation
        self.labels_ = None
        self.n_iter_ = 0
        
    def _build_autoencoder(self):
        """Build autoencoder based on configuration."""
        ae_class = VAutoencoder if self.use_vae else Autoencoder
        self.autoencoder = ae_class(
            input_dim=self.input_dim,
            latent_dim=self.latent_dim,
            hidden_dims=self.hidden_dims
        )
        logger.info(f"Built {'VAutoencoder' if self.use_vae else 'Autoencoder'}: "
                    f"{self.input_dim} -> {self.latent_dim}")
    
    def _get_latent(self, X, batch_size=None):
        """Get latent representation for entire dataset."""
        self.autoencoder.eval()
        if batch_size is None:
            batch_size = self.batch_size * 4
        
        X_tensor = torch.FloatTensor(X).to(self.device)
        dataset = TensorDataset(X_tensor)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        
        latents = []
        with torch.no_grad():
            for batch in dataloader:
                x = batch[0]
                z = self.autoencoder.encode(x)
                latents.append(z.cpu().numpy())
        
        return np.concatenate(latents, axis=0)
    
    def _compute_clustering_loss(self, Z, centers, U, m):
        """Compute fuzzy clustering loss in latent space.
        
        L_cluster = sum_i sum_j u_ij^m * ||z_i - v_j||^2
        """
        n, c = U.shape
        loss = 0.0
        for j in range(c):
            diff = Z - centers[j]
            dists = np.sqrt(np.sum(diff ** 2, axis=1))
            loss += np.sum(U[:, j] ** m * dists)
        return loss
    
    def pretrain(self, X):
        """Pretrain autoencoder with reconstruction loss."""
        logger.info(f"Pretraining autoencoder for {self.ae_epochs} epochs...")
        if self.autoencoder is None:
            self._build_autoencoder()
        
        losses = pretrain_autoencoder(
            self.autoencoder, X,
            epochs=self.ae_epochs,
            batch_size=self.batch_size,
            lr=self.ae_lr,
            device=self.device,
            verbose=self.verbose
        )
        
        recon_error = self._compute_reconstruction_error(X)
        logger.info(f"Pretrain complete. Reconstruction error: {recon_error:.6f}")
        return losses
    
    def _compute_reconstruction_error(self, X):
        """Compute MSE reconstruction error."""
        from .autoencoder import compute_reconstruction_error
        return compute_reconstruction_error(self.autoencoder, X, self.device)
    
    def _joint_training_step(self, X, optimizer):
        """Single step of joint training on a batch."""
        X_tensor = torch.FloatTensor(X).to(self.device)
        
        optimizer.zero_grad()
        
        # Forward through autoencoder
        output = self.autoencoder(X_tensor)
        if len(output) == 2:  # Autoencoder
            X_recon, Z = output
        else:  # VAutoencoder
            X_recon, Z, mu, log_var = output
        
        # Reconstruction loss
        recon_loss = nn.MSELoss()(X_recon, X_tensor)
        
        # Clustering loss on latent representation
        Z_np = Z.detach().cpu().numpy()
        
        # Update FCM on latent representation
        self.fcm_model.fit(Z_np)
        centers = self.fcm_model.centers_
        U = self.fcm_model.U_
        labels = self.fcm_model.labels_
        
        m_t = self.adaptive_m(self.n_iter_)
        
        # Compute clustering loss (differentiable approximation)
        clustering_loss = 0.0
        for j in range(centers.shape[0]):
            diff = Z - torch.FloatTensor(centers[j]).to(self.device)
            dists = torch.sqrt(torch.sum(diff ** 2, dim=1) + 1e-10)
            um = torch.FloatTensor(U[:, j] ** m_t).to(self.device)
            clustering_loss += torch.sum(um * dists)
        
        # KL loss for VAE
        total_loss = recon_loss + self.lambda_cluster * clustering_loss
        if len(output) == 4:
            kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
            total_loss += 0.001 * kl_loss
        
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.autoencoder.parameters(), 1.0)
        optimizer.step()
        
        self.n_iter_ += 1
        return recon_loss.item(), clustering_loss.item(), total_loss.item()
    
    def fit(self, X, y=None):
        """
        Full training pipeline:
        1. Pretrain autoencoder
        2. Joint optimization of reconstruction + clustering
        
        Args:
            X: numpy array of shape (n_samples, n_features)
            y: optional true labels for evaluation
        
        Returns:
            self
        """
        if self.input_dim is None:
            self.input_dim = X.shape[1]
        
        np.random.seed(self.random_state)
        torch.manual_seed(self.random_state)
        
        # Step 1: Build and pretrain autoencoder
        self.pretrain(X)
        
        # Step 2: Get initial latent representation
        Z = self._get_latent(X)
        
        # Step 3: Auto-discover K if needed
        if self.n_clusters == 'auto':
            discover = AutomaticClusterDiscovery()
            k_range = range(2, min(15, int(np.sqrt(Z.shape[0])) + 1))
            self.n_clusters = discover.consensus_search(Z, k_range, self.fcm_model)
            self.fcm_model.n_clusters = self.n_clusters
            logger.info(f"Auto-discovered K = {self.n_clusters} in latent space")
        
        # Step 4: Initialize FCM on latent space
        self.fcm_model.fit(Z)
        
        # Step 5: Joint training
        logger.info(f"Joint training for {self.joint_epochs} epochs "
                    f"(lambda={self.lambda_cluster}, latent_dim={self.latent_dim})...")
        
        optimizer = optim.Adam(self.autoencoder.parameters(), lr=self.joint_lr)
        
        X_tensor = torch.FloatTensor(X).to(self.device)
        dataset = TensorDataset(X_tensor)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        for epoch in range(self.joint_epochs):
            epoch_recon = 0.0
            epoch_cluster = 0.0
            epoch_total = 0.0
            n_batches = 0
            
            for batch in dataloader:
                x_batch = batch[0].cpu().numpy()
                r_loss, c_loss, t_loss = self._joint_training_step(x_batch, optimizer)
                epoch_recon += r_loss
                epoch_cluster += c_loss
                epoch_total += t_loss
                n_batches += 1
            
            self.loss_history_['reconstruction'].append(epoch_recon / n_batches)
            self.loss_history_['clustering'].append(epoch_cluster / n_batches)
            self.loss_history_['total'].append(epoch_total / n_batches)
            
            if self.verbose and (epoch + 1) % 25 == 0:
                logger.info(f"Epoch {epoch+1}/{self.joint_epochs}: "
                           f"Recon={self.loss_history_['reconstruction'][-1]:.4f}, "
                           f"Cluster={self.loss_history_['clustering'][-1]:.4f}, "
                           f"Total={self.loss_history_['total'][-1]:.4f}")
        
        # Step 6: Final clustering on learned latent space
        self.Z_ = self._get_latent(X)
        self.fcm_model.fit(self.Z_)
        self.labels_ = self.fcm_model.labels_
        
        recon_error = self._compute_reconstruction_error(X)
        logger.info(f"Training complete. Final reconstruction error: {recon_error:.6f}")
        logger.info(f"Final latent clusters: {self.n_clusters}, "
                    f"Silhouette: {self._silhouette_score():.4f}")
        
        return self
    
    def _silhouette_score(self):
        """Compute silhouette score in latent space."""
        from sklearn.metrics import silhouette_score
        if self.labels_ is not None and len(set(self.labels_)) > 1:
            return silhouette_score(self.Z_, self.labels_)
        return 0.0
    
    def predict(self, X):
        """Predict cluster labels for new data.
        
        Steps: encode with autoencoder -> cluster in latent space.
        """
        Z = self._get_latent(X)
        return self.fcm_model.predict(Z)
    
    def fit_predict(self, X, y=None):
        self.fit(X, y)
        return self.labels_
    
    def transform(self, X):
        """Transform data to latent representation."""
        return self._get_latent(X)
    
    def get_explanation(self, cluster_id, feature_names=None):
        """Explain cluster in original feature space.
        
        Maps latent-space clusters back to original features via decoder gradients.
        """
        if feature_names is None:
            feature_names = [f"f{i}" for i in range(self.input_dim)]
        
        center_z = self.fcm_model.centers_[cluster_id]
        center_z_tensor = torch.FloatTensor(center_z).unsqueeze(0).to(self.device)
        
        # Decode center to original space
        self.autoencoder.eval()
        with torch.no_grad():
            center_x = self.autoencoder.decode(center_z_tensor).cpu().numpy()[0]
        
        # Feature importance: gradient of decoder output w.r.t. latent -> original
        center_z_tensor.requires_grad_(True)
        decoded = self.autoencoder.decode(center_z_tensor)
        
        # Compute feature sensitivity
        feature_importance = {}
        for i, name in enumerate(feature_names):
            grad = torch.autograd.grad(decoded[0, i], center_z_tensor, retain_graph=True)[0]
            importance = float(torch.norm(grad).item())
            feature_importance[name] = importance
        
        # Normalize
        total = sum(feature_importance.values())
        if total > 0:
            feature_importance = {k: v / total for k, v in feature_importance.items()}
        
        return {
            'cluster_id': cluster_id,
            'latent_center': center_z.tolist(),
            'decoded_center': center_x.tolist(),
            'feature_importance': feature_importance,
            'natural_description': self._describe_cluster(cluster_id, feature_names, feature_importance)
        }
    
    def _describe_cluster(self, cluster_id, feature_names, importances=None):
        """Generate natural language description of a cluster."""
        if importances is None:
            importances = self.get_explanation(cluster_id, feature_names)['feature_importance']
        top_features = sorted(importances.items(), key=lambda x: -x[1])[:3]
        
        mask = self.labels_ == cluster_id
        size = int(mask.sum())
        pct = size / len(self.labels_) * 100
        
        desc = f"Cluster {cluster_id} contains {size} points ({pct:.1f}% of total). "
        desc += "Key features: " + ", ".join([f"{name} ({imp:.2f})" for name, imp in top_features]) + ". "
        
        if self.Z_ is not None:
            z_cluster = self.Z_[mask]
            if len(z_cluster) > 0:
                desc += f"Latent space: mean={np.mean(z_cluster, axis=0).round(3)}, "
                desc += f"spread={np.std(z_cluster, axis=0).round(3)}"
        
        return desc
    
    def get_latent_representation(self):
        """Return the learned latent representation."""
        return self.Z_
    
    def save(self, path):
        """Save model checkpoint."""
        import pickle
        checkpoint = {
            'autoencoder_state': self.autoencoder.state_dict(),
            'fcm_model': self.fcm_model,
            'input_dim': self.input_dim,
            'latent_dim': self.latent_dim,
            'n_clusters': self.n_clusters,
            'labels_': self.labels_,
            'loss_history_': self.loss_history_
        }
        torch.save(checkpoint, path + '.pt')
        logger.info(f"Model saved to {path}.pt")
    
    def load(self, path):
        """Load model checkpoint."""
        checkpoint = torch.load(path + '.pt', map_location=self.device, weights_only=False)
        self.input_dim = checkpoint['input_dim']
        self.latent_dim = checkpoint['latent_dim']
        self.n_clusters = checkpoint['n_clusters']
        self.labels_ = checkpoint['labels_']
        self.loss_history_ = checkpoint['loss_history_']
        self._build_autoencoder()
        self.autoencoder.load_state_dict(checkpoint['autoencoder_state'])
        self.fcm_model = checkpoint['fcm_model']
        logger.info(f"Model loaded from {path}.pt")
        return self
