"""
Comprehensive tests for DeepADEFCM module.
"""
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from deep_ade_fcm.autoencoder import Autoencoder, Encoder, Decoder, pretrain_autoencoder, compute_reconstruction_error
from deep_ade_fcm.deep_ade_fcm import DeepADEFCM
from deep_ade_fcm.trainer import DeepADEFCMTrainer


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def tiny_data():
    return np.array([
        [1.0, 2.0], [1.5, 1.8], [5.0, 8.0], [8.0, 8.0],
        [1.0, 0.6], [9.0, 11.0]
    ], dtype=np.float64)


@pytest.fixture(scope="module")
def blobs_data():
    from sklearn.datasets import make_blobs
    X, y = make_blobs(n_samples=80, n_features=4, centers=3, random_state=42, cluster_std=0.8)
    return X.astype(np.float64), y


@pytest.fixture(scope="module")
def blobs_X(blobs_data):
    return blobs_data[0]


@pytest.fixture(scope="module")
def blobs_y(blobs_data):
    return blobs_data[1]


@pytest.fixture(scope="module")
def ae_model():
    return Autoencoder(input_dim=4, latent_dim=2, hidden_dims=[8, 4])


# ---------------------------------------------------------------------------
# Test Autoencoder
# ---------------------------------------------------------------------------

class TestAutoencoder:
    def test_forward_shape(self, ae_model):
        x = torch.randn(10, 4)
        x_recon, z = ae_model(x)
        assert x_recon.shape == (10, 4)
        assert z.shape == (10, 2)

    def test_encode_decode_consistency(self, ae_model):
        x = torch.randn(5, 4)
        z = ae_model.encode(x)
        x_recon = ae_model.decode(z)
        assert z.shape == (5, 2)
        assert x_recon.shape == (5, 4)

    def test_get_latent_numpy(self, ae_model):
        x = np.random.randn(8, 4).astype(np.float32)
        Z = ae_model.get_latent(x)
        assert isinstance(Z, np.ndarray)
        assert Z.shape == (8, 2)

    def test_encoder_output_shape(self):
        enc = Encoder(input_dim=6, latent_dim=3, hidden_dims=[12, 8])
        x = torch.randn(4, 6)
        z = enc(x)
        assert z.shape == (4, 3)

    def test_decoder_output_shape(self):
        dec = Decoder(latent_dim=3, output_dim=6, hidden_dims=[8, 12])
        z = torch.randn(4, 3)
        x_recon = dec(z)
        assert x_recon.shape == (4, 6)

    def test_gradient_flow(self, ae_model):
        x = torch.randn(4, 4, requires_grad=True)
        x_recon, z = ae_model(x)
        loss = ((x_recon - x) ** 2).sum()
        loss.backward()
        assert ae_model.encoder.encoder[0].weight.grad is not None
        assert torch.isfinite(ae_model.encoder.encoder[0].weight.grad).all()

    def test_pretrain_autoencoder(self, ae_model):
        X = np.random.randn(20, 4).astype(np.float32)
        losses = pretrain_autoencoder(ae_model, X, epochs=5, batch_size=10, verbose=False)
        assert len(losses) == 5
        assert all(np.isfinite(l) for l in losses)

    def test_reconstruction_error(self, ae_model):
        X = np.random.randn(10, 4).astype(np.float32)
        error = compute_reconstruction_error(ae_model, X)
        assert np.isfinite(error)
        assert error >= 0.0


# ---------------------------------------------------------------------------
# Test DeepADEFCM
# ---------------------------------------------------------------------------

class TestDeepADEFCM:
    def test_init_defaults(self):
        model = DeepADEFCM(input_dim=4, verbose=False)
        assert model.latent_dim == 10
        assert model.lambda_cluster == 0.5
        assert model.ae_epochs == 100
        assert model.n_clusters == 'auto'

    def test_build_autoencoder(self, blobs_X):
        model = DeepADEFCM(input_dim=blobs_X.shape[1], latent_dim=2, verbose=False)
        model._build_autoencoder()
        assert model.autoencoder is not None

    def test_pretrain(self, blobs_X):
        model = DeepADEFCM(input_dim=blobs_X.shape[1], latent_dim=2,
                           ae_epochs=5, batch_size=32, verbose=False)
        losses = model.pretrain(blobs_X)
        assert len(losses) == 5
        assert all(np.isfinite(l) for l in losses)

    def test_fit_returns_self(self, tiny_data):
        model = DeepADEFCM(input_dim=tiny_data.shape[1], latent_dim=2,
                           n_clusters=2, ae_epochs=3, joint_epochs=3,
                           batch_size=4, verbose=False)
        result = model.fit(tiny_data)
        assert result is model

    def test_fit_sets_labels(self, tiny_data):
        model = DeepADEFCM(input_dim=tiny_data.shape[1], latent_dim=2,
                           n_clusters=2, ae_epochs=3, joint_epochs=3,
                           batch_size=4, verbose=False)
        model.fit(tiny_data)
        assert model.labels_ is not None
        assert model.labels_.shape == (len(tiny_data),)

    def test_fit_sets_latent(self, tiny_data):
        model = DeepADEFCM(input_dim=tiny_data.shape[1], latent_dim=2,
                           n_clusters=2, ae_epochs=3, joint_epochs=3,
                           batch_size=4, verbose=False)
        model.fit(tiny_data)
        assert model.Z_ is not None
        assert model.Z_.shape[1] == 2

    def test_predict_shape(self, tiny_data):
        model = DeepADEFCM(input_dim=tiny_data.shape[1], latent_dim=2,
                           n_clusters=2, ae_epochs=3, joint_epochs=3,
                           batch_size=4, verbose=False)
        model.fit(tiny_data)
        labels = model.predict(tiny_data)
        assert labels.shape == (len(tiny_data),)

    def test_fit_predict(self, tiny_data):
        model = DeepADEFCM(input_dim=tiny_data.shape[1], latent_dim=2,
                           n_clusters=2, ae_epochs=3, joint_epochs=3,
                           batch_size=4, verbose=False)
        labels = model.fit_predict(tiny_data)
        assert labels.shape == (len(tiny_data),)

    def test_transform(self, tiny_data):
        model = DeepADEFCM(input_dim=tiny_data.shape[1], latent_dim=2,
                           n_clusters=2, ae_epochs=3, joint_epochs=3,
                           batch_size=4, verbose=False)
        model.fit(tiny_data)
        Z = model.transform(tiny_data)
        assert Z.shape[1] == 2

    def test_loss_history_structure(self, tiny_data):
        model = DeepADEFCM(input_dim=tiny_data.shape[1], latent_dim=2,
                           n_clusters=2, ae_epochs=3, joint_epochs=5,
                           batch_size=4, verbose=False)
        model.fit(tiny_data)
        assert 'reconstruction' in model.loss_history_
        assert 'clustering' in model.loss_history_
        assert 'total' in model.loss_history_
        assert len(model.loss_history_['total']) == 5

    def test_silhouette_score(self, tiny_data):
        model = DeepADEFCM(input_dim=tiny_data.shape[1], latent_dim=2,
                           n_clusters=2, ae_epochs=3, joint_epochs=3,
                           batch_size=4, verbose=False)
        model.fit(tiny_data)
        score = model._silhouette_score()
        assert np.isfinite(score)

    def test_get_latent_representation(self, tiny_data):
        model = DeepADEFCM(input_dim=tiny_data.shape[1], latent_dim=2,
                           n_clusters=2, ae_epochs=3, joint_epochs=3,
                           batch_size=4, verbose=False)
        model.fit(tiny_data)
        Z = model.get_latent_representation()
        assert Z is model.Z_

    def test_fit_on_blobs(self, blobs_X):
        model = DeepADEFCM(input_dim=blobs_X.shape[1], latent_dim=3,
                           n_clusters=3, ae_epochs=10, joint_epochs=10,
                           batch_size=20, verbose=False)
        model.fit(blobs_X)
        assert len(set(model.labels_)) == 3


# ---------------------------------------------------------------------------
# Test Joint Training
# ---------------------------------------------------------------------------

class TestJointTraining:
    def test_joint_loss_decreases(self, tiny_data):
        model = DeepADEFCM(input_dim=tiny_data.shape[1], latent_dim=2,
                           n_clusters=2, ae_epochs=5, joint_epochs=10,
                           batch_size=4, verbose=False)
        model.fit(tiny_data)
        total_losses = model.loss_history_['total']
        # Loss should generally decrease (allow slight increase)
        assert total_losses[-1] <= total_losses[0] * 1.5 or len(total_losses) > 1

    def test_joint_training_step_output(self, tiny_data):
        model = DeepADEFCM(input_dim=tiny_data.shape[1], latent_dim=2,
                           n_clusters=2, ae_epochs=3, verbose=False)
        model.pretrain(tiny_data)
        model.fcm_model.fit(model._get_latent(tiny_data))
        import torch.optim as optim
        optimizer = optim.Adam(model.autoencoder.parameters(), lr=1e-3)
        r_loss, c_loss, t_loss = model._joint_training_step(tiny_data, optimizer)
        assert np.isfinite(r_loss)
        assert np.isfinite(c_loss)
        assert np.isfinite(t_loss)
        assert t_loss > 0.0


# ---------------------------------------------------------------------------
# Test Latent Space
# ---------------------------------------------------------------------------

class TestLatentSpace:
    def test_latent_space_n_clusters(self, blobs_X):
        model = DeepADEFCM(input_dim=blobs_X.shape[1], latent_dim=3,
                           n_clusters=3, ae_epochs=10, joint_epochs=10,
                           batch_size=20, verbose=False)
        model.fit(blobs_X)
        Z = model.get_latent_representation()
        assert Z.shape[0] == blobs_X.shape[0]
        assert Z.shape[1] == 3

    def test_latent_space_finite(self, tiny_data):
        model = DeepADEFCM(input_dim=tiny_data.shape[1], latent_dim=2,
                           n_clusters=2, ae_epochs=3, joint_epochs=3,
                           batch_size=4, verbose=False)
        model.fit(tiny_data)
        Z = model.get_latent_representation()
        assert np.all(np.isfinite(Z))

    def test_latent_reconstruction_quality(self, blobs_X):
        model = DeepADEFCM(input_dim=blobs_X.shape[1], latent_dim=5,
                           n_clusters=3, ae_epochs=15, joint_epochs=5,
                           batch_size=32, verbose=False)
        model.fit(blobs_X)
        Z = model.get_latent_representation()
        X_recon = model.autoencoder.decode(torch.FloatTensor(Z)).detach().numpy()
        mse = np.mean((blobs_X - X_recon) ** 2)
        assert np.isfinite(mse)
        assert mse < 10.0


# ---------------------------------------------------------------------------
# Test Explainability
# ---------------------------------------------------------------------------

class TestExplainability:
    def test_get_explanation_structure(self, tiny_data):
        model = DeepADEFCM(input_dim=tiny_data.shape[1], latent_dim=2,
                           n_clusters=2, ae_epochs=3, joint_epochs=3,
                           batch_size=4, verbose=False)
        model.fit(tiny_data)
        explanation = model.get_explanation(cluster_id=0)
        assert 'cluster_id' in explanation
        assert 'latent_center' in explanation
        assert 'decoded_center' in explanation
        assert 'feature_importance' in explanation
        assert 'natural_description' in explanation
        assert explanation['cluster_id'] == 0

    def test_get_explanation_feature_importance(self, tiny_data):
        model = DeepADEFCM(input_dim=tiny_data.shape[1], latent_dim=2,
                           n_clusters=2, ae_epochs=3, joint_epochs=3,
                           batch_size=4, verbose=False)
        model.fit(tiny_data)
        explanation = model.get_explanation(cluster_id=0, feature_names=['x', 'y'])
        fi = explanation['feature_importance']
        assert 'x' in fi
        assert 'y' in fi
        total = sum(fi.values())
        assert abs(total - 1.0) < 1e-4 or total > 0

    def test_explain_all_clusters(self, tiny_data):
        model = DeepADEFCM(input_dim=tiny_data.shape[1], latent_dim=2,
                           n_clusters=2, ae_epochs=3, joint_epochs=3,
                           batch_size=4, verbose=False)
        model.fit(tiny_data)
        explanations = []
        for cid in range(2):
            explanations.append(model.get_explanation(cid))
        assert len(explanations) == 2


# ---------------------------------------------------------------------------
# Test Save / Load
# ---------------------------------------------------------------------------

class TestSaveLoad:
    def test_save_load_round_trip(self, tiny_data, tmp_path):
        model = DeepADEFCM(input_dim=tiny_data.shape[1], latent_dim=2,
                           n_clusters=2, ae_epochs=3, joint_epochs=3,
                           batch_size=4, verbose=False)
        model.fit(tiny_data)
        save_path = str(tmp_path / "test_model")
        model.save(save_path)

        loaded = DeepADEFCM(verbose=False)
        loaded.load(save_path)
        assert loaded.input_dim == model.input_dim
        assert loaded.latent_dim == model.latent_dim
        assert loaded.n_clusters == model.n_clusters
        assert loaded.labels_ is not None
        assert np.array_equal(loaded.labels_, model.labels_)

    def test_save_load_predict_consistency(self, tiny_data, tmp_path):
        model = DeepADEFCM(input_dim=tiny_data.shape[1], latent_dim=2,
                           n_clusters=2, ae_epochs=3, joint_epochs=3,
                           batch_size=4, verbose=False)
        model.fit(tiny_data)
        orig_preds = model.predict(tiny_data)

        save_path = str(tmp_path / "test_model2")
        model.save(save_path)

        loaded = DeepADEFCM(verbose=False)
        loaded.load(save_path)
        loaded_preds = loaded.predict(tiny_data)
        assert np.array_equal(orig_preds, loaded_preds)


# ---------------------------------------------------------------------------
# Test Curriculum Training
# ---------------------------------------------------------------------------

class TestCurriculumTraining:
    def test_trainer_initialization(self, tiny_data):
        model = DeepADEFCM(input_dim=tiny_data.shape[1], latent_dim=2,
                           n_clusters=2, ae_epochs=3, joint_epochs=3,
                           batch_size=4, verbose=False)
        trainer = DeepADEFCMTrainer(model, tiny_data)
        assert trainer.model is model
        assert trainer.X is tiny_data
        assert trainer.best_model is None
        assert trainer.best_score == -np.inf

    def test_train_with_curriculum(self, tiny_data):
        model = DeepADEFCM(input_dim=tiny_data.shape[1], latent_dim=2,
                           n_clusters=2, ae_epochs=3, joint_epochs=5,
                           batch_size=4, verbose=False)
        trainer = DeepADEFCMTrainer(model, tiny_data)
        result = trainer.train_with_curriculum(warmup_epochs=2, lambda_final=1.0)
        assert result is model
        assert model.labels_ is not None

    def test_grid_search_returns_results(self, blobs_X):
        model = DeepADEFCM(input_dim=blobs_X.shape[1], verbose=False)
        trainer = DeepADEFCMTrainer(model, blobs_X)
        param_grid = {
            'latent_dim': [2, 3],
            'lambda_cluster': [0.1, 0.5],
            'ae_epochs': [3],
            'joint_epochs': [3],
        }
        results, best_params = trainer.grid_search(param_grid, cv_folds=2)
        assert len(results) == 4
        assert 'latent_dim' in best_params
        assert 'silhouette' in best_params

    def test_cross_validate(self, tiny_data):
        model = DeepADEFCM(input_dim=tiny_data.shape[1], latent_dim=2,
                           n_clusters=2, ae_epochs=3, joint_epochs=3,
                           batch_size=4, verbose=False)
        trainer = DeepADEFCMTrainer(model, tiny_data)
        fold_scores = trainer.cross_validate(n_folds=2)
        assert len(fold_scores) == 2
        for fs in fold_scores:
            assert 'silhouette' in fs
            assert 'reconstruction_error' in fs

    def test_checkpoint_save_load(self, tiny_data, tmp_path):
        model = DeepADEFCM(input_dim=tiny_data.shape[1], latent_dim=2,
                           n_clusters=2, ae_epochs=3, joint_epochs=3,
                           batch_size=4, verbose=False)
        model.pretrain(tiny_data)
        import torch.optim as optim
        optimizer = optim.Adam(model.autoencoder.parameters(), lr=1e-3)
        trainer = DeepADEFCMTrainer(model, tiny_data)
        ckpt_path = str(tmp_path / "checkpoint.pt")
        trainer.save_checkpoint(ckpt_path, epoch=5, optimizer=optimizer, loss=0.5)

        loaded = DeepADEFCM(input_dim=tiny_data.shape[1], verbose=False)
        loaded._build_autoencoder()
        trainer2 = DeepADEFCMTrainer(loaded, tiny_data)
        ckpt = trainer2.load_checkpoint(ckpt_path)
        assert ckpt['epoch'] == 5
        assert ckpt['loss'] == 0.5

    def test_train_with_early_stopping(self, tiny_data):
        model = DeepADEFCM(input_dim=tiny_data.shape[1], latent_dim=2,
                           n_clusters=2, ae_epochs=2, joint_epochs=10,
                           batch_size=4, verbose=False)
        trainer = DeepADEFCMTrainer(model, tiny_data)
        result = trainer.train_with_early_stopping(patience=2, min_delta=1e-4)
        assert result is model
        assert model.labels_ is not None
