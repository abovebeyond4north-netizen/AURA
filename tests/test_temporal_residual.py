"""Acceptance checks for action-conditioned image/latent multi-horizon effects."""
import numpy as np
import pytest
import torch
from aura.perception.encoder import ObservationEncoder
from aura.world_model.temporal_heads import TemporalHeads


def synthetic_episode(seed, steps=16):
    """Unlabeled observable action effects, not simulator-provided concepts."""
    rng = np.random.default_rng(seed)
    x = np.zeros((3, 7, 7), dtype=np.uint8)
    images, actions, endings = [x.copy()], [], []
    for t in range(steps):
        action = int(rng.integers(5))
        actions.append(action)
        x = x.copy()
        x[action % 3] = np.clip(x[action % 3].astype(np.int32) + 20, 0, 255)
        images.append(x.copy())
        endings.append(t == steps - 1)
    return np.stack(images), actions, endings


def test_residual_predictions_start_as_exact_persistence():
    torch.manual_seed(11)
    model = TemporalHeads(ObservationEncoder(8), 8, (1, 2, 4),
                          hidden_dim=24, mode='residual', freeze_encoder=True)
    o, actions, _ = synthetic_episode(21)
    for h in model.horizons:
        prediction = model.predict(o[0], actions[:h])
        assert prediction.shape == (1, 3, 7, 7)
        assert torch.allclose(prediction[0], torch.as_tensor(o[0], dtype=torch.float32) / 255, atol=1e-7)


def test_residual_image_and_latent_effects_improve_out_of_sample():
    torch.manual_seed(13)
    model = TemporalHeads(ObservationEncoder(8), 8, (1, 2, 4), hidden_dim=24,
                          lr=.003, mode='residual', freeze_encoder=True)
    training = [synthetic_episode(i) for i in (1, 2, 3)]
    heldout = [synthetic_episode(i) for i in (101, 102, 103)]
    before_image, before_latent = model.evaluate(heldout), model.evaluate_latent(heldout)
    observed = model.train_trajectories(training, updates=300)
    after_image, after_latent = model.evaluate(heldout), model.evaluate_latent(heldout)
    assert observed['available_horizons'] == [1, 2, 4]
    assert model.latent_is_trained
    assert all(after_image[h] < before_image[h] * .9 for h in (1, 2, 4))
    assert all(after_latent[h] < before_latent[h] * .9 for h in (1, 2, 4))


def test_temporal_checkpoint_v3_roundtrip_and_v1_migration(tmp_path):
    torch.manual_seed(10)
    encoder = ObservationEncoder(8)
    model = TemporalHeads(encoder, 8, (1, 2, 4), hidden_dim=24,
                          mode='residual', freeze_encoder=True)
    model.train_trajectories([synthetic_episode(4)], updates=8)
    file = tmp_path / 'temporal-v3.pt'
    model.save(file, update_steps=8)
    restored = TemporalHeads(encoder, 8, (1, 2, 4), hidden_dim=24,
                             mode='residual', freeze_encoder=True)
    assert restored.restore(file) == 8
    assert restored.latent_is_trained
    o, actions, _ = synthetic_episode(4)
    assert torch.allclose(model.predict(o[0], actions[:4]), restored.predict(o[0], actions[:4]))
    assert torch.allclose(model.predict_latent(o[0], actions[:4]), restored.predict_latent(o[0], actions[:4]))
    with pytest.raises(ValueError, match='Incompatible'):
        TemporalHeads(encoder, 8, mode='absolute').restore(file)
    legacy = tmp_path / 'temporal-v1.pt'
    legacy_head = TemporalHeads(encoder, 8, (1, 2, 4), hidden_dim=24, mode='absolute')
    torch.save({'state_dict':legacy_head.heads.state_dict(), 'update_steps':10,
                'schema_version':1, 'horizons':(1, 2, 4)}, legacy)
    restored_legacy = TemporalHeads(encoder, 8, (1, 2, 4), hidden_dim=24, mode='absolute')
    assert restored_legacy.restore(legacy) == 10
    assert not restored_legacy.latent_is_trained
    with pytest.raises(ValueError, match='Incompatible'):
        restored.restore(legacy)


def test_invalid_sequence_rejected_in_image_and_latent_heads():
    model = TemporalHeads(ObservationEncoder(8),8,mode='residual')
    o,_,_=synthetic_episode(9)
    with pytest.raises(ValueError):
        model.predict(o[0], [9])
    with pytest.raises(ValueError):
        model.predict_latent(o[0], [-1])
    with pytest.raises(ValueError):
        model.predict(o[0], [0,1,2])


def test_optimizer_state_restored_for_exact_continuation(tmp_path):
    torch.manual_seed(77)
    encoder=ObservationEncoder(8)
    trained=TemporalHeads(encoder,8,(1,2,4),hidden_dim=24,
                          lr=.003,mode='residual',freeze_encoder=True)
    tr=[synthetic_episode(4),synthetic_episode(5)]
    trained.train_trajectories(tr,updates=25)
    file=tmp_path/'resume.pt'
    trained.save(file,update_steps=25)
    continued=TemporalHeads(encoder,8,(1,2,4),hidden_dim=24,
                            lr=.003,mode='residual',freeze_encoder=True)
    assert continued.restore(file)==25
    trained.train_trajectories(tr,updates=12)
    continued.train_trajectories(tr,updates=12)
    for a,b in zip(trained.heads.parameters(),continued.heads.parameters()):
        assert torch.allclose(a,b,atol=1e-7)
    for a,b in zip(trained.latent_heads.parameters(),continued.latent_heads.parameters()):
        assert torch.allclose(a,b,atol=1e-7)
