from aura.agent import AuraOmega
from aura.config import Config
from aura.environments.task_generator import task_seed


def keys(batch):
    return {(x.tobytes(), int(a), y.tobytes()) for x, a, y in zip(*batch)}


def test_online_validation_is_fixed_and_separate_from_training(tmp_path, monkeypatch):
    config = Config(seed=73, steps=32, batch_size=4, warmup=4,
                    latent_dim=8, hidden_dim=16, ensemble_size=2,
                    max_episode_steps=12, log_dir=str(tmp_path))
    agent = AuraOmega(config)
    train_batches, validation_batches = [], []
    train = agent.world_model.train_batch
    predict = agent.world_model.prediction_loss
    def track_train(batch):
        train_batches.append(keys(batch))
        return train(batch)
    def track_prediction(batch):
        validation_batches.append(keys(batch))
        return predict(batch)
    monkeypatch.setattr(agent.world_model, 'train_batch', track_train)
    monkeypatch.setattr(agent.world_model, 'prediction_loss', track_prediction)
    try:
        report = agent.train(task='noise', condition='fixed')
        assert len(validation_batches) >= 2
        assert all(batch == validation_batches[0] for batch in validation_batches)
        assert not set.union(*train_batches).intersection(validation_batches[0])
        dev = report['online_development_validation']
        assert dev['seed'] == task_seed('dev', config.seed)
        assert dev['transitions'] == 8 and dev['task'] == 'noise'
        assert report['resource_metrics']['development_interactions'] == 8
        assert report['resource_metrics']['total_environment_interactions'] == 40
        assert len(agent.ledger.records('experience')) == 32
        assert len(agent.ledger.records('development_validation')) == 1
        assert report['ledger_valid']
    finally:
        agent.close()


def test_random_baseline_does_not_collect_development_data(tmp_path):
    agent = AuraOmega(Config(steps=12, log_dir=str(tmp_path)))
    try:
        report = agent.train(condition='random', task='key')
        assert report['online_development_validation'] is None
        assert report['resource_metrics']['development_interactions'] == 0
        assert report['resource_metrics']['total_environment_interactions'] == 12
    finally:
        agent.close()


def test_no_development_budget_means_no_validation_not_replay_fallback(tmp_path):
    from aura.meta_learning.resource_allocator import Budget
    agent = AuraOmega(Config(steps=20, warmup=4, batch_size=4, latent_dim=8,
                             hidden_dim=16, ensemble_size=2, log_dir=str(tmp_path)))
    agent.budget = Budget(interactions=20, updates=60, development_interactions=0)
    try:
        report = agent.train(condition='fixed', task='key')
        assert report['last_training_loss'] is not None
        assert report['online_development_validation'] is None
        assert not agent.curiosity.progress.errors
        assert report['resource_metrics']['total_environment_interactions'] == 20
        assert not agent.ledger.records('development_validation')
    finally:
        agent.close()
