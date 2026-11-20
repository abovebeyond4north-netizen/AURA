from aura.config import Config
from aura.self_improvement import study
from protected_verifier.promotion import decide


def test_blocked_study_stops_without_lineage(monkeypatch, tmp_path):
    def blocked(*args, **kwargs):
        raise RuntimeError('verifier unavailable')
    monkeypatch.setattr(study, 'evaluate_candidate', blocked)
    path = tmp_path / 'lineage.jsonl'
    result = study.run_study(Config(), path)
    assert result['rounds'][0]['status'] == 'BLOCKED'
    assert not result['two_accepted_experiments'] and not result['deployed']
    assert not path.exists()


def test_rejected_study_stops(monkeypatch, tmp_path):
    def rejected(config, mod, **kw):
        return decide(kw['version'], mod.parent_version, [0]*5, [.1]*5, [.1]*5, {'trained_model_steps': 300})
    monkeypatch.setattr(study, 'evaluate_candidate', rejected)
    result = study.run_study(Config(), tmp_path / 'lineage.jsonl')
    assert len(result['rounds']) == 1 and result['rounds'][0]['status'] == 'REJECTED'
    assert not result['deployed']


def test_all_task_smoke_records_independent_runs(tmp_path):
    from pathlib import Path
    from aura.environments.procedural_world import TASKS
    from scripts.minimum_architecture import run_learning_matrix
    report = run_learning_matrix(Config(steps=12, warmup=4, batch_size=4,
                                        latent_dim=8, hidden_dim=16, ensemble_size=2,
                                        max_episode_steps=6), tmp_path)
    assert set(report['tasks']) == set(TASKS)
    assert report['all_ledgers_valid']
    assert report['training_interactions'] == 12 * len(TASKS)
    assert report['development_interactions'] == 8 * len(TASKS)
    assert report['completed_episodes'] >= 2 * len(TASKS)
    for task, run in report['tasks'].items():
        assert run['online_development_validation']['task'] == task
        assert Path(run['checkpoint']).exists()
