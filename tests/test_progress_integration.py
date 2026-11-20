import json
from pathlib import Path
from aura.agent import AuraOmega
from aura.config import Config


def test_validated_curiosity_runs_in_real_training_and_preserves_ledger(tmp_path):
    conf=Config(seed=41,steps=32,batch_size=8,warmup=8,
                max_episode_steps=16,temporal_updates_per_episode=0,
                curiosity_policy='validated_progress',log_dir=str(tmp_path))
    agent=AuraOmega(conf)
    try:
        result=agent.train(steps=32,condition='curiosity',task='noise',log_every=16)
        assert result['steps']==32
        assert result['ledger_valid']
        assert result['exploration_policy']=='validated_progress'
        assert sum(agent.progress_explorer.action_count)==32
        assert sum(agent.progress_explorer.training_count)+sum(
            len(x) for x in agent.progress_explorer.validation)==32
        summaries=[json.loads(line) for line in (Path(agent.output_dir)/'metrics.jsonl').read_text().splitlines()]
        last=summaries[-1]
        assert last['exploration_metrics']['method']=='frozen online validation; progress is an observed local proxy'
        assert (Path(agent.output_dir)/'checkpoint.pt').exists()
    finally:
        agent.close()
