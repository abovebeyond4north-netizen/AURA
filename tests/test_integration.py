from pathlib import Path
from aura.agent import AuraOmega
from aura.config import Config
from aura.evaluation.runner import evaluate_world_model

def test_short_end_to_end_run(tmp_path):
    c=Config(steps=20,batch_size=6,warmup=6,latent_dim=8,hidden_dim=16,ensemble_size=2,log_dir=str(tmp_path))
    a=AuraOmega(c)
    result=a.train(log_every=10)
    assert result['steps']==20 and result['ledger_valid']
    assert Path(result['checkpoint']).exists() and len(a.ledger.records('experience'))==20
    assert evaluate_world_model(a.world_model)['heldout_image_mse']>=0
    a.close()
