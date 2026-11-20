from experiments.noisy_tv import run

def test_knowledge_vs_irreducible_tv_noise_ablation():
    # A true exploratory action policy is used, not a manually chosen outcome.
    samples=[(run('learning_progress',s),run('prediction_error',s),run('random',s)) for s in range(3)]
    curiosity=sum(sample[0]['noise_action_fraction_last_100'] for sample in samples)/len(samples)
    raw_error=sum(sample[1]['noise_action_fraction_last_100'] for sample in samples)/len(samples)
    assert curiosity < raw_error and curiosity < .5, (curiosity,raw_error)
    assert all(sample[0]['heldout_learnable_mse'] < .15 for sample in samples)
