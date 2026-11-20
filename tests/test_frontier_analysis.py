import pytest
from experiments.analyze_frontier import analyze


def trial(seed,policy,coverage,interactions=50):
    return {'task':'key','seed':seed,'policy':policy,'interactions':interactions,
            'unique_episode_positions':coverage+5,'unique_physical_positions':coverage,
            'successful_episodes':0,
            'common_heldout_evaluation':{'prediction_feature_mse':.01}}


def test_paired_intervals_from_observed_seedwise_differences():
    rows=[]
    for seed in range(3):
        rows.extend([trial(seed,'random',10+seed),trial(seed,'visual_frontier',14+seed)])
    result=analyze(rows)['tasks']['key']
    assert result['paired_physical_position_coverage']['mean_difference']==4
    assert result['paired_physical_position_coverage']['confidence_interval_95']==[4,4]
    assert result['seeds']==[0,1,2]


def test_analysis_fails_closed_on_missing_duplicate_or_unmatched_budgets():
    x=trial(2,'random',2);y=trial(2,'visual_frontier',5)
    with pytest.raises(ValueError):analyze([x,y,y])
    with pytest.raises(ValueError):analyze([x])
    with pytest.raises(ValueError):analyze([x,y])  # only one pair
    a=trial(1,'random',2);b=trial(1,'visual_frontier',5,interactions=51)
    with pytest.raises(ValueError):analyze([x,y,a,b])
