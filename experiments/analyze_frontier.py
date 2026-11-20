"""Strict paired analysis of observed gridworld exploration results.

No hidden scores or built-in success thresholds. Reject missing, duplicate, or
unequal-budget comparisons rather than silently pairing unrelated runs.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
from scipy.stats import t


def _paired_ci(differences):
    sample=np.asarray(differences,dtype=float)
    if sample.size<2 or not np.all(np.isfinite(sample)):
        raise ValueError('Require at least two finite paired measurements')
    midpoint=float(sample.mean())
    half=float(t.ppf(.975,df=len(sample)-1)*sample.std(ddof=1)/np.sqrt(len(sample)))
    return {'mean_difference':midpoint,'confidence_interval_95':[midpoint-half,midpoint+half],
            'n_matched_seeds':len(sample),'estimator':'paired Student t, per-seed differences'}


def analyze(rows, baseline='random', candidate='visual_frontier'):
    if baseline==candidate:
        raise ValueError('Distinct baseline and candidate required')
    data={}
    for row in rows:
        key=(row['task'],row['seed'],row['policy'])
        if key in data:raise ValueError(f'Duplicate trial: {key}')
        if row['interactions']<1:raise ValueError('Invalid interaction budget')
        data[key]=row
    tasks=sorted({task for task,seed,policy in data if policy==candidate})
    if not tasks:raise ValueError('No candidate rows')
    summary={}
    for task in tasks:
        candidates={seed:row for (name,seed,policy),row in data.items()
                    if name==task and policy==candidate}
        controls={seed:row for (name,seed,policy),row in data.items()
                  if name==task and policy==baseline}
        if candidates.keys()!=controls.keys() or len(candidates)<2:
            raise ValueError(f'Missing matched pair for task {task}')
        seeds=sorted(candidates)
        for seed in seeds:
            if candidates[seed]['interactions']!=controls[seed]['interactions']:
                raise ValueError('Unequal experiment budget')
        def metrics(metric):
            return _paired_ci([candidates[seed][metric]-controls[seed][metric] for seed in seeds])
        def heldout():
            return _paired_ci([
                candidates[seed]['common_heldout_evaluation']['prediction_feature_mse']-
                controls[seed]['common_heldout_evaluation']['prediction_feature_mse']
                for seed in seeds])
        summary[task]={
            'seeds':seeds,
            'interactions_per_policy_per_seed':candidates[seeds[0]]['interactions'],
            'paired_episode_position_coverage':metrics('unique_episode_positions'),
            'paired_physical_position_coverage':metrics('unique_physical_positions'),
            'paired_heldout_feature_mse_difference':heldout(),
            'candidate_successful_episodes':sum(candidates[s]['successful_episodes'] for s in seeds),
            'baseline_successful_episodes':sum(controls[s]['successful_episodes'] for s in seeds),
            'candidate_mean_physical_coverage':float(np.mean([candidates[s]['unique_physical_positions'] for s in seeds])),
            'baseline_mean_physical_coverage':float(np.mean([controls[s]['unique_physical_positions'] for s in seeds])),
        }
        if task=='noisy_tv':
            summary[task]['candidate_mean_tv_fraction_last_half']=float(np.mean([candidates[s]['tv_fraction_last_half'] for s in seeds]))
            summary[task]['baseline_mean_tv_fraction_last_half']=float(np.mean([controls[s]['tv_fraction_last_half'] for s in seeds]))
    return {'baseline':baseline,'candidate':candidate,'tasks':summary,
            'limitations':'Fresh local seeds, not externally sealed; intervals are unadjusted exploratory CIs across 3 tasks, do not establish task mastery.'}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('records',nargs='+')
    parser.add_argument('--output',default='runs/visual_frontier_final_report.json')
    args=parser.parse_args()
    all_rows=[]
    for path in args.records:
        result=json.loads(Path(path).read_text())
        all_rows.extend(result if isinstance(result,list) else result['rows'])
    report=analyze(all_rows)
    dest=Path(args.output);dest.parent.mkdir(parents=True,exist_ok=True)
    dest.write_text(json.dumps(report,indent=2,sort_keys=True,allow_nan=False))
    print(json.dumps(report['tasks'],indent=2))

if __name__=='__main__':main()
