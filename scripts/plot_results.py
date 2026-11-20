#!/usr/bin/env python3
"""Create reproducible observed-results chart; no smoothing or invented data."""
from pathlib import Path
import argparse,json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def main():
    p=argparse.ArgumentParser();p.add_argument('--report',default='sample_outputs/baseline_report.json')
    p.add_argument('--output',default='sample_outputs/baseline_prediction_mse.png')
    args=p.parse_args()
    rows=json.loads(Path(args.report).read_text())
    groups=['random','fixed','curiosity','adaptive','omega']
    values=[[float(x['heldout_image_mse']) for x in rows if x['condition']==group] for group in groups]
    if any(not v for v in values): raise ValueError('Missing baseline group')
    means=[np.mean(v) for v in values]
    sems=[np.std(v,ddof=1)/np.sqrt(len(v)) if len(v)>1 else 0 for v in values]
    fig,ax=plt.subplots(figsize=(9,5))
    ax.bar(groups,means,yerr=sems,capsize=4)
    ax.set_ylabel('Held-out next-image mean squared error')
    ax.set_xlabel('Experimental condition')
    ax.set_title('AURA-Ω short-run baseline comparison (descriptive; n=5 seeds)')
    fig.tight_layout();out=Path(args.output);out.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(out,dpi=160);plt.close(fig)
    print(out)
if __name__=='__main__':main()
