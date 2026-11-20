"""Controlled two-arm noisy-TV benchmark for intrinsic-selection heuristics.

Arm 0: learnable deterministic binary response equal to visible context.
Arm 1: uncontrollable independent Bernoulli(0.5) TV output.
No external labels are used to train action-specific Beta-Bernoulli predictors.
A fixed held-out stream estimates prediction MSE for each arm.
This is a diagnostic benchmark, not proof of robust exploration in the gridworld.
"""
import argparse,json
from pathlib import Path
import numpy as np
from aura.exploration.curiosity import Curiosity

class Predictor:
    def __init__(self): self.counts=np.ones((2,2,2),dtype=float)
    def predict(self,action,context):
        v=self.counts[action,context]; return float(v[1]/v.sum())
    def update(self,action,context,outcome): self.counts[action,context,outcome]+=1
    def epistemic_proxy(self,action):
        a=self.counts[action,:,1];b=self.counts[action,:,0]
        return float(np.mean(a*b/((a+b)**2*(a+b+1))))
    def val_loss(self,action,contexts,targets):
        p=np.array([self.predict(action,int(x)) for x in contexts])
        return float(np.mean((p-targets)**2))

def run(policy,seed=0,steps=350):
    rng=np.random.default_rng(seed)
    p=Predictor();c=Curiosity(None,noise_penalty=10,diversity=.07,cost=0)
    ctx_test=rng.integers(2,size=200)
    y_test=np.vstack([ctx_test.copy(),rng.integers(2,size=200)])
    selects=[];prev=np.ones(2)*.25;progress=np.zeros(2)
    for step in range(steps):
        losses=np.array([p.val_loss(a,ctx_test,y_test[a]) for a in range(2)])
        if step<20 or policy=='random' or rng.random()<.05: action=int(rng.integers(2))
        elif policy=='prediction_error':action=int(np.argmax(losses))
        elif policy=='learning_progress':
            scores=[c.score(p.epistemic_proxy(a),a,noise_estimate=losses[a],
                            transferable_benefit=progress[a]-.12*losses[a]) for a in range(2)]
            action=int(np.argmax(scores))
        else:raise ValueError('Unsupported policy')
        context=int(rng.integers(2))
        outcome=context if action==0 else int(rng.integers(2))
        p.update(action,context,outcome)
        new_loss=p.val_loss(action,ctx_test,y_test[action])
        progress[action]=.8*progress[action]+.2*max(0,prev[action]-new_loss)
        prev[action]=new_loss
        c.counts[action]+=1
        selects.append(action)
    final=np.array([p.val_loss(a,ctx_test,y_test[a]) for a in range(2)])
    return {'policy':policy,'seed':seed,'steps':steps,
            'noise_action_fraction_last_100':float(np.mean(np.asarray(selects)[-100:]==1)),
            'heldout_learnable_mse':float(final[0]),'heldout_noise_mse':float(final[1])}

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--seeds',type=int,default=5)
    parser.add_argument('--steps',type=int,default=350);parser.add_argument('--output',default='runs/noisy_tv_report.json')
    args=parser.parse_args()
    rows=[run(policy,seed,args.steps) for policy in ('random','prediction_error','learning_progress') for seed in range(args.seeds)]
    out=Path(args.output);out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(rows,indent=2))
    for policy in ('random','prediction_error','learning_progress'):
        relevant=[r for r in rows if r['policy']==policy]
        print(policy, 'noise_selection_last100_mean',round(float(np.mean([r['noise_action_fraction_last_100'] for r in relevant])),3))
    print('Saved',out)
if __name__=='__main__':main()
