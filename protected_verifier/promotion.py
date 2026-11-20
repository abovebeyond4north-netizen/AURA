"""Immutable-in-process policy constants; never accepted from candidate JSON."""
from aura.types import EvaluationReport
from .metrics import paired_bootstrap
import math

MIN_SEEDS=5
MIN_GAIN=0.0001
MAX_REGRESSION=0.015

def decide(candidate_version,baseline_version,paired_prediction_gains,baseline_losses,candidate_losses,cost):
    # Reject malformed or non-finite evidence before bootstrap or promotion.
    # Do not silently zip unequal arrays or allow NaN/Inf in serialized reports.
    vectors=(paired_prediction_gains,baseline_losses,candidate_losses)
    if not all(isinstance(v,(list,tuple)) for v in vectors):
        raise ValueError('Verifier evidence must be sequences')
    if not paired_prediction_gains or any(len(v)!=len(paired_prediction_gains) for v in vectors):
        raise ValueError('Missing or unpaired verifier evidence')
    if any(isinstance(x,bool) or not isinstance(x,(int,float)) or not math.isfinite(x)
           for v in vectors for x in v):
        raise ValueError('Non-finite or non-numeric verifier evidence')
    if any(x<0 for v in (baseline_losses,candidate_losses) for x in v):
        raise ValueError('Prediction losses must be nonnegative')
    if any(abs(g-(b-c))>1e-9 for g,b,c in zip(*vectors)):
        raise ValueError('Paired gains disagree with measured losses')
    lower,upper=paired_bootstrap(paired_prediction_gains)
    regress=max([max(0,c-b) for b,c in zip(baseline_losses,candidate_losses)],default=float('inf'))
    accepted=(len(paired_prediction_gains)>=MIN_SEEDS and lower>MIN_GAIN and regress<=MAX_REGRESSION)
    return EvaluationReport(candidate_version,baseline_version,
        {'mean_prediction_gain':sum(paired_prediction_gains)/len(paired_prediction_gains),
         'baseline_mean_loss':sum(baseline_losses)/len(baseline_losses),
         'candidate_mean_loss':sum(candidate_losses)/len(candidate_losses)},
        {'prediction_gain_ci95':(lower,upper)},
        {'max_seed_regression':regress},cost,accepted,
        'PROMOTED' if accepted else ('INCONCLUSIVE_INSUFFICIENT_SEEDS' if len(paired_prediction_gains)<MIN_SEEDS else 'NO_VALIDATED_GAIN_OR_REGRESSION'))
