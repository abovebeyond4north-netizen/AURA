"""Reject anything not in registry; verifier operates in a fresh OS process."""
import subprocess,sys,json,math
from dataclasses import asdict
from aura.types import Modification,EvaluationReport
from aura.config import Config
from .modification_registry import apply_modification
from protected_verifier.promotion import MIN_SEEDS, MIN_GAIN, MAX_REGRESSION

def evaluate_candidate(config:Config,mod:Modification,steps=60,replicates=5,version='v1'):
    apply_modification(config,mod)
    request={'config':config.model_dump(),'modification':asdict(mod),
             'steps':steps,'replicates':replicates,'parent':mod.parent_version,'candidate':version}
    process=subprocess.run([sys.executable,'-m','protected_verifier.worker'],
        input=json.dumps(request),text=True,capture_output=True,timeout=900)
    if not process.stdout.strip(): raise RuntimeError('Verifier failed: '+process.stderr[:300])
    result=_parse_response(process.stdout)
    if result['ok'] is False:
        raise RuntimeError(result['error'])
    if process.returncode:
        raise RuntimeError('Invalid verifier response: success with nonzero exit')
    report = result.get('report')
    if (result['ok'] is not True or not isinstance(report, dict)
            or report.get('candidate_version') != version
            or report.get('baseline_version') != mod.parent_version
            or type(report.get('accepted')) is not bool
            or report.get('reason') not in ('PROMOTED', 'INCONCLUSIVE_INSUFFICIENT_SEEDS',
                                            'NO_VALIDATED_GAIN_OR_REGRESSION')
            or report['accepted'] != (report['reason'] == 'PROMOTED')):
        raise RuntimeError('Invalid verifier response: inconsistent identity or acceptance')
    _validate_evidence(report, steps, replicates)
    return EvaluationReport(**report)


def _parse_response(stdout):
    """Accept exactly one unambiguous JSON envelope, not the last stdout line."""
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError('duplicate key')
            result[key] = value
        return result

    def reject_constant(value):
        raise ValueError('non-finite constant')

    try:
        result = json.loads(stdout, object_pairs_hook=unique, parse_constant=reject_constant)
        if not isinstance(result, dict) or type(result.get('ok')) is not bool:
            raise ValueError('invalid envelope')
        if result['ok']:
            if set(result) != {'ok', 'report'}:
                raise ValueError('invalid success envelope')
        elif set(result) != {'ok', 'error'} or not isinstance(result['error'], str):
            raise ValueError('invalid error envelope')
    except (ValueError, RecursionError) as exc:
        raise RuntimeError('Invalid verifier response: malformed JSON envelope') from exc
    return result


def _validate_evidence(report, steps, replicates):
    """Check summary consistency; this cannot authenticate underlying measurements."""
    def require(condition):
        if not condition:
            raise RuntimeError('Invalid verifier response: inconsistent evidence or budget')

    def number(value):
        try:
            return type(value) in (int, float) and math.isfinite(value)
        except OverflowError:
            return False

    def fields(value, names):
        require(isinstance(value, dict) and set(value) == set(names))
        return value

    cost = fields(report.get('total_cost'), ('trained_model_steps', 'replicates'))
    require(type(cost['trained_model_steps']) is int and type(cost['replicates']) is int
            and cost['trained_model_steps'] == 2 * steps * replicates
            and cost['replicates'] == replicates)
    metrics = fields(report.get('metrics'), ('mean_prediction_gain', 'baseline_mean_loss', 'candidate_mean_loss'))
    require(all(number(v) for v in metrics.values()))
    gain, baseline, candidate = (metrics[k] for k in ('mean_prediction_gain', 'baseline_mean_loss', 'candidate_mean_loss'))
    require(baseline >= 0 and candidate >= 0 and math.isclose(gain, baseline-candidate, abs_tol=1e-9))
    interval = fields(report.get('confidence_intervals'), ('prediction_gain_ci95',))['prediction_gain_ci95']
    require(isinstance(interval, (list, tuple)) and len(interval) == 2 and all(number(v) for v in interval))
    lower, upper = interval
    require(lower <= upper)
    # Nonnegative paired losses bound every individual gain and bootstrap mean.
    if replicates < 3:
        require((lower, upper) == (-1e100, 1e100))  # protected bootstrap sentinel
    else:
        min_gain, max_gain = -candidate * replicates, baseline * replicates
        require(math.isfinite(min_gain) and math.isfinite(max_gain))
        require((lower >= min_gain or math.isclose(lower, min_gain, abs_tol=1e-9))
                and (upper <= max_gain or math.isclose(upper, max_gain, abs_tol=1e-9)))
    regression = fields(report.get('regressions'), ('max_seed_regression',))['max_seed_regression']
    require(number(regression) and regression >= 0)
    accepted = replicates >= MIN_SEEDS and lower > MIN_GAIN and regression <= MAX_REGRESSION
    reason = ('PROMOTED' if accepted else 'INCONCLUSIVE_INSUFFICIENT_SEEDS'
              if replicates < MIN_SEEDS else 'NO_VALIDATED_GAIN_OR_REGRESSION')
    require(report['accepted'] == accepted and report['reason'] == reason)

