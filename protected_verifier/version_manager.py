"""Protected, versioned config deployment with rollback (not arbitrary code)."""
from pathlib import Path
import json,os,re
from aura.config import Config
from aura.types import Modification, EvaluationReport
from aura.self_improvement.modification_registry import apply_modification
from .promotion import decide

_VERSION_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,127}\Z", re.ASCII)


def validate_version(version):
    """Disallow path components and control characters in all persisted version IDs."""
    if not isinstance(version, str) or _VERSION_PATTERN.fullmatch(version) is None:
        raise ValueError('Unsafe version name')
    return version


class VersionManager:
    def __init__(self,path,initial:Config):
        self.path=Path(path);self.path.mkdir(parents=True,exist_ok=True)
        self.active_file=self.path/'ACTIVE.json'
        if not self.active_file.exists():
            self._write('v0',initial)
            self._activate('v0')

    def _write(self,version,config):
        path=self.path/f'{validate_version(version)}.json'
        if path.exists(): raise ValueError('Version already exists')
        path.write_text(json.dumps(config.model_dump(),sort_keys=True))

    def _activate(self,version):
        validate_version(version)
        if not (self.path/f'{version}.json').is_file(): raise ValueError('Unknown version')
        temp=self.path/'ACTIVE.new'
        temp.write_text(json.dumps({'active_version':version}))
        os.replace(temp,self.active_file)

    @property
    def active_version(self): return validate_version(json.loads(self.active_file.read_text())['active_version'])

    def get_active(self): return Config.model_validate_json((self.path/f'{self.active_version}.json').read_text())

    def promote(self,mod:Modification,report:EvaluationReport,baseline_losses,candidate_losses):
        # These protected evaluator measurements MUST be passed by the verifier,
        # not by learner-controlled JSON. Recompute policy rather than trusting
        # a caller-supplied `accepted=True` boolean.
        if len(baseline_losses)!=len(candidate_losses) or not baseline_losses:
            raise PermissionError('Missing paired verifier measurements')
        computed=decide(report.candidate_version,report.baseline_version,
            [b-c for b,c in zip(baseline_losses,candidate_losses)],
            baseline_losses,candidate_losses,report.total_cost)
        if (not computed.accepted or not report.accepted or
            abs(computed.metrics['mean_prediction_gain']-report.metrics.get('mean_prediction_gain',-1000))>1e-10 or
            report.baseline_version!=self.active_version or mod.parent_version!=self.active_version):
            raise PermissionError('Independent promotion gate denied or parent mismatch')
        version=report.candidate_version
        validate_version(version)
        candidate=apply_modification(self.get_active(),mod)
        self._write(version,candidate)
        self._activate(version)

    def rollback(self,version): self._activate(version)
