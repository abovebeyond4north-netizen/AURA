"""Bounded two-generation research study; never deploys a candidate."""
from .runner import evaluate_candidate
from .modification_registry import apply_modification
from .proposer import ModificationProposer
from .search_strategy import ImprovementSearch
from .candidate_archive import CandidateArchive


def run_study(config, path, *, steps=30, replicates=5):
    archive = CandidateArchive(path)
    search = ImprovementSearch(config.seed)
    proposer = ModificationProposer(search)
    parent = 'v0'
    rounds = []
    for _ in range(2):
        name, mod = proposer.propose(parent, weakness=1)
        index = len(archive.versions)
        while f'study-{index}' in archive.versions:
            index += 1
        version = f'study-{index}'
        try:
            report = evaluate_candidate(config, mod, steps=steps, replicates=replicates, version=version)
        except (RuntimeError, OSError) as exc:
            rounds.append({'candidate': version, 'parent': parent, 'status': 'BLOCKED', 'reason': str(exc)})
            break
        archive.record(version, parent, mod, report)
        rounds.append({'candidate': version, 'parent': parent, 'status': 'ACCEPTED_EXPERIMENT' if report.accepted else 'REJECTED', 'reason': report.reason})
        if not report.accepted:
            break
        search.update(name, report.metrics['mean_prediction_gain'], report.total_cost['trained_model_steps'])
        config = apply_modification(config, mod)
        parent = version
    return {'rounds': rounds, 'two_accepted_experiments': len(rounds) == 2 and all(r['status'] == 'ACCEPTED_EXPERIMENT' for r in rounds), 'deployed': False}
