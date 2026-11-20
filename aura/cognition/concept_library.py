"""Evidence-gated revisions to reusable predictive prototypes."""
from dataclasses import dataclass
import numpy as np

from aura.types import Source


@dataclass
class Prototypes:
    centers: np.ndarray
    effects: np.ndarray

    def assign(self, x):
        return ((x[:, None] - self.centers)**2).sum(axis=-1).argmin(axis=1)

    def predict(self, x):
        return self.effects[self.assign(x)]


class ConceptLibrary:
    def __init__(self, ledger, context, *, transfer_weight=1., complexity_cost=.02):
        self.ledger = ledger
        self.context = context
        self.transfer_weight = transfer_weight
        self.complexity_cost = complexity_cost
        self.model = None
        for record in ledger.records('concept_revision'):
            payload = record['payload']
            if payload.get('context') == context:
                state = payload['prototypes']
                self.model = (Prototypes(np.asarray(state['centers']), np.asarray(state['effects']))
                              if state is not None else None)

    def _read(self, ids):
        records = {r['record_id']: r for r in self.ledger.records('concept_sample')}
        selected = [records[i] for i in ids]
        if any(r['source'] != Source.OBSERVED.value for r in selected):
            raise ValueError('Concept fitting requires observations')
        return tuple(np.asarray([r['payload'][key] for r in selected], dtype=float)
                     for key in ('features', 'outcome'))

    @staticmethod
    def _fit(x, y, centers):
        for _ in range(15):
            groups = ((x[:,None] - centers)**2).sum(axis=-1).argmin(axis=1)
            if len(set(groups)) != len(centers):
                return None
            centers = np.stack([x[groups == k].mean(axis=0) for k in range(len(centers))])
        groups = ((x[:,None] - centers)**2).sum(axis=-1).argmin(axis=1)
        if len(set(groups)) != len(centers):
            return None
        return Prototypes(centers, np.stack([y[groups == k].mean(axis=0)
                                            for k in range(len(centers))]))

    def revise(self, revision, train, validation, transfer):
        ids = [*train, *validation, *transfer]
        if min(len(train), len(validation), len(transfer)) == 0 or len(ids) != len(set(ids)):
            raise ValueError('Need three disjoint nonempty observation sets')
        x, y = self._read(train)
        folds = [self._read(split) for split in (validation, transfer)]
        default = Prototypes(x.mean(axis=0, keepdims=True), y.mean(axis=0, keepdims=True))
        active = self.model or default

        def losses(model):
            return [float(np.mean((model.predict(features) - targets)**2))
                    for features, targets in folds]

        centers = active.centers
        n = len(centers)
        proposals = [('retire', default)] if self.model is not None else []
        if n > 2:
            for i in range(n):
                for j in range(i+1,n):
                    merged = np.vstack((np.delete(centers, [i,j], axis=0), (centers[i]+centers[j])/2))
                    proposals.append(('merge', self._fit(x,y,merged)))
        groups = active.assign(x)
        if n < 8:
            for i in range(n):
                members = x[groups == i]
                if len(members) < 2:
                    continue
                a = members[((members-centers[i])**2).sum(axis=1).argmax()]
                b = members[((members-a)**2).sum(axis=1).argmax()]
                split = np.vstack((np.delete(centers,i,axis=0),a,b))
                proposals.append(('create' if self.model is None else 'split',self._fit(x,y,split)))

        old = losses(active)
        comparisons = []
        best = None
        for action, model in proposals:
            if model is None:
                continue
            new = losses(model)
            gain_p, gain_t = (a-b for a,b in zip(old,new))
            added = len(model.centers)-n
            score = gain_p + self.transfer_weight*gain_t - self.complexity_cost*added
            row = dict(operation=action, prediction_gain=gain_p, transfer_gain=gain_t,
                       added_prototypes=added, score=score)
            comparisons.append(row)
            if score > 1e-9 and min(gain_p,gain_t) >= -1e-9 and (best is None or score > best[0]['score']):
                best = row, model
        accepted = best[0]['operation'] if best else None
        next_model = (None if accepted == 'retire' else best[1]) if best else self.model
        payload = dict(context=self.context, accepted=accepted, baseline_losses=old, candidates=comparisons,
                       train_ids=list(train), validation_ids=list(validation), transfer_ids=list(transfer),
                       prototypes=None if next_model is None else dict(
                           centers=next_model.centers.tolist(), effects=next_model.effects.tolist()))
        self.ledger.append(f'concept:{self.context}:{revision}', 'concept_revision', Source.INFERRED, payload)
        self.model = next_model
        return payload
