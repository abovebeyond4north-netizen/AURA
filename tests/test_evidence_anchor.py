import hashlib
from pathlib import Path
import sqlite3
from tempfile import TemporaryDirectory
import unittest

import numpy as np

from aura.memory.anchor import canonical, checkpoint, verify
from aura.memory.episodic import EpisodicMemory
from aura.memory.evidence_ledger import EvidenceLedger
from aura.memory.procedural import ProceduralMemory
from aura.memory.semantic import SemanticMemory
from aura.types import Experience, Hypothesis, Skill, Source


class EvidenceAnchorTest(unittest.TestCase):
    def test_memory_revisions_and_externally_pinned_prefix(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / 'ledger.sqlite'
            ledger = EvidenceLedger(path)
            episodic = EpisodicMemory(ledger)
            image = np.zeros((3, 7, 7), dtype=np.uint8)
            for i, reward in enumerate((1., -.01)):
                episodic.add(Experience(image, 4, image, reward, bool(i == 0), False,
                                        str(i), f'observed:{i}', seed=i, context='key'))
            semantic = SemanticMemory(ledger)
            semantic.revise(Hypothesis('opens', {'opens': True}, evidence_ids=['observed:0']))
            semantic.revise(Hypothesis('opens', {'opens': True}, evidence_ids=['observed:0'],
                                       counterevidence_ids=['observed:1'], confidence=.4))
            ProceduralMemory(ledger).revise(Skill('probe', {'near': 'door'}, 'interact',
                {'checked': True}, {'opens': True}, ['observed:0','observed:1'], 1, 1, ['key']))
            ledger.append('imagined', 'rollout', Source.IMAGINED, {'opens': True})
            with self.assertRaises(ValueError):
                ledger.append('fake', 'door_observation', Source.IMAGINED, {})
            with self.assertRaises(ValueError):
                episodic.reconstruct(ledger.records('rollout')[0])
            self.assertEqual(ledger.records('hypothesis')[1]['payload']['counterevidence_ids'], ['observed:1'])
            self.assertEqual(ledger.records('skill')[0]['payload']['failures'], 1)
            raw = canonical(checkpoint(path))
            pin = hashlib.sha256(raw).hexdigest()
            ledger.append('later', 'note', Source.EXTERNALLY_SPECIFIED, {})
            self.assertTrue(verify(path, raw, pin))
            self.assertTrue(ledger.verify())
            ledger.close()
            with sqlite3.connect(path) as db:
                db.execute('DROP TRIGGER ledger_no_update')
                db.execute("UPDATE ledger SET recorded_at='changed' WHERE sequence=1")
            self.assertFalse(verify(path, raw, pin))

    def test_independent_replacement_and_rollback(self):
        with TemporaryDirectory() as tmp:
            original, replacement = [Path(tmp)/name for name in ('original','replacement')]
            first, second = [EvidenceLedger(p) for p in (original, replacement)]
            try:
                first.append('a','experience',Source.OBSERVED,{'reward':1})
                first.append('b','experience',Source.OBSERVED,{'reward':0})
                raw = canonical(checkpoint(original))
                pin = hashlib.sha256(raw).hexdigest()
                second.append('a','experience',Source.OBSERVED,{'reward':1})
                self.assertFalse(verify(replacement, raw, pin))
                second.append('b','experience',Source.OBSERVED,{'reward':1})
                self.assertTrue(second.verify())
                self.assertFalse(verify(replacement, raw, pin))
                self.assertFalse(verify(replacement, canonical(checkpoint(replacement)), pin))
            finally:
                first.close()
                second.close()
