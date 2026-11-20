import unittest
import numpy as np

from aura.cognition.door_mechanisms import DoorCompetition, MODELS, confirm
from aura.cognition.grid_door_study import replay
from aura.cognition.hypotheses import HypothesisEngine
from aura.environments.procedural_world import ProceduralWorld
from aura.memory.evidence_ledger import EvidenceLedger


class DoorStudyTests(unittest.TestCase):
    def test_four_live_competitors_revise_and_discriminate(self):
        contest = DoorCompetition()
        for i in range(12):
            contest.observe(0, False, 'absent'+str(i))
            contest.observe(3, True, 'present'+str(i))
        np.testing.assert_allclose(contest.weights[:4],contest.weights[0])
        information, p = contest.design()
        self.assertGreater(min(p[1:3]),max(p[0],p[3]))
        self.assertGreater(information[1],information[0])
        before = contest.weights.copy()
        contest.observe(2, None, 'sensor-failed')
        np.testing.assert_array_equal(contest.weights,before)
        contest.observe(2, True, 'separating')
        self.assertGreater(contest.weights[0],contest.weights[1])
        hs = {h.expression['model']: h for h in contest.hypotheses('blind')}
        self.assertIn('separating',hs['KEY'].evidence_ids)
        self.assertIn('separating',hs['SWITCH'].counterevidence_ids)
        self.assertNotIn('sensor-failed',hs['KEY'].counterevidence_ids)
        self.assertGreater(hs['KEY'].confidence,hs['SWITCH'].confidence)
        self.assertNotEqual(contest.design()[1].tolist(),p.tolist())

    def test_absolute_support_rejection_and_partial_identification(self):
        for name, table in MODELS.items():
            counts = [[0,64,0] if b else [64,0,0] for b in table]
            self.assertEqual(confirm(counts)['status'],name)
        self.assertEqual(confirm([[0,64,0]]*4)['status'],'ALL_CANDIDATES_REJECTED')
        self.assertEqual(confirm([[0,0,64]]*4)['status'],'INCONCLUSIVE')
        self.assertEqual(confirm([[1,0,0]]*4)['status'],'INCONCLUSIVE')
        self.assertEqual(confirm([[64,0,0],[0,0,64],[0,0,64],[0,64,0]])['status'],'INCONCLUSIVE')
        wide=confirm([[15,0,49]]*4)['simultaneous_intervals'][0]
        self.assertEqual(wide[0],0)
        self.assertGreater(wide[1],.75)

    def test_budget_censoring_raw_replay_and_changed_actions(self):
        ledger=EvidenceLedger()
        self.addCleanup(ledger.close)
        engine=HypothesisEngine()
        result=engine.investigate_door(lambda:ProceduralWorld('switch',max_steps=1),ledger,
                                       study_id='blind',seed=19,coupled_trials=0,
                                       discovery_trials=1,confirmation_per_arm=1)
        self.assertEqual(result['status'],'INCONCLUSIVE')
        self.assertEqual(result['costs'],{'trials':5,'actions':5})
        self.assertTrue(ledger.verify())
        self.assertEqual(len(engine.hypotheses),4)
        data=ledger.records('door_observation')[0]['payload']
        arm=ledger.records('door_assignment')[0]['payload']['arm']
        self.assertEqual(replay(data,arm),(None,'episode ended before probe completed'))
        changed=dict(data,actions=[(data['actions'][0]+1)%5])
        with self.assertRaises(ValueError):
            replay(changed,arm)
        with self.assertRaises(ValueError):
            replay(dict(data,frames_sha256='wrong'),arm)

    def test_shifted_door_survey_retains_measurable_motion(self):
        from aura.exploration.door_probe import measure_door
        from experiments.section09_acceptance import EvaluationCamera
        world = ProceduralWorld('key',11,noise=.01,max_steps=80)
        env = EvaluationCamera(world, {'door_column':6}, {})
        image,_ = env.reset(seed=106)
        result = measure_door(image,lambda a: env.step(a)[0],2)
        self.assertEqual(result,(True,None))

    def test_legacy_hypothesis_and_observational_controls(self):
        from tests.test_hypotheses import (test_hypothesis_revision_and_inconclusive,
            test_causal_inference_needs_interventions_and_sufficient_samples)
        from tests.test_causal_interventions import (test_confounding_creates_false_passive_association,
            test_randomized_control_recovers_no_causal_effect,test_randomized_control_detects_real_effect)
        for check in (test_hypothesis_revision_and_inconclusive,test_causal_inference_needs_interventions_and_sufficient_samples,
                      test_confounding_creates_false_passive_association,test_randomized_control_recovers_no_causal_effect,
                      test_randomized_control_detects_real_effect):
            with self.subTest(check=check.__name__):
                check()


if __name__=='__main__':
    unittest.main()
