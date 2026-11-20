import unittest

import numpy as np

from aura.world_model.uncertainty import TransitionRisk


class TransitionRiskTest(unittest.TestCase):
    def test_threshold_rank_and_sparse_action_abstention(self):
        risk = TransitionRisk({0:np.linspace(0,.2,19),1:[.5]})
        self.assertAlmostEqual(risk.bounds[0], .2 * 17 / 18)
        zero = np.zeros((3,7,7),dtype=np.uint8)
        bright = np.full_like(zero,255)
        self.assertFalse(risk.check(0,zero,zero)['shifted'])
        self.assertTrue(risk.check(0,zero,bright)['shifted'])
        self.assertIsNone(risk.check(1,zero,bright)['shifted'])
        self.assertIsNone(risk.check(2,zero,bright)['shifted'])

    def test_invalid_calibration(self):
        for values in ([], [float('nan')],[-.1]):
            with self.assertRaises(ValueError):
                TransitionRisk({0:values})
