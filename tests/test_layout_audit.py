import unittest

import numpy as np

from aura.environments.procedural_world import ProceduralWorld
from aura.environments.task_generator import layout_spec


class LayoutTests(unittest.TestCase):
    def test_layout_partitions_and_seed_namespaces(self):
        splits = [[layout_spec(split,i) for i in range(30)]
                  for split in ('train','dev','protected')]
        columns = [set(x['door_column'] for x in group) for group in splits]
        self.assertEqual(sum(map(len,columns)),len(set.union(*columns)))
        self.assertEqual(len({x['seed'] for group in splits for x in group}),90)

    def test_action_change_after_observed_key_collection(self):
        world = ProceduralWorld(reverse_after_key=True, diagnostics=True)
        world.reset(seed=1)
        # Before pickup, south moves down. After pickup, south moves up.
        world.step(1)
        self.assertEqual(world.diagnostic_state()['agent'],(2,1))
        world.step(1)
        self.assertEqual(world.diagnostic_state()['agent'],(1,1))

    def test_observation_noise_is_reproducible_and_has_no_physical_effect(self):
        clean, noisy, again = [ProceduralWorld(noise=noise, diagnostics=True)
                               for noise in (0,.25,.25)]
        a,b,c = [env.reset(seed=9)[0] for env in (clean,noisy,again)]
        np.testing.assert_array_equal(b,c)
        self.assertFalse(np.array_equal(a,b))
        for action in (1,3,4,1):
            for world in (clean,noisy,again):
                world.step(action)
            self.assertEqual(clean.diagnostic_state(),noisy.diagnostic_state())


if __name__ == '__main__':
    unittest.main()
