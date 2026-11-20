# Section 13 · Curriculum

The optional prerequisite-aware scheduler chooses among ready tasks and defers already-mastered tasks while eligible tasks remain. It learns competence from observed outcomes and logs priority components for novelty, progress, prerequisite transfer, failure, and predicted difficulty. The declared graph includes exploration, prediction, acquisition, composition and adaptation.

The controlled workbench compared adaptive scheduling, fixed round robin and uniform random scheduling. All arms used the same learner and 45 training attempts per seed. Each arm encountered the same 20 hidden action sets. Adaptive scheduling passed the downstream task on 20 seeds, round robin on 19, and uniform random on 16. Each seed was also checked under 20 action relabelings given equally to all arms. These checks are repeated tests of the same underlying mechanism within a seed.

Reproduce with `python -m experiments.curriculum_transfer --output runs/section13.json` after creating the output's parent directory. Raw schedules, choices and outcomes are in `verification/section13/results.json`. The graph is supplied to the scheduler. The study matches attempts, uses a simple workbench, and does not establish an improvement in the default neural trainer or on novel prerequisite structures.
