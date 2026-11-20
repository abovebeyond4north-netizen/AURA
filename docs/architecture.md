# Architecture and data contracts

`ProceduralWorld` emits partial RGB uint8 images `[3,7,7]`. `Experience` is tagged **OBSERVED** and validated on construction. `ExperienceReplay` stores only these verified transition objects; `WorldModel.imagine` returns tensors never appended to the ledger. The CNN maps `[B,3,7,7]` to `[B,Z]`; each independent GRUCell receives `[B,Z+5]` and returns predicted latent `[B,Z]`, image `[B,3,7,7]`, and hidden `[B,H]`. `Actor` predicts five Q-values from detached latent states. `EpisodicMemory`, `SemanticMemory`, and `ProceduralMemory` append evidence-linked revisions in one hash-chained SQLite ledger.

`Curriculum`, `MetaController`, `Curiosity`, `HypothesisEngine`, and `SkillDiscovery` consume interactions. Training, hypothetical rollouts, and heldout verifier runs are different execution contexts. The learner cannot supply arbitrary Python to the modification registry. For stronger physical isolation see limitations.
