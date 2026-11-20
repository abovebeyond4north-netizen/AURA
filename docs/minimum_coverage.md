# Minimum implementation map (v36)

Coverage is a source-level entry point, not an acceptance pass. Open gates remain open. No deployment occurs in the new smoke command.

| Section | Minimum executable mechanism or evidence entry point |
|---|---|
|01|`scripts.minimum_architecture`: integrated learning plus bounded study; mission open|
|02|`protected_verifier.worker`, namespace probe; evaluation blocked|
|03|`pyproject.toml`, Config; dependency matrix incomplete|
|04|`aura.agent.AuraOmega`|
|05|`aura.perception.encoder`|
|06|`aura.world_model.hierarchy`, `temporal_heads`|
|07|`aura.world_model.uncertainty`|
|08|`aura.exploration.curiosity`, `visual_frontier`|
|09|`aura.cognition.intervention_study`|
|10|`aura.memory.evidence_ledger`|
|11|`aura.cognition.concepts`|
|12|`aura.skills.online_options`|
|13|`aura.curriculum.generator`|
|14|`experiments.continual_learning`|
|15|`aura.meta_learning.controller`|
|16|`aura.self_improvement.study`|
|17|`protected_verifier.worker`; blocked in real smoke|
|18|`aura.meta_learning.resource_allocator`|
|19|`aura.environments.procedural_world`|
|20|repository tree|
|21|`aura.types`|
|22|`AuraOmega.train`|
|23|`aura.world_model.dynamics`|
|24|`experiments.gridworld_exploration`|
|25|`aura.self_improvement.study`; two accepted generations not demonstrated|
|26|bounded integrated smoke and existing phase register|
|27|`python -m scripts.minimum_architecture --output NEW_DIRECTORY`|
|28|`tests/`; full suite incomplete in this run|
|29|Git history and verification logs|
|30|`docs/section_acceptance.md`; final acceptance open|
|31|fail-closed study and protected promotion policy|

The smoke produces a report even when the protected verifier rejects startup. Exit 2 indicates the two-generation criterion was not met; it does not mean the learning smoke failed. Tests of mocked verifier rejection are workflow tests only.

## v37 integrated execution baseline

`python -m scripts.minimum_architecture --output NEW_DIRECTORY --all-tasks --steps-per-task 40` exercises all five procedural task families with fresh agents. Reports include separate training/development interaction totals, episode counts, checkpoints and evidence-ledger integrity. The online feedback bank (`aura.evaluation.online_development`) is fixed and excluded from replay; budget exhaustion disables the signal. A development bank is not independent promotion evidence because it is reused for adaptive feedback. Eight additional development interactions are allocated per trained agent; comparisons with random-only agents must account for this extra cost. The 31-section source map above remains in force; no acceptance gate was upgraded.
