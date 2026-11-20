# Objectives

The model optimizes `MSE(ensemble_next_image, actual_next_image) + 0.05 MSE(predicted_next_latent, stop_grad(encoded_next)) + 0.25 MSE(reconstructed_current_image, current_image) + 0.05 ReLU(0.3 - per_dim_std(encoded_current)).mean()` using Adam and gradient clipping (norm 1). Distinct ensemble heads are randomly initialized but NOT independently bootstrapped; epistemic estimate is uncalibrated disagreement, and predictive image error includes aleatoric uncertainty.

Actor uses smooth-L1 one-step TD error `r + gamma*(1-terminal)*max_a Q(z_next,a)` with gamma 0.95. Time-limit truncation is bootstrap-eligible, while terminal episodes are not. Curiosity utility combines disagreement / `(1+4*noise_floor)`, held-out learning progress on a fixed replay sample, action-diversity bonus and fixed interaction cost. A progress update compares the *same* local validation batch.

Promotion uses paired heldout MSE difference `baseline - candidate`, a 95% paired bootstrap CI, at least 5 paired seeds, `LCB > 0.0001` and worst per-seed regression <= 0.015. Repeated looks are NOT alpha-spending corrected, so repeated search invalidates nominal 95% claims; this is a conservative prototype gate, not confirmatory research.
