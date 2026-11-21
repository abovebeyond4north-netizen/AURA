# AURA-Ω external evaluation handoff — NOT YET COMPLETED

**Status:** No outside evaluator has been identified or authorized. Neither GitHub-hosted
CI nor public confirmation establishes independent certification or allows promotion.

## Separation of duties

1. Freeze candidate source at a specific immutable Git commit before evaluation designs
   and secret seeds are disclosed to project authors. Check the protected verifier and
   `docs/section_acceptance.md` against that commit.
2. An organizationally independent assessor controls its own runtime, hidden layouts,
   changed-action dynamics, observations, trial budgets, raw results and signing key.
   A GitHub-hosted runner using project-authored tests is NOT independently administered.
3. The assessor preregisters a private manifest hash. Include unexpected failures and
   negative controls, not just successes. Keep the hidden manifest inaccessible to
   candidate authors until after the code is frozen and the trial is complete.
4. Run paired trials for EfficientRouteCamera, PersistentPoseCamera and
   ComposedRouteCamera with matched seeds and action budgets, across unseen sizes,
   door locations, movement regimes, corruption and additional task families.
   Include v117/public-v1 failure regimes. Report every episode and its action trace.
5. Record source commit SHA, environment/dependency versions, manifest SHA-256,
   per-trial configurations and actions, independent diagnostics, all exceptions,
   the complete raw results and a matched summary. Do not silently drop timeouts.
6. Emit a `AURA-external-v1` receipt with `source_commit`, `manifest_sha256`,
   `rows`, and `summary`. Bind receipt to the hidden manifest via its SHA-256,
   and sign it with a detached GPG signature controlled by the assessor.
7. A reviewer obtains the public key and its 40-hex fingerprint through an
   independently authenticated channel and runs:

   ```bash
   python -m tools.verify_external_eval \
     --receipt external/receipt.json --manifest external/manifest.json \
     --signature external/receipt.json.asc --public-key external/evaluator.asc \
     --trusted-fingerprint OUT_OF_BAND_REVIEWED_40_HEX_FINGERPRINT
   ```

   The script checks signature, signer fingerprint, manifest hash, source commit shape,
   full paired-trial coverage, action budgets and summary consistency.
   It deliberately returns `independence_verified: false` and
   `acceptance_promoted: false`: independent identity, assignment integrity,
   unverifiable environment details, and absence of conflicts require human review.
8. Reproduce the source checkout, verify experimental controls and statistical
   uncertainty, then review each original acceptance section separately. No automatic
   changes to the protected verifier or promotion state.

## Current public experiment

The committed public matrix `public_holdout_v1.json` is non-blinded and controlled.
Locally reproduced on v118: 216 trials, 20/72 EfficientRoute goals,
53/72 PersistentPose goals and 49/72 ComposedRoute goals.
The composition gained 8 paired trials and lost 12 versus persistent localization.
This is negative/limited generalization evidence, not independent evaluation.

## Blocking prerequisites

An actual outside assessor, source freeze, confidential trial manifest, independent
execution, valid signed receipt, raw results and documented conflict-of-interest review.
These cannot be replaced by generated identities, self-signed results or CI alone.
