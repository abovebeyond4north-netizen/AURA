# v54 experimental posterior camera fallback

Status: REJECTED for promotion. Section 08 remains PARTIAL; 13/31 locally verified in the inherited register.

## Intervention

`BeliefGoalCamera` extends the previously preferred `PassageRecoveryCamera` and invokes the v53 `PoseBelief` only after stationary, palette and conventional visual-displacement methods fail. The posterior uses a known independent 3% corruption model and an 80% expected-motion prior. It rejects motion that disagrees with the camera contract. It does not use simulator coordinates. The mechanism is not a general persistent pose belief or learned likelihood.

## Matched evaluation

All policies have 65 actions maximum; identical seed/layout/reversal/noise pairings. Development seeds 73100-73104: clean baseline 40/40, candidate 40/40; 3% corruption baseline 38/40, candidate 38/40. Fresh seeds 74100-74109: clean baseline 80/80, candidate 80/80; 3% corruption baseline 78/80, candidate 78/80. Total 480 trials (240 per controller), including both clean and noisy. Full per-trial action/outcome traces are stored alongside the source archive.

21 focused tests passed across pose belief and navigation recovery tests. Full repository regression was not executed. The candidate has no observed goal-success benefit and must not replace the preferred controller. This study does not independently certify the 13-section register.

## Follow-up

Instrument actual posterior fallback usage and assess likelihood calibration, abstention and localization accuracy on a truly disjoint noise model and larger layouts. Avoid adding heuristics absent measured gain.
