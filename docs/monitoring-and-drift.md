# Monitoring and Drift

## Purpose

THAI-MOD is a multilingual Thai + English toxicity detection system used as a decision-support tool for moderators. The monitoring addition stays intentionally lightweight:

- keep one fixed reference profile
- log real prediction requests from the running app
- summarize the latest recent-request window
- compare recent traffic against the reference profile
- show a compact drift summary in the admin page

This is a course-scale monitoring capability. It is not a full production observability stack.

## What is monitored

The monitoring summary compares:

- `reference_profile`
- `recent_live_requests`

Recent live requests come from the normal app path:

- `POST /api/predict`
- `POST /api/batch-predict`

Each prediction request appends one monitoring event into a local persistent JSONL log.

## Reference profile

The fixed reference profile is now a generated artifact, not an unexplained hand-prepared CSV.

Artifacts:

- `datasets/monitoring/reference_profile.json`
- `datasets/monitoring/reference_batch.csv`

Generation path:

- `python scripts/build_reference_profile.py`

Why a fixed reference is needed:

- data drift compares recent observed traffic against a baseline distribution
- without a fixed baseline, PSI has no stable point of comparison
- the baseline should stay fixed during runtime and presentation so changes in recent traffic are interpretable

## Reference source and rationale

The repository does not contain historical production traffic, so the reference profile is built from project datasets to approximate expected deployment traffic.

The current reference baseline is generated as:

- deterministic holdout built from all `8` project datasets
- preprocess-aligned deduplication on `processed_text`
- up to `100` samples per dataset
- deterministic sampling with seed `42`
- current total sample count = `800`

This keeps the baseline:

- source-diverse, because every dataset contributes to the baseline
- protected from large-source dominance, because the biggest datasets are capped
- still multilingual, because the final language mix emerges from all source datasets together
- reproducible, because the script always rebuilds the same reference batch and profile

The important design choice is:

- the script does **not** fix a Thai/English ratio in advance
- it fixes the generation procedure instead
- the actual reference language mix is whatever results from that reproducible all-source sampling process

This is positioned honestly as:

- a reproducible baseline profile constructed from representative project data to approximate expected deployment traffic

It is not claimed to be real production traffic.

## Recent-request log

Storage:

- local JSONL file
- `models/monitoring_recent_requests.jsonl`

Stored fields:

- `timestamp`
- `text_length`
- `english_char_ratio`
- `language_bucket`
- `toxicity_score`
- `predicted_label`

The raw comment text is not stored in the monitoring log.

## Admin access

The admin page and monitoring endpoints use the app's session-based admin login:

- configure `THAI_MOD_AUTH_USERNAME`, `THAI_MOD_AUTH_PASSWORD`, and `THAI_MOD_SESSION_SECRET`
- sign in through `/login` before opening `/admin`
- `/api/monitoring` and `/api/monitoring/reset` require an authenticated session

## Recent monitoring window

The monitoring summary uses:

- the latest `100` logged requests

Minimum-request handling:

- fewer than `20` requests
  - state = `collecting data`
  - the admin page shows recent traffic metrics but does not show a real drift verdict yet
- `20` to `49` requests
  - drift is computed
  - the result is marked `provisional`
  - this is useful for a live demo, but the signal is still low-confidence
- `50` or more requests
  - normal drift status is shown using the standard thresholds

This keeps the monitor more credible than treating very small windows as stable.

## Metrics

The admin panel stays intentionally small. It shows:

- drift state
- PSI
- primary drift
- recent request count
- recent window size
- reference vs recent language mix
- reference vs recent `toxic_ratio`
- reference vs recent `average_toxicity_score`
- reference vs recent `average_text_length`

The reference profile artifact also stores:

- `sample_count`
- `source_counts`
- `language_bucket_counts`
- `psi_reference_distribution`
- `model_context`

In the current generated profile, the observed language mix is approximately:

- `thai_only 55.00%`
- `english_only 37.50%`
- `mixed_script 7.25%`
- `other 0.25%`

## Primary data drift

The headline drift signal is:

- `Language mix drift`

Language mix is represented with these buckets:

- `thai_only`
- `english_only`
- `mixed_script`
- `other`

The app still computes `english_char_ratio` internally as a supporting signal for each request, but it is no longer the main drift concept shown in the UI.

## PSI computation

PSI is computed from the change in the language-mix distribution itself.

Reference distribution:

- proportion of `thai_only`
- proportion of `english_only`
- proportion of `mixed_script`
- proportion of `other`

Recent distribution:

- proportion of the same four buckets in the latest recent-request window

The monitoring summary applies PSI across those category proportions to measure how far recent traffic moved away from the reference profile.

This fits THAI-MOD better than a single numeric English-ratio feature because the project is explicitly about multilingual Thai + English moderation traffic.

Performance degradation is a separate issue. Drift can be detected from unlabeled traffic, but confirming recall or F2 degradation would require reviewed labels.

## Degradation handling

The degradation policy is documented separately from the admin UI.

- `healthy`
  - continue routine review
- `observe`
  - watch the next recent window
- `warning`
  - increase manual review and inspect recent shifted samples
- `degraded`
  - treat recent traffic as degraded and collect examples for follow-up

The system remains human-in-the-loop. Drift changes review priority, not the final moderation decision.

## Retraining and update policy

Retraining remains a documented workflow, not something inferred from unlabeled traffic.

The workflow is:

1. Repeated warning or degraded traffic increases review and investigation.
2. Moderator-reviewed examples from shifted traffic are collected when available.
3. Those reviewed examples are merged with the existing training data.
4. A fresh candidate model is retrained from scratch by default.
5. The candidate is compared against the current model on held-out evaluation data.
6. Evaluation should prioritize toxic recall, then F2 when available, then precision and flagging burden.
7. A candidate is promoted only if the safety-oriented objective is not worse.

Important boundary:

- the system does not fabricate recall or F2 degradation from unlabeled monitoring traffic
- the system does not compute `retraining_candidate` from recent requests alone

## Admin UI summary

The admin monitoring panel now shows only the monitoring signals:

- drift state
- PSI
- primary drift = `Language mix`
- recent request count
- recent window size
- reference vs recent language mix
- supporting metrics
- refresh button
- clear recent log button

The admin page no longer shows a recommended-action block. The operational policy is documented here instead.

## Simple live demo flow

1. Run the app.
2. Open the comment analyzer and `/admin`.
3. Click `Clear recent log`.
4. Send several comments that are reasonably close to the reference traffic mix through the normal analyzer flow.
5. Click `Refresh monitoring`.
6. Send more English-heavy or clearly mixed-script comments through the normal analyzer flow.
7. Click `Refresh monitoring` again.
8. Show that the recent language mix and PSI changed.

For a faster demo setup, the presenter can also use `POST /api/batch-predict` with several comments in one request. Those predictions still go through the same real app path and are included in the recent-request log.

## Limitations

- This is a lightweight local monitor, not a production observability stack.
- The reference profile is still a constructed baseline, not real historical production traffic.
- The request log is persistent on the local machine, but it is still a simple JSONL file.
- It does not store moderator-reviewed labels for the monitored window.
- It does not claim measured recall or F2 degradation for recent traffic.
- The admin panel uses manual refresh rather than automatic streaming updates.

## Files

- `src/thai_mod_api/monitoring.py`
- `src/thai_mod_api/main.py`
- `src/thai_mod_api/static/admin.html`
- `src/thai_mod_api/static/admin.js`
- `scripts/build_reference_profile.py`
- `datasets/monitoring/reference_profile.json`
- `datasets/monitoring/reference_batch.csv`
