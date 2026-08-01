# Live benchmark evidence — live-suite-r1

**This is an exploratory live benchmark, not a statistically significant evaluation.**

Sanitized evidence snapshot from benchmark `live-suite-r1-20260801-142257`, frozen for portfolio review. Raw run artifacts (prompts, generated code, SDK responses) are intentionally excluded.

## Headline results

In this exploratory six-task run, the repair-enabled contract mode succeeded on 5 of 6 trials, compared with 1 of 6 for each other mode.

| Mode | Successful trials | Success rate |
| --- | --- | --- |
| single_agent | 1/6 | 16.67% |
| contract_no_repair | 1/6 | 16.67% |
| contract_with_repair | 5/6 | 83.33% |

## Run summary

- **Dataset:** software_delivery v1 (6 tasks)
- **Model:** gpt-5.6-luna (generation and review)
- **Trials:** 18/18 completed; 0 provider or infrastructure failures
- **Successful trials:** 7
- **Observed provider calls:** 56
- **Observed tokens:** 208,099 (119,766 prompt + 88,333 completion)
- **Wall clock:** 714.7 seconds
- **Repair attempts:** 4
- **Repair successes:** 3

## Limitations

- Same-model reviewer (generation and review both used gpt-5.6-luna)
- Generated code was statically inspected but **not executed**
- Pricing unavailable (not configured)
- One repetition per task/mode — no variance estimate
- Deterministic gates only; no runtime test execution

## Failure categories

| Category | Count |
| --- | --- |
| reviewer_rejected | 10 |
| deterministic_gate_failed | 1 |

## Files

| File | Purpose |
| --- | --- |
| `summary.json` | Compact aggregate metrics (schema v1.0) |
| `results.csv` | Per-trial outcomes (sanitized) |
| `methodology.json` | Run design and caveats |

## Earlier pilot (separate run)

The following results are from a **separate pilot** (`live-url-shortener-r3-20260801-134858`) and are **not** included in the 18-trial suite statistics above.

URL-shortener only, 3 repetitions per mode:

| Mode | Successful trials |
| --- | --- |
| single_agent | 1/3 |
| contract_no_repair | 3/3 |
| contract_with_repair | 1/3 |

- Repair attempts: 3
- Repair successes: 1

This pilot used the same model and methodology but a single task with higher repetition; it should not be compared directly to the six-task suite totals.
