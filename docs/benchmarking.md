# Benchmarking

This document describes the reproducible benchmark framework added in Stage 4B.

## Benchmark question

Does contract-driven decomposition improve valid artifact generation compared to a
single-agent baseline? Does one bounded repair attempt improve success rates? What
are the token, latency, and estimated-cost trade-offs?

Mock benchmark results validate the benchmark harness and do not measure real model
quality.

## Modes

| Mode | Description |
| --- | --- |
| `single_agent` | One generation call producing `SingleAgentDelivery` (architecture + artifacts), then review and gates |
| `contract_no_repair` | Architect → implementer → reviewer with `max_repair_attempts=0` |
| `contract_with_repair` | Same pipeline with `max_repair_attempts=1` |

## Trial status semantics

| Status | Meaning |
| --- | --- |
| `completed` | Provider calls finished and the result was evaluated, even when reviewer or gates rejected it |
| `failed` | The trial could not be evaluated normally because of provider, configuration, malformed output, persistence, or internal failure |
| `cancelled` | Execution was interrupted before normal evaluation |

Typed failure categories include `provider_configuration`, `provider_authentication`,
`provider_rate_limit`, `provider_timeout`, `provider_connection`,
`malformed_structured_output`, `reviewer_rejected`, `deterministic_gate_failed`,
`persistence_failed`, and `internal_error`.

Reviewer rejection and deterministic gate failure are **completed** trials with
`success=false`. Provider, configuration, malformed-output, persistence, and internal
failures are **failed** trials.

### Reviewer status versus failure category

| Field | Meaning |
| --- | --- |
| `reviewer_status` | Raw reviewer outcome: `approved`, `needs_revision`, or `rejected` |
| `failure_category=reviewer_rejected` | Any non-approved reviewer outcome (`needs_revision` or `rejected`) |

The failure category name is historical shorthand for “reviewer did not approve.” It does
not mean the raw status was necessarily `rejected`.

## Token accounting

| Location | Scope |
| --- | --- |
| `BenchmarkTrial.total_tokens` | Exact sum of all observed provider-call usage in the trial |
| Run-report usage totals | Same as trial totals: architect + implementer + reviewer (+ repair) provider calls |
| Run-report attempt provider-token row | Contract modes: implementer + reviewer (+ repair) only; architect tokens appear in usage totals but not attempt rows. Single-agent: baseline generation + reviewer |

Attempt-row provider-token counts are therefore usually lower than usage totals in contract
modes even when only one attempt ran.

## Rate denominators

Performance rates use **non-cancelled attempted trials** as the denominator:

`attempted = completed + failed` (cancelled trials excluded)

`success_rate = successful_trials / attempted`

The same denominator is used for reviewer approval rate, required-gate pass rate,
acceptance-criterion pass rate, syntax pass rate, forbidden-content pass rate,
required-files pass rate, and repair-attempt rate.

`repair_success_rate = repair_success_count / repair_attempt_count`

Repair-success rate uses repair-attempted trials as its denominator; both numerator
and denominator are stored and displayed explicitly.

Rates are stored as fractions. Displayed percentages use standard Python formatting
to two decimal places (`f"{rate:.2%}"`). Displayed fractions always match the
percentage denominator.

## Provider call counts

Plan-level minimum and maximum provider calls are estimates used only in dry-run
planning:

| Mode | Min | Max |
| --- | ---: | ---: |
| `single_agent` | 2 | 2 |
| `contract_no_repair` | 3 | 3 |
| `contract_with_repair` | 3 | 5 |

Trial-level `provider_call_count` is an **exact observation** of attempted
`generate()` calls, including calls that raise. Each trial uses a fresh counter;
generation and reviewer providers share one counter when they are the same instance.

## Fairness controls

- All modes share the same task definition, reviewer policy, required files, permitted
  paths, deterministic gates, model configuration, and temperature.
- The single-agent baseline is reviewed by the same `ReviewerAgent` and evaluated by
  the same `GateRunner`, including architecture-dependent gates.
- Task `notes` are excluded from agent context.
- Equivalent underlying failures produce equivalent benchmark status and category across
  all three modes.
- When generation and review use the same provider/model, that limitation is recorded
  in the benchmark summary.

## Dataset

Versioned dataset: `benchmarks/datasets/software_delivery_v1.json`

The dataset version is displayed exactly as stored (for example `software_delivery v1`).
The CLI and reports do not prepend an extra `v`.

Six bounded software-delivery tasks:

1. URL Shortening API
2. Support Ticket Classification Service
3. Document Ingestion API
4. Feature Flag Service
5. Inventory Reservation Service
6. Incident Summary Service

## Metrics

Per mode and per task:

- Attempted, completed, failed, cancelled, and successful trial counts
- Success rate, reviewer approval rate, gate pass rates
- Repair attempt and success rates (contract-with-repair mode only; counts actual repair
  runs, not mode eligibility)
- Total observed provider calls
- Mean, median, min, and max for tokens, provider latency, and wall-clock duration
- Failure-category counts
- Estimated cost when user-supplied pricing is configured

## Exit codes

| Code | Meaning |
| ---: | --- |
| 0 | Benchmark completed and every non-cancelled trial succeeded |
| 1 | Fatal benchmark command or execution failure |
| 2 | CLI usage or argument validation error (Click/Typer) |
| 3 | Benchmark completed, outputs were persisted, but one or more non-cancelled trials did not succeed |

## Benchmark IDs

Benchmark IDs must be a single safe path segment matching
`[A-Za-z0-9][A-Za-z0-9._-]{0,63}`. Path separators, traversal, absolute paths, and
empty values are rejected at the CLI and schema boundaries.

## Pricing file format

Example (fictional values only): `benchmarks/pricing/example-pricing.json`

```json
{
  "model_label": "example-model",
  "input_usd_per_million_tokens": "1.50",
  "output_usd_per_million_tokens": "6.00",
  "source_or_note": "Fictional example values...",
  "effective_date": "2026-01-01",
  "currency": "USD"
}
```

Costs use Decimal arithmetic. Rates are never claimed to be current unless you supply
and date them.

## Mock versus live

**Default:** mock/offline execution. No network calls.

**Mock:** Deterministic fixtures exercise success, reviewer rejection, gate failure,
repair success, repair failure, and provider failure scenarios.

**Live:** Requires all of:

- `--provider openai`
- `--model <name>`
- `--confirm-live`
- `RUN_LIVE_BENCHMARKS=1`
- repetitions ≤ 5

Live benchmarks print a dry-run plan before execution. API keys are read from
environment variables only (`OPENAI_API_KEY`); never pass keys on the CLI.

Normal pytest and CI clear `RUN_LIVE_BENCHMARKS` and never construct live clients.

## Offline commands

Plan (no provider calls):

```bash
python -m cognitive_agent_syndicate benchmark plan \
  --dataset benchmarks/datasets/software_delivery_v1.json \
  --modes single_agent,contract_no_repair,contract_with_repair \
  --repetitions 1 \
  --provider mock
```

Run mock benchmark:

```bash
python -m cognitive_agent_syndicate benchmark run \
  --dataset benchmarks/datasets/software_delivery_v1.json \
  --modes single_agent,contract_no_repair,contract_with_repair \
  --repetitions 1 \
  --provider mock \
  --output-dir benchmark_results
```

## Live command template

```bash
export OPENAI_API_KEY="your-key-here"
export RUN_LIVE_BENCHMARKS=1

python -m cognitive_agent_syndicate benchmark run \
  --dataset benchmarks/datasets/software_delivery_v1.json \
  --modes single_agent,contract_no_repair,contract_with_repair \
  --repetitions 1 \
  --provider openai \
  --model gpt-4.1-mini \
  --confirm-live \
  --output-dir benchmark_results
```

## Output structure

```
benchmark_results/<benchmark-id>/
  benchmark-config.json
  trials.jsonl
  summary.json
  summary.md
  results.csv
  failures.json
  runs/<task-id>/<mode>/<repetition>/run-report.json
```

## Reproducibility and limitations

- Sequential trial execution (no hidden concurrency in Stage 4B)
- Generated code is never executed during benchmarks
- Static gates are deterministic but not a complete security scanner
- Same-model reviewer reduces review independence
- CI never runs live benchmarks

## Automated live validation

Stage 4C adds a one-command live validation workflow that replaces the manual
export-key → smoke → plan → benchmark → cleanup sequence.

```bash
python -m cognitive_agent_syndicate validate-live \
  --task-ids task-url-shortener \
  --repetitions 3 \
  --model gpt-5.6-luna \
  --confirm-live
```

When `OPENAI_API_KEY` is not already set, the command securely prompts:

```
OpenAI API key:
```

Input is hidden (no echo). The key is never passed on the command line, never
persisted to disk, and never included in reports or `live-validation.json`.
On every exit path—including success, smoke failure, benchmark failure,
validation errors, and Ctrl+C—the prior credential environment is restored.

### Workflow

1. **Preflight** — validates dataset, tasks, modes, repetitions (≤ 5 live),
   model, output path, benchmark ID, and optional git cleanliness.
2. **Credential acquisition** — uses existing `OPENAI_API_KEY` or prompts once.
3. **Smoke test** — one structured-output provider call through the production
   path (`Settings` → `create_model_provider` → `OpenAIModelProvider`). If smoke
   fails, the benchmark does not start.
4. **Plan display** — prints trial and provider-call bounds before benchmark
   provider calls.
5. **Benchmark execution** — reuses the existing benchmark runner with live
   progress output at trial boundaries (and provider-call timing when available).
6. **Handoff** — prints a concise summary and writes `live-validation.json`
   beside benchmark outputs.

Use `--smoke-only` to run preflight and smoke without starting a benchmark.

Use `--allow-dirty` to override the default refusal to run on a dirty git working
tree.

### Progress output

During benchmark execution the terminal shows trial progress, for example:

```
[1/9] single_agent / repetition 1 started
      provider call 1 completed in 0.0s
[1/9] completed: completed
```

Prompts, generated code, API keys, and raw SDK responses are never printed.

### Safety

- `--confirm-live` is required before any live provider call.
- Generated code execution remains disabled during benchmarks.
- API usage may incur cost.
- Pre-run plans do not print monetary cost estimates because future token usage
  is unknown. When a pricing file is supplied, rates and metadata appear in the
  plan; actual cost is calculated after execution from observed token usage.
- Ctrl+C requests graceful cancellation. An in-flight provider call may finish
  before shutdown completes. Repeated Ctrl+C does not bypass credential cleanup
  or restore handlers early.
- Benchmark outputs are written under `benchmark_results/<benchmark-id>/`
  including `summary.md` and `live-validation.json`.
