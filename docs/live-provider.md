# Live OpenAI Provider Smoke Test

This project keeps all normal pytest and CI runs offline. Stage 4A adds an optional OpenAI provider that can be exercised manually when you explicitly opt in.

## Base installation versus OpenAI extra

Install the core package without OpenAI for mock-only usage:

```bash
python -m pip install -e .
```

Install the optional OpenAI SDK when you need the live provider:

```bash
python -m pip install -e ".[openai]"
```

For development and offline provider tests:

```bash
python -m pip install -e ".[dev,openai]"
```

Base installation supports:

- `import cognitive_agent_syndicate`
- `import cognitive_agent_syndicate.cli`
- `python -m cognitive_agent_syndicate --help`
- default mock CLI runs
- mock provider factory creation

OpenAI mode requires the `[openai]` extra.

## Environment setup

Keys are read from environment configuration only. Do not pass keys on the CLI.

Key precedence:

1. `OPENAI_API_KEY` is preferred.
2. Legacy `API_KEY` is used only when `OPENAI_API_KEY` is unset.

Whitespace-only values are treated as missing.

### Option A — local `.env` file

```bash
cp .env.example .env
```

Edit `.env` locally with your key. `.env` is gitignored and must never be committed.

### Option B — non-echoing shell prompt

```bash
read -s OPENAI_API_KEY
export OPENAI_API_KEY
echo
```

## Explicit provider selection

Mock mode remains the default and requires no API key:

```bash
python -m cognitive_agent_syndicate run examples/briefs/url_shortener.json
```

OpenAI mode requires both `--provider openai` and `--model`:

```bash
python -m cognitive_agent_syndicate run examples/briefs/url_shortener.json \
  --provider openai \
  --model MODEL_NAME
```

## Safe API key handling

- Keys are stored as `SecretStr` in settings and never appear in `repr(settings)` or `str(settings)`.
- Keys are not stored in `PipelineState` or run reports.
- Provider and CLI error messages avoid printing secrets, prompts, or raw response bodies.
- Live tests read keys only after opt-in checks inside the test function.

## Manual live smoke test (NOT RUN in CI)

Live tests are marked with `@pytest.mark.live`. Normal pytest and CI skip live-test **execution** via marker selection (`-m 'not live'`).

Opt in explicitly after configuring your key locally:

```bash
RUN_LIVE_TESTS=1 \
  pytest tests/live/test_openai_provider_live.py -m live -q
```

This performs one small structured-output provider request. It does not run the full three-agent pipeline, does not execute generated code, and live calls incur API usage and cost.

CI and default pytest runs also clear `RUN_LIVE_TESTS`, `OPENAI_API_KEY`, and `API_KEY` so inherited repository secrets cannot activate live tests accidentally.

## Known limitations

- Non-streaming, non-background Responses API calls only.
- No tools, web search, code interpreter, or generated-code execution.
- Each pipeline call is independent; no conversation state is retained remotely (`store=False`).
- Cost estimation is not implemented in Stage 4A.
- Mock mode and CI remain fully offline.
- Generated code is never executed by the pipeline.
- Prompts and API keys are excluded from persisted reports.
