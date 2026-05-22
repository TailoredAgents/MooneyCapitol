# OpenAI AI Features Plan

## Purpose

OpenAI should be used as a read-only explanation, summarization, journaling, and research layer for MooneyCapitol. It should make the scout, reports, learning output, and ticker context easier for a trader/operator to understand.

OpenAI must not place trades, size trades, approve copier actions, override risk controls, or run inside the sub-300ms copier hot path.

## Approved AI Features

The approved feature set is:

1. Scout alert explanations.
2. Daily trading recap.
3. Learning report translation.
4. Trade journal automation.
5. Ticker/catalyst research summaries.

The explicitly excluded feature is:

- Chat-style or assistant-style dashboard Q&A.

The trader does not want a dashboard assistant. Do not build an interactive chat surface unless this product decision changes later.

## Model Routing

Use different models by task instead of sending everything to one expensive model.

| Feature | Default Model | Upgrade Model | Timing | Notes |
|---|---|---|---|---|
| Scout alert explanations | `gpt-5.4-mini` | `gpt-5.4` | Near-real-time, after alert creation | Short, fast explanation from structured alert data. |
| Daily trading recap | `gpt-5.4` | `gpt-5.5` | End of day | Higher reasoning value because it reviews trades, copied fills, slippage, latency, and learning signals. |
| Learning report translation | `gpt-5.4-mini` | `gpt-5.4` | Nightly after learning job | Convert feature importance and model metrics into plain English. |
| Trade journal automation | `gpt-5.4-mini` | `gpt-5.4` | Background after trade/result finalizes | Structured, repetitive journal entry generation. |
| Ticker/catalyst research summaries | `gpt-5.5` | `gpt-5.5` with web/search tools | On alert or premarket batch | Highest risk for bad summaries; use stronger model and source-aware prompts. |

Operational rule:

- Default to `gpt-5.4-mini` for short structured text.
- Use `gpt-5.4` or `gpt-5.5` only when the feature needs deeper synthesis or source-sensitive research.
- Keep model names configurable with environment variables so they can be changed without code edits.

Suggested env vars:

```text
OPENAI_API_KEY=
OPENAI_SCOUT_EXPLANATION_MODEL=gpt-5.4-mini
OPENAI_DAILY_RECAP_MODEL=gpt-5.4
OPENAI_LEARNING_TRANSLATION_MODEL=gpt-5.4-mini
OPENAI_TRADE_JOURNAL_MODEL=gpt-5.4-mini
OPENAI_RESEARCH_MODEL=gpt-5.5
OPENAI_AI_FEATURES_ENABLED=0
OPENAI_RESEARCH_ENABLED=0
```

## Architecture

Add a separate AI service layer:

```text
Scout / reports / learning / ledger data
        |
        v
app/services/ai_*.py
        |
        v
OpenAI API
        |
        v
Stored explanation/report/journal/research artifact
        |
        v
Dashboard and/or Slack display
```

Rules:

- AI calls must be async/background where possible.
- AI failures must not fail scanner, copier, reports, or learning jobs.
- AI output must be stored with enough source data to audit what was summarized.
- AI output should be regenerated safely if prompts or models improve.
- AI content should be labeled as explanation/summary, not as trading advice.

## Phase 1: Scout Alert Explanations

Goal: each PRIMED and Active alert gets a concise explanation.

Inputs:

- Symbol.
- Direction.
- Entry, stop, target, R:R.
- Spread.
- RVOL.
- L2 display/imbalance if available.
- p2R score.
- Box summary.
- Gating notes and warning pills.

Output:

- One short sentence for Slack.
- One slightly richer paragraph for the dashboard card/detail view.
- Optional factor list: setup quality, risk/reward, volume, L2/depth confirmation, and warning conditions.

Acceptance criteria:

- Explanation appears after the alert is created, not before.
- A slow OpenAI request cannot delay Slack alert delivery or dashboard lane updates.
- If OpenAI fails, the alert still appears normally.
- Stored output can be reused by Slack/dashboard without another API call.

## Phase 2: Learning Report Translation

Goal: translate nightly XGBoost/logistic fallback output into operator-readable insight.

Inputs:

- Model type.
- Feature importance.
- Source breakdown.
- Ranking metrics.
- Training row count.
- Fallback reason if applicable.

Output:

- Plain-English nightly summary.
- Top 3 drivers.
- Top warning/caution.
- Whether the model changed meaningfully from prior run.

Acceptance criteria:

- The existing learning job still completes if OpenAI fails.
- Slack can receive a short translated learning note.
- Dashboard Launch/Learning area can display latest AI-translated learning note later.

## Phase 3: Trade Journal Automation

Goal: automatically create structured trade notes from master trades and copied outcomes.

Inputs:

- Master execution and fills.
- Copied order/fill result.
- Latency.
- Slippage.
- P&L/R multiple when available.
- Whether scout alerted first.
- Alert payload/explanation if linked.

Output:

- Journal title.
- Setup summary.
- Execution summary.
- Outcome summary.
- Copy-performance summary.
- Tags: scout_alert, manual_no_alert, copied, rejected, high_slippage, under_300ms, etc.

Acceptance criteria:

- Journal notes are generated in background after data exists.
- Manual no-alert trades are clearly labeled.
- Copied-account performance is separated from master performance.

## Phase 4: Daily Trading Recap

Goal: generate an end-of-day report that a human can read quickly.

Inputs:

- Ledger daily summary.
- Master trades.
- Copied trades.
- Rejected/blocked orders.
- Latency distribution.
- Slippage.
- Scout alerts taken/ignored.
- Learning report summary if available.

Output:

- Daily headline.
- Best/worst trades.
- Copier health.
- Scout quality.
- Learning observations.
- Issues to check before next session.

Acceptance criteria:

- Report posts to Slack and can be displayed through existing report endpoints later.
- Report does not make trading promises or automated recommendations.
- Report separates confirmed facts from inferred interpretation.

## Phase 5: Ticker/Catalyst Research Summaries

Goal: provide quick source-aware context for alerted symbols without slowing alerts.

Inputs:

- Symbol.
- Alert type/state.
- Price/volume context.
- Available external market/news/filing data.

Output:

- Catalyst summary.
- Known news/filing context.
- Risk flags such as dilution, offering, halt/news uncertainty, reverse split, or low-float volatility.
- Source list or source timestamp where available.

Acceptance criteria:

- Research runs after alert creation or in premarket batches.
- Research summary is clearly timestamped.
- If sources are missing or stale, output says so.
- Research does not block the scout or copier.

## Data Storage

Preferred table design:

```text
ai_artifacts
- id
- artifact_type
- source_type
- source_id
- symbol
- model
- prompt_version
- input_json
- output_json
- output_text
- status
- error
- created_at
- updated_at
```

Potential `artifact_type` values:

- `scout_explanation`
- `learning_translation`
- `trade_journal`
- `daily_recap`
- `ticker_research`

## Safety Rules

- Never call OpenAI before child order submission in the copier hot path.
- Never let AI output modify trade sizing or broker orders.
- Never treat AI research as verified unless source metadata is present.
- Always store model and prompt version with output.
- Add budget controls before enabling all features at once.
- Start with AI features disabled by default in Render.

## Recommended Implementation Order

1. Add OpenAI config/env parsing and a no-op-safe AI client wrapper. (Phase 1 implemented)
2. Add `ai_artifacts` table and repository. (Phase 1 implemented)
3. Implement Scout alert explanations first. (Phase 2 implemented)
4. Implement learning report translation second. (Phase 3 implemented)
5. Implement trade journal automation third.
6. Implement daily recap fourth.
7. Implement ticker/catalyst research last, after source handling is designed.
