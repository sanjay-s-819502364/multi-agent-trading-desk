# Multi-Agent Intraday Trading Monitor — Project Brief

Third portfolio project, deliberately separate from the fintech compliance
theme (Bedrock RAG project + planned Node.js compliance agent layer), to
show range and to build hands-on multi-agent orchestration experience.

Domain shifted from options to **intraday stock trading** partway through
design — cleaner fit for a demo (faster-moving data, clearer guardrails,
more visible agentic reasoning).

## Scope revision (post-brief): risk management dropped, signal/model quality is now the core

After this brief was written, scope was narrowed further: the risk
assessment calculation, strategy agent, and reconciliation/escalation
logic (all documented below) are **cut from the active build**. The
project now centers entirely on the **signal generation subsystem** — the
model-designer/orchestrator loop that writes, trains, and iterates a
traditional ML model under walk-forward optimization.

Everything below this point (risk calc, strategy agent, escalation tiers)
is kept for historical context on how the design evolved, but is **not
being built** unless scope is revisited. The "why this design is
defensible" section's points about WFO rigor and overfitting-awareness
still apply directly to the current scope; the points about bounding
agent autonomy via escalation no longer apply since there's no
trading-relevant action being gated.

---

## Design principle: not every step deserves to be an LLM agent

Early design had four "agents" for every step. Correctly challenged during
design: market data pulling and risk math are deterministic — wrapping them
in an LLM call adds cost and unreliability without adding judgment. Revised
to a **two-tier architecture**: plain scripts for deterministic steps, LLM
agents only where reasoning/negotiation/code-generation actually happens.
This distinction is itself a strong interview talking point — knowing where
an LLM adds value vs. where it's just an expensive function call.

---

## Architecture

### Deterministic (script, no LLM)
- **Market data pipeline** — pulls/normalizes intraday price, volume,
  momentum data (live or simulated), fast refresh rate
- **Risk assessment calculation** — position size vs. account, drawdown,
  volatility, stop-loss proximity, sector/correlation risk. Pure
  calculation against formulas/thresholds. (Optional: LLM call afterward
  just to narrate the finding in plain language — not required for the
  core logic.)

### Agentic (LLM-based reasoning/negotiation/code-gen)
- **Signal generation subsystem** — see below, its own loop
- **Strategy agent** — reasons over risk assessment + signal generation
  output, proposes actions (trim position, tighten stop, exit). This is
  where judgment/negotiation actually lives.
- **Orchestrator agent** — manages workflow, invokes other agents/scripts,
  runs the reconciliation loop, enforces the approval gate, and (see below)
  drives the iterative model-improvement loop.

---

## Scope decision: risk management, not trade origination

Explicitly considered and deliberately scoped out (for v1): the strategy
agent does **not** originate new trades (entry signal, sizing, profit
targets). It only manages risk on positions that already exist — trim,
tighten stop, exit.

**Why:** "AI that decides when to enter trades" invites much heavier
scrutiny around backtesting rigor and overfitting than "AI that manages
risk on positions already held." The narrower scope is the more defensible
portfolio story and matches existing strengths (risk/hedging discipline
over signal generation).

Signal generation was then reframed (see below) not as "LLM predicts the
market" but as "LLM builds and iterates the traditional ML model that does
the predicting" — a meaningfully different and more defensible role.

---

## Signal generation subsystem (the interesting part)

**Key insight:** trade origination/signal prediction is a traditional ML
problem (pattern recognition over structured time-series data — gradient
boosting, logistic regression, engineered features), not an LLM reasoning
problem. An LLM agent's job here is to be an **ML engineering agent** — it
writes and iterates the code that builds the model; a trained traditional
ML model does the actual predicting.

### Iterative loop (agent writes code, local compute trains/tests it)

1. **Orchestrator agent** → decides it's time to try a new model, sends a
   task to the **model designer agent**
2. **Model designer agent** → writes ML code (feature engineering, model
   choice, hyperparameters), **saves script to disk**
3. **Local file-watcher script** (e.g. Python `watchdog`) detects the new
   file, **runs training + walk-forward optimization (WFO) test**
4. **Results saved to disk** (structured — JSON/CSV — with per-window
   scores, not just one aggregate number)
5. **Orchestrator agent** polls for new results, reads them
6. **If score too low** → diagnose why, send new instructions back to the
   model designer agent
7. Loop continues until threshold met or iteration cap hit

### Why walk-forward optimization, not a static train/val/test split

A single static split risks the agent iterating until it fits that one
slice by chance (classic overfitting trap for agentic ML/AutoML loops).
WFO slides a training/validation window through time (train months 1-3 →
validate month 4; train months 2-4 → validate month 5; etc.), testing
whether an edge holds up across *different* market regimes, not one lucky
split. "Good enough" should be judged on **consistency across windows**,
not just average score — a model that spikes on one window and fails on
the rest is a red flag even with a fine average.

This directly echoes the "document-level vs. claim-level scoring" lesson
from the Bedrock RAG project: don't trust an aggregate score that could be
hiding a real failure underneath.

### Open design decisions for this subsystem (not yet finalized)
- **How does the orchestrator "understand why" a score is low?**
  - Option A: LLM reasons directly over raw per-window metrics
    (more "agentic," riskier — could misdiagnose)
  - Option B: lightweight programmatic diagnostic step first (flag "high
    variance across windows," "weak in volatile periods," etc.)
    programmatically, then hand that summary to the orchestrator
    (more reliable/cheaper)
  - Leaning: B, or a hybrid, for reliability — not yet committed
- **File naming/versioning** — each iteration's script + results need to
  be tagged (iteration number/timestamp) so the full history is traceable
  (iteration 3 tried feature X, scored Y → iteration 4 changed Z)
- **Sandboxing** — the local script executes LLM-generated code. Needs a
  restricted environment (resource limits, no network access from the
  generated model script) even for a portfolio/demo context
- **Polling interval / max iteration cap** — needs an explicit cap so the
  loop can't run indefinitely or oscillate without converging

---

## Escalation logic (three tiers) — for the strategy agent's output

1. **Autonomous agreement** — risk assessment and strategy agent's proposal
   don't conflict → proceed
2. **One reconciliation pass** — if they conflict, strategy agent sees the
   risk agent's specific objection and gets **one** chance to revise its
   proposal or justify holding firm
3. **Forced human escalation** — if still unresolved after that pass, OR
   immediately if a hard-coded safety threshold is breached (e.g. max
   drawdown per position) regardless of what the agents concluded between
   themselves

**Why one reconciliation pass, not immediate escalation on any
disagreement:** escalating on every disagreement is basically an alerting
system with extra steps — doesn't showcase meaningful agentic reasoning.
One negotiated pass produces a richer artifact while hard safety limits
still bound actual trading-relevant decisions. Good interview story about
knowing where autonomy should and shouldn't apply.

---

## Why this design is defensible in interviews

- Shows judgment about **when an LLM should and shouldn't be used** in a
  pipeline (scripts vs. agents split), not "wrap everything in an LLM"
- Shows understanding that **trade signal generation is a traditional ML
  problem**, and correctly separates "agent that builds/improves a model"
  from "agent that predicts the market"
- Shows **agent-to-tool handoff via file system**, a pattern beyond basic
  function calling
- Shows overfitting-awareness **built into the design from the start**
  (WFO, per-window consistency, train/val/test discipline) rather than
  bolted on after a failure
- Reconciliation/escalation logic gives a concrete, explainable story about
  bounding autonomy in a real-money-adjacent domain

---

## Status: ready to start building

**Data source — decided:** simulated/historical intraday data, not a live
API. Chosen for reproducibility and easier demoing/evaluation (same
reasoning as the walk-forward design — a fixed historical dataset lets
results be reproduced and audited, rather than depending on whatever the
market happened to do on demo day).

Decisions still open before/while building:
- Tech stack (Python/Bedrock vs. LangGraph vs. custom orchestrator) —
  not yet decided
- Repo scaffolding / structure
- Orchestrator diagnostic approach (Option A vs. B above)
- Sandboxing approach for LLM-generated model code
