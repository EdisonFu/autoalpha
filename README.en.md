# AutoAlpha

**An LLM multi-agent system for automated alpha factor mining — one research direction in, factors that survive out-of-sample testing out.**

[![tests](https://github.com/EdisonFu/autoalpha/actions/workflows/tests.yml/badge.svg)](https://github.com/EdisonFu/autoalpha/actions/workflows/tests.yml)
[![lint](https://github.com/EdisonFu/autoalpha/actions/workflows/lint.yml/badge.svg)](https://github.com/EdisonFu/autoalpha/actions/workflows/lint.yml)
[![license](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](pyproject.toml)

English · [简体中文](README.md)

```
              input: "cross-sectional momentum in A-share ETFs"
                                ↓
  ┌────────────────────────────────────────────────────────────────┐
  │  Intent     informal text → canonical query + market detection │
  │  Retrieval  arXiv + OpenAlex + Crossref in parallel → ranking  │
  │  Summary    batched summaries and relevance judgements         │
  │  Extract    papers → computable hypotheses (data-constrained)  │
  │  Implement  hypothesis → Python → sandbox → targeted self-fix  │
  │  Judge      economic soundness / fidelity / overfitting risk   │
  │  Extend     variants around passing factors + diversity filter │
  └────────────────────────────────────────────────────────────────┘
                                ↓
        deterministic engine: IS/OOS two-stage gating → library
```

**Seven LLM agents generate and judge; one deterministic engine decides.**
Agents can be wrong, hallucinate, or emit code that does not run — the engine
does not let any of that through. The split, the costs and the thresholds are
enforced by the engine and are not negotiable by an agent.

---

## What it solves

The real bottleneck in quant research is not running out of factor ideas.
It is that three separate things are expensive:

| Done by hand | Cost | AutoAlpha |
|---|---|---|
| Literature: read abstracts, judge relevance, translate | hours per batch | 110 records from three sources, each with a summary and a relevance call |
| Paper → running code: parse formula, align data, debug | hours per factor | sandboxed execution with repair targeted by failure class |
| Rigorous evaluation: layers, turnover, costs, OOS | easy to skip | enforced by the engine; cannot be skipped |

A full cycle measures at **1–2 minutes and $0.15–0.22** (four factors, including
retrieval and code generation), producing factor code traceable to its source
paper along with a complete backtest.

### What is technically hard

**This is not "ask an LLM for a formula".** Every step carries a concrete
engineering constraint:

- **Retrieval** — the same intent phrased differently differs by **800x** in
  recall (`ETF轮动` returns 18, the canonical English phrasing 14,494), so
  terminology must be normalised first. Market matters too: conclusions do not
  transfer across markets, so the market is detected and used to qualify the query.
- **Code generation** — generated code runs in a subprocess sandbox. Failures
  are classified as `NO_MAIN / SYNTAX / RUNTIME / DATA / TIMEOUT / ENV / LOGIC`
  and fed back as *targeted* guidance rather than a generic "it failed".
- **Shape checks** — two checks sit in front of the backtest: date sparsity and
  cross-sectional degeneracy. Such factors run fine and emit a valid CSV while
  their IC is identically zero.
- **Leakage prevention** — the split lives in the engine; the repair loop and
  experience memory only ever see in-sample data. Statistical correction does
  not catch look-ahead bias: a deliberately leaky oracle posting a Sharpe of 35
  passes Deflated Sharpe and PBO testing **completely**.
- **Experience memory** — failure modes are persisted and injected as context
  when generating similar factors later, so the same mistake is not repeated.

---

## Why the weight is on evaluation

Generating factors with an LLM is easy. Knowing **which ones to trust** is not.

Here is a factor this pipeline actually produced. Its IC is positive and its
monthly win rate is 66% — under the usual IC/ICIR gates it would be accepted
into a factor library. Its quintile returns:

```
Q1 +5.4%   Q2 +2.6%   Q3 -2.3%   Q4 +7.5%   Q5 -6.3%
                                            ↑ the top quintile is the worst
```

The gains sit in the middle quintiles, so a long-short book built on it misses
exactly the layers that earn. **A significant IC does not mean a factor earns.**
IC is a rank correlation and carries no magnitude, so its divergence from
layered returns is expected — and invisible if IC is all you measure.

The literature turned to this problem through 2026. [2608.27734] tested two
frontier models over 100 candidates with five repeats and reported that
*"honest evaluation certifies passive benchmarks, **rejects every
LLM-discovered strategy**"*.

| | Typical LLM factor miner | AutoAlpha |
|---|---|---|
| Metrics | IC / ICIR | plus layer monotonicity, long-short Sharpe, turnover, net-of-cost return |
| Out-of-sample | often absent | split held by the engine; the search loop **cannot reach** OOS |
| Thresholds | hardcoded constants | derived from cross-section width, adapting to the dataset |
| Deduplication | IC-series correlation | portfolio NAV correlation (the former misses same-family factors) |
| Data constraints | none | availability injected at extraction; inventing proxies is forbidden |
| Semantic review | none | a separate judge agent that rejects, rather than passes, on a parse failure |

---

## Quick start

### Try it without an API key

```bash
git clone https://github.com/EdisonFu/autoalpha.git
cd autoalpha
pip install -e .
python main.py --web --demo
```

Open http://127.0.0.1:8000 and step through a complete run with preloaded
sample results, including the factor above. Nothing is called out to; the data
is clearly marked as a demonstration.

### Run it for real

```bash
pip install -e ".[anthropic]"      # or ".[openai]"

export LLM_API_KEY=...             # your key
export LLM_MODEL=<model-id>        # model id
python main.py --web
```

Data platform SDKs are optional. With none installed the pipeline uses public
data interfaces; configure an account and it prefers your platform. Key, model,
base URL and data sources are all editable in the UI.

---

## Interface

See the [Chinese README](README.md#界面一览) for the full walkthrough with
screenshots of every stage.

![overview](docs/screenshots/00-overview.png)

One sentence in, eight steps out. Every step on the bar carries its own
output: the intent resolves to "A股 · momentum · daily", retrieval returns
3 papers, data prep yields 92 symbols over 752 in-sample days, extraction
produces 2 hypotheses, evaluation passes 3 of 6 through the gates. Click any
step to expand it.

![intent](docs/screenshots/04-step1-intent.png)

**Entries missing an abstract are backfilled across databases.** A sizeable
share of hits arrive with nothing but a title — Crossref misses 59%, OpenAlex
32% — and those get scored on the title alone, so they are almost always
discarded. Working papers suffer most.

The fix is querying another catalogue, not scraping publisher pages: a DOI
Crossref lacks is usually present in OpenAlex, and vice versa. The two were
measured to have zero overlap. Backfill recovers **59% in 4 seconds with no new
dependency**, where scraping the publisher pages recovers 6% — identically for
urllib, curl_cffi and Scrapling, since those URLs are nearly all `doi.org`,
which times out at Cloudflare or lands on a login wall. Fingerprint spoofing
addresses neither failure.

```
            full text   abstract   title only
before         62%        20%         18%
after          66%        29%          5%
```

**Every factor states where its parameters came from.** For arXiv papers the
PDF is downloaded and the methodology sections are fed to the extractor; every
other source keeps its full text behind a publisher paywall, so only the
abstract is available. Measured across three papers, the 7 details needed to
reproduce a factor — lookback window, quantile split, weighting, rebalance
frequency, costs, sample period, metrics — appear **0–1 times in abstracts and
4–6 times in full text**. So a `lookback=20` derived from an abstract is the
model's own choice, not the paper's, and the card says so in amber rather than
letting it pass as a reproduction. Full-text parsing needs
`pip install -e ".[fulltext]"` (pypdf, 404KB).

Each factor then gets a full layered backtest: a long-short cumulative return
curve (Y axis in percent), in-sample and out-of-sample side by side, and
per-quintile annual returns. Factors that fail the IS gate never touch OOS
data, so their OOS column reads "not reached" rather than showing numbers that
look plausible.

![evaluation](docs/screenshots/08-step5-backtest.png)

The second factor in that view, `etf_reversal_overlay`, is the case from the
top of this page: IC is significant at t=2.10, but layer monotonicity is −0.30
and the top quintile performs worst. An IC/ICIR-only pipeline would file it as
a working factor.

---

## Evaluation gates

```
IS   |t| > derived threshold · monotonicity > 0.6 · LS Sharpe > 0.5
     · coverage > 0.7 · turnover < frequency-calibrated cap
OOS  net Sharpe > 0.3 · Sharpe decay > -50% · monotonicity > 0.4
     · NAV correlation < 0.7
```

The turnover cap is calibrated per rebalance frequency (0.60 daily, 0.40
weekly, 0.15 monthly). A fixed 0.30 would make daily factors unpassable and
monthly ones trivial — the gate would measure frequency rather than quality.

Factors that pass can seed an extension round: parameter variants, formula
variants, and orthogonal combinations, filtered for diversity so variants are
not just renamed parameters. Each variant faces the same two-stage gates, and
extension does not guarantee improvement — a combination that raises turnover
can end up worse than its base factor after costs.

Thresholds are derived rather than fixed: `ic_mean_floor = t / sqrt(n_obs) /
sqrt(width)`. A hardcoded 0.02 came from a 24-asset universe; at 92 assets it
is 2.3x too strict, and nothing in the code recorded that coupling.

Expect most factors to fail. That is the gates working — with weak signals,
admission rates of 1–7% are what the literature reports.

---

## Documentation

| File | Contents |
|---|---|
| [CLAUDE.md](CLAUDE.md) | Development guide: invariants and traps worth knowing first |
| [CONTRIBUTING.md](CONTRIBUTING.md) | How to contribute |
| [docs/P0_BACKTEST_DESIGN.md](docs/P0_BACKTEST_DESIGN.md) | Backtest layer design and implementation notes |
| [docs/E2E_REVIEW.md](docs/E2E_REVIEW.md) | End-to-end review and measured findings |
| [docs/TOKEN_COST.md](docs/TOKEN_COST.md) | Measured cost breakdown and levers |
| [README_WEB.md](README_WEB.md) | Web console usage |

## License

MIT — see [LICENSE](LICENSE).
