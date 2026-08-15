# The metric nobody owns: expected loss for agents that touch money

Last year I fine-tuned a 1.7B model on a laptop to run financial reconciliation, and wrote down why the scoreboard everyone was using was the wrong one. That post ended with a claim I couldn't yet prove at system scale: severity-weighted expected loss, not accuracy and not token price, is the objective that matters for agents that touch money. LossBench is the attempt to make that claim measurable, auditable, and usable. This post is the launch writeup: what we built, the numbers we're anchoring to, and the parts that are still honestly open.

## The hook: Switchyard's formula has no error-cost term

NVIDIA's Switchyard benchmark is the cleanest current example of what the routing industry optimizes. On their flagship demo workload, 93% of calls get routed to a 30B open model — which handles 10.4% of the spend — and the whole scheme costs 74% less for a 6-point accuracy hit against the frontier model. The numbers are good. The framing is where the problem lives.

Switchyard's viability formula accounts for judge cost and for the price gap between models. It never accounts for the consequence of a wrong answer. Read the formula again and you'll find what's missing: a `K`, a cost-of-being-wrong term, multiplied by the probability of an error. Their scoreboard answers "how cheaply can I keep accuracy acceptable." For an agent that routes a wire transfer, mints a settlement entry, or files a repair against a live ledger, the question the operator actually asks is "how much money does my agent's mistake cost me, after review?" Those are different questions, and when errors have asymmetric costs, they rank policies differently.

That's the gap LossBench exists for. Accuracy and token price are real quantities, but for money-touching agents they are inputs to a cost equation, not the equation itself.

## The thesis: expected loss is the objective nobody owns

Here's the full claim, the one we're building the public benchmark around:

> A production agent should estimate the cost of being wrong, choose an execution strategy, stop before irreversible harm, escalate when appropriate, and produce evidence that the policy worked.

Concretely, expected loss per trajectory, with three adjectives, each of which is a design decision:

- **Severity-weighted.** A missed amount mismatch on a $2M wire is principal at risk. A duplicate is a rebook at worst. Scoring both as one "error" is not a measurement, it's a choice to ignore the difference that determines the business outcome.
- **Calibrated.** The agent's `p̂` must be a probability, not a vibe. Escalation decisions only make sense with calibrated risk.
- **Trajectory-level.** The unit of measurement is the whole trajectory — classify, verify, propose, mutate, verify again — and its final state, not the last natural-language answer.

Now the honest special case, because we want to be falsified if it doesn't hold. When all severities have equal cost — flat K — expected-loss ranking reduces exactly to accuracy ranking. Your current leaderboard is the flat-cost special case of our metric. We don't argue with it; we ship it as an executable test (`test_flat_cost_theorem.py`), so if the mathematics is wrong, the test fails and we fix it.

## What we built

The benchmark covers three finance back-office domains: reconciliation (carrying forward ReconForge's nine-class exception taxonomy and severity scheme), payment exception repair, and settlement-risk escalation. Every task is generated from a seed, verified, and scored. The properties we require of the benchmark itself are enforced by code:

- **Verifier-as-oracle.** An independent verifier recomputes the gold outcome from the task state alone, never reading the label the generator attached. No task is accepted unless verifier and generator agree: 100% agreement, by construction, with a self-check loop that throws out and regenerates disagreeing candidates.
- **Seeded determinism.** Same seed, byte-identical task suite, every time. The honest variance in results comes from the sampling marginal at inference, not from the benchmark.
- **Contamination monitoring.** Every task gets a SHA-256 signature over its sorted field values. The monitor detects injected leaks at 1.0 at every leak fraction from 5% to 50% and never false-fires on a clean set.
- **Outcome-verified pass^k with false-success correction.** A trajectory counts as successful only if the final ledger state is what the verifier says it should be. An agent that says "fixed it" and leaves the state wrong is not a success — it's the worst kind of error, because it looks like a win. We track that rate explicitly.
- **Trajectory Proper Score.** A scoring implementation that is proper over trajectories: you can't game it by hedging, and calibration is scored, not assumed.
- **Cost-sensitivity analysis.** Every report shows how rankings change as severity cost ratios change. Where the ordering flips as K ratios move, that flip is a finding, not an edge case to bury.

## The control plane: record, calibrate, decide, escalate, replay

The benchmark is the measurement; the control plane is what turns the measurement into operating decisions. It's a thin layer over any harness — CLI, Python middleware, OpenAI-compatible proxy, OpenTelemetry recorder — consuming and producing one stable event contract.

The loop is: **record** every decision as an append-only, hash-chained ledger event; **calibrate** risk features against resolved outcomes; **decide** with a Bayes-optimal rule — route to the model minimizing `p̂·K + price`, escalate when expected avoided loss exceeds review cost; **escalate** through durable review workflows with SLAs; and **replay** policies offline.

The flagship demo is the replay step: re-run last month's workload under a different risk policy. Flip a threshold, replay the recorded events deterministically — zero LLM calls, because the decisions were already made and recorded — and read the counterfactual: total cost before and after, review load before and after, per-case differences. That turns "should we tighten the escalation threshold?" from a debate into a report. And because the ledger is hash-chained, the record you replayed is verifiably the record that happened; `verify()` recomputes the chain and detects tampering.

One design rule keeps this from becoming marketing: cost models are sourced, versioned, pluggable inputs, and conclusions are always shown across a K range, never at one K. `K` is a versioned input to the benchmark, not a hidden constant. The report tells you where the answer holds and where it changes.

## The empirical anchors

None of this is usable without numbers, and numbers are the contested part, so the anchors are published in an open registry with sources and confidence levels. Ten entries so far, order-of-magnitude:

- **Misrouted wire:** typical $5M, range $1M–10M — the principal exposure when a Fedwire transfer lands in the wrong account (Fedwire service overview).
- **Misposted ACH credit:** typical $3,881 — the average ACH credit value from the Federal Reserve Payments Study 2024, with the range spanning small retail to corporate postings.
- **Fraud multiplier:** $3.75 of total cost per dollar of direct loss, range 2.5–5.0 (LexisNexis True Cost of Fraud).
- **AML false-positive review:** $3–10 per analyst review — in a regime where more than 90% of AML alerts are false positives (LexisNexis). That number is why calibration and escalation precision are first-class metrics: review load is real money.
- **Missed SAR:** $1M–100M regulatory penalty exposure (FinCEN enforcement actions).
- **Settlement failure:** up to the $1B high bound, reflecting large-value FX turnover (BIS).

The caveat is deliberate: these are public, order-of-magnitude anchors, not actuarial values. The registry says so, entry by entry, and invites you to contest them with your own ledger. The benchmark ships multiple cost profiles — flat, reconciliation, principal-risk, review-heavy — and your own YAML drops in.

## What's still open

Keeping the negative results, as before:

- **All data is synthetic.** There is no production financial data in the benchmark. The methodology is the subject; the claim is that the measurement is defensible, not that synthetic severity equals real financial loss. That equivalence is explicitly out of scope and we will not pretend otherwise.
- **Severity costs are contested inputs.** The registry anchors are public and sourced, but "the cost of a misposted ACH" is not a physical constant. This is why every conclusion is shown across a K range.
- **The local judge calibration gap carries forward.** In ReconForge, a rubric fix took the DeepSeek judge from kappa 0.74 to 0.90 while the local fine-tuned judge regressed from 0.74 to 0.37 — extra rules pushed it off its training distribution. That gap stays open until a judge-specific fine-tune exists. LossBench inherits it.
- **No production deployment data yet.** The replay lab runs on recorded traces from our own harnesses and the control plane's adapters. The 12-month goal — a finance operator or fintech team running the control plane on real workload — is a milestone, not an achievement.
- **Small-n statistics.** Where runs use few seeds or few tasks, the reports say so and pair per-seed deltas with confidence intervals instead of hiding variance behind a single number.

## What's next, and how to run it

Run it yourself, no account, no cloud:

```sh
make install && make validate
```

That gives you the full benchmark with the theorem tests green. The control-plane path is one command:

```sh
lossbench simulate --trace traces.parquet --policy policy-v2.yaml
```

flip a threshold in the policy YAML, and read the counterfactual cost and review load.

The twelve-week plan has the release artifacts staged: the Hugging Face dataset with a machine-readable eval configuration and contamination certificate, the leaderboard Space, the interactive control-plane demo — flip a threshold slider over a recorded month of workload and watch the counterfactual loss move — model baselines including the 1.7B continuity run and a Qwen3.8-27B finance adapter, the arXiv paper on trajectory-level severity-weighted loss, and a Switchyard issue proposing a loss-aware evaluation term.

I don't expect anyone to adopt our specific K values. I expect the point that K belongs in the formula to become hard to argue with. Nobody owns the metric yet. That's why we're publishing it.

## Links

- Repository: `https://github.com/your-org/regretbench` (placeholder — repo URL)
- Architecture and design docs: `https://github.com/your-org/regretbench/blob/main/docs/ARCHITECTURE.md`
- Hugging Face dataset + leaderboard: `https://huggingface.co/datasets/your-org/lossbench` (placeholder — to be published)
- Interactive demo Space: `https://huggingface.co/spaces/your-org/lossbench-control-demo` (placeholder — to be published)
- arXiv draft: `https://arxiv.org/abs/xxxx.xxxxx` (placeholder — to be published)
- Previous post, ReconForge: `https://huggingface.co/blog/your-org/reconforge` (placeholder)
