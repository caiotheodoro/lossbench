# LossBench: Severity-Weighted Expected Loss for Agentic Back-Office Evaluation

## 1. Abstract

Accuracy and token cost are the wrong objectives for production agents that touch money. A misrouted payment, an unreconciled high-value position, or an unescalated settlement exception has a business cost that no accuracy score and no per-token price captures. We present LossBench, an open benchmark and scoring engine that evaluates agents on severity-weighted, trajectory-level expected loss: realized failure cost $K(\sigma)$ charged per outcome error, plus model, judge, human-review, and latency costs, summed over the trajectory. We prove and ship as an executable test the flat-cost theorem (H0): when all severities share one cost, loss ranking reduces to accuracy ranking, making every existing accuracy leaderboard a flat-cost special case. We show the loss ranking diverges from the accuracy ranking as cost asymmetry grows, and we make the cost model a versioned, pluggable input whose contested values are resolved by sensitivity analysis rather than hidden constants.

## 2. Introduction

Production deployments of LLM agents increasingly route between models, reasoning efforts, and human reviewers. The optimization target of the current generation of routing systems is a price-quality frontier: minimize cost at a fixed benchmark quality, or maximize quality at a fixed budget. NVIDIA Switchyard is the clearest example: its published viability formula accounts for judge cost and the price gap between models, but has no term for the consequence of an error. LiteLLM's automatic routing does the same. The implicit assumption is that the error of an agent has the same cost regardless of where it occurs. For chat assistants this is approximately true; for agents that execute payments, mutate ledgers, and sign off on settlement risk it is not. The cost asymmetry between a LOW-severity exception and a CRITICAL one spans orders of magnitude, and it is precisely where agent autonomy is being deployed first.

The second development is that harness infrastructure is commoditizing. Managed-agent products and open harnesses such as DeepSeek Harness demonstrate that middleware placement changes agent quality without changing weights, and that the runtime is increasingly interchangeable. The scarce asset is no longer the runtime; it is the neutral measurement and control layer that works across runtimes. A benchmark that scores trajectories on expected operational loss, and a control plane that makes the same decisions at runtime, can occupy that layer without competing with any harness.

The gap is a measurement gap. No public benchmark scores an agent trajectory against a business cost function that distinguishes severity bands, charges for human review and judge invocations, and verifies the terminal state of the world rather than the final natural-language answer. Accuracy, pass-rate, and token-cost metrics are all severity-blind and state-blind.

**Contributions.** We release:

1. **A metric.** Severity-weighted trajectory-level expected loss $\mathrm{TotalLoss}(\pi,\tau)$ over a versioned, sourced cost profile $K(\sigma)$, including conditional judge cost and human-review cost, with regret measured against explicit oracle baselines.
2. **An executable theorem (H0).** When $K$ is flat, the loss ranking equals the accuracy ranking; the claim ships as a runnable test, not a rhetorical position.
3. **A benchmark.** Three finance back-office domains — reconciliation, payment-exception repair, settlement-risk escalation — with verifier-as-oracle ground truth: no task enters the suite unless an independent verifier agrees with the gold outcome, enforced by code.
4. **A trajectory proper score (TPS).** A strictly proper per-step scoring objective over the prefix-conditioned success process, with the documented limitation that its scalarized ECE companion is resolution-blind.
5. **Outcome-verified pass$^k$ with false-success correction.** Credit is granted only when recorded actions reproduce the gold terminal state; trajectories that claim success without an outcome are penalized as false successes.
6. **Cost-sensitivity analysis.** Curves, ranking stability, and crossover ratios across cost regimes, so every reported conclusion is shown across a range of severity ratios rather than a single contested constant.
7. **An open control plane.** A runtime, CLI, library, and replay simulator using the same trajectory and decision contracts, so the benchmark is not a dead artifact.

## 3. Related Work

**Selective prediction and learning to defer.** Chow formalized the reject option for classifiers as a Bayes tradeoff between error rate and reject rate [Chow, 1970]. Geifman and El-Yaniv extended rejection to deep networks under coverage constraints [Geifman & El-Yaniv, 2017]. Mozannar and Sontag introduced consistent estimators for learning to defer to an expert, treating deferral as a structured action [Mozannar & Sontag, 2020]. HiL-Bench operationalizes this for agents with Ask-F1, the harmonic mean of question precision and blocker recall, and documents a universal judgment gap in help-seeking [Trinh et al., 2026]. Our escalation metrics (escalation precision/recall, missed-high-loss rate, Ask-F1) inherit this line, but embed it in a costed trajectory where deferral itself has a priced review cost and the failure of non-deferral has a severity-weighted cost. Selective prediction optimizes rejection tradeoffs; we price the full operating loop.

**Calibration.** ECE and reliability curves [Guo et al., 2017], temperature scaling, and conformal risk control [Angelopoulos & Bates, 2021; Angelopoulos et al., 2022] are standard tools for making $\hat{p}$ trustworthy. Kuhn et al. show that semantic entropy outperforms token-level uncertainty for LLM sampling [Kuhn et al., 2023]; Tian et al. show that verbalized confidence is miscalibrated unless elicited with self-evaluation [Tian et al., 2023]. Our contribution is not a new calibration method: we assume calibrated risk features as an input contract and measure the *loss consequence* of calibration gaps, reporting trajectory-level ECE and TPS alongside costed outcomes.

**Agent benchmarks.** tau-bench tests tool-agent-user interaction with state-based verification and introduces pass$^k$ for reliability [Yao et al., 2024]; tau$^2$-bench adds a dual-control environment where the user also modifies shared state [Barres et al., 2025]. SWE-bench evaluates real GitHub issue resolution against executable tests [Jimenez et al., 2024]. AgentDojo and AgentHarm measure security properties — prompt-injection robustness and agentic harm — with outcome-oriented grading [Debenedetti et al., 2024; Andriushchenko et al., 2024]. All of these grade the terminal state correctly but score it as a binary success, blind to the business cost of failure. A mislabeled HIGH-severity exception and a mislabeled LOW-severity duplicate cost the same in accuracy terms. LossBench is the first agent benchmark, to our knowledge, to make the cost of each outcome part of the score, with the cost model itself contested and versioned.

**Routing.** RouteLLM learns routers from preference data to trade cost against quality [Ong et al., 2024]. NVIDIA Switchyard's viability formula and LiteLLM auto-routing optimize the same price-quality objective. The cost of the routing *mistake* — the misjudged error that is then executed against a ledger — is absent from all three. WMCC makes a related point for systematic-review screening, showing that accuracy-optimal and cost-weighted rankings disagree on the best LLM in a majority of evaluated studies [Madeyski et al., 2025]; we generalize this observation from document screening to multi-step agent trajectories and make the divergence measurable through crossover ratios.

## 4. Formal Model

Let a trajectory be $\tau = (s_0, a_1, s_1, \ldots, a_T, s_T)$: states and actions over $T$ decision points. Let $\pi$ be an execution policy; at each decision point a policy may select a model, a reasoning effort, a tool action, a verification step, abstention, or human escalation. For each decision point $t$:

- $K(\sigma_t)$: business cost of the applicable failure severity,
- $\hat{p}_t$: calibrated probability of the relevant failure,
- $c^{\mathrm{model}}_t$: inference cost, including reasoning tokens,
- $c^{\mathrm{judge}}_t$: cost of an invoked judge or verifier,
- $c^{\mathrm{human}}_t$: human-review cost,
- $c^{\mathrm{latency}}_t$: optional SLA/latency penalty with weight $\lambda$,
- $e_t \in \{0,1\}$: indicator of an outcome error or unsafe action.

The total loss of a trajectory is:

$$
\mathrm{TotalLoss}(\pi, \tau) = \sum_{t=1}^{T} \Big[
K(\sigma_t)\, e_t
+ c^{\mathrm{model}}_t
+ c^{\mathrm{judge}}_t \cdot \mathbb{1}[\mathrm{judge\ invoked}_t]
+ c^{\mathrm{human}}_t \cdot \mathbb{1}[\mathrm{human\ review}_t]
+ \lambda\, c^{\mathrm{latency}}_t
\Big]
$$

with the convention that only *unreviewed* errors are charged $K(\sigma_t)$: an error caught by an escalated human review is charged the review cost, not the full business loss, and the error indicator for reviewed decisions is recorded as resolved rather than realized. Expected policy loss is taken over the task distribution:

$$
J(\pi) = \mathbb{E}_{\tau}\big[\mathrm{TotalLoss}(\pi, \tau)\big], \qquad
\mathrm{Regret}(\pi) = J(\pi) - J(\pi^*)
$$

where $\pi^*$ is an oracle or hindsight policy used for evaluation only and never available to the live controller. In the benchmark report, regret is reported against at least three baselines: always-frontier (highest capability model on every decision), always-cheap, and the oracle $\pi^*$.

**Bayes decision rules.** The controller's local routing decision is:

$$
\mathrm{route}_t = \arg\min_{m \in \mathcal{M}} \big[\hat{p}_{m}(x_t)\, K(\sigma_t) + \mathrm{price}(m, \mathrm{effort})\big]
$$

Escalation is worthwhile when the expected avoided loss exceeds the incremental cost of review and inference:

$$
\mathrm{escalate}_t \iff
\mathrm{expected\_avoided\_loss}_t
> \Delta c^{\mathrm{model}}_t + c^{\mathrm{judge}}_t + c^{\mathrm{human}}_t
$$

**Reasoning-effort pricing.** Reasoning effort is a first-class policy dimension, not a sampling detail:

$$
\mathrm{price}(m, \mathrm{effort}) =
\mathrm{input\_cost} + \mathrm{output\_cost} + \mathrm{reasoning\_token\_cost}(\mathrm{effort})
$$

A controller may therefore raise reasoning effort from low to high before escalating to a different model; the escalation rule prices the effort increase as part of $\Delta c^{\mathrm{model}}_t$.

**Judge-cost cancellation rule.** Judge cost must not be subtracted unconditionally. If a judge or verifier runs on every request, its cost is common to all policies and cancels in any pairwise comparison. If it runs conditionally, the invocation probability must be included explicitly as $c^{\mathrm{judge}}_t \cdot \mathbb{1}[\cdot]$, exactly as in $\mathrm{TotalLoss}$. Violating this rule silently penalizes verification-heavy policies.

**Flat-cost reduction (H0).** If $K(\sigma) = K_0$ for all $\sigma$, then $\mathrm{TotalLoss}(\pi,\tau) = K_0 \cdot \#\{\text{unreviewed errors}\} + \text{costs common across the comparison}$, and under matched execution costs the loss ranking of any two policies coincides with their error-rate (accuracy) ranking. Accuracy leaderboards are therefore the flat-$K$ special case of loss evaluation. This reduction is the executable content of the design's claim C2.

## 5. Metrics

The benchmark report emits all of the following; no single score is sufficient.

**Loss metrics.** Expected loss per trajectory; normalized expected loss; regret against always-frontier, always-cheap, and oracle policies; and loss at fixed budgets — best achievable loss along the risk-coverage curve at or under a fixed spend, human-review, or latency budget. These translate "which model is best" into "which policy is cheapest to be wrong with at a given operating point."

**Risk and deferral metrics.** Severity-weighted risk-coverage curves (loss against the share of work auto-processed); escalation precision and recall (are escalations correct, and are dangerous cases escalated); missed-high-loss rate (share of HIGH/CRITICAL-severity outcomes neither correctly auto-processed nor escalated); review load and review cost; calibration ECE, Brier score, and reliability curve; and Ask-F1-style escalation quality to prevent escalation spam, following HiL-Bench [Trinh et al., 2026]. Ask-F1's harmonic structure is kept because it is architecturally robust to question spam: an agent that asks constantly collapses its question precision.

**Trajectory Proper Score (TPS).** Let $q_t = P(\text{success} \mid \mathrm{history}_t)$ be the prefix-conditioned success probability. The estimator maps each committed decision (ALLOW, VERIFY, ROUTE, ESCALATE) to $q_t = 1 - \hat{p}_t$ with $\hat{p}_t$ the event's calibrated failure probability, clamped to $[0.01, 0.99]$ so the Brier penalty stays bounded and bins never degenerate; DENY and ABSTAIN emit no success forecast and are scored at the maximum-entropy $q_t = 0.5$, as are events missing a calibrated probability. The trajectory proper score is the sum of per-step Brier penalties:

$$
\mathrm{TPS}(\tau) = \sum_{t=1}^{T} (q_t - y)^2, \qquad y = \mathbb{1}[\text{trajectory success}]
$$

where $y$ is the observed final outcome applied to every step, because the trajectory outcome is only known at the end. The per-step Brier penalty is strictly proper for the binary outcome, so the sum is strictly proper for the $q_t$ process: a well-calibrated forecaster minimizes it in expectation. **Documented limitation:** the companion scalarized trajectory ECE (ten bins over the final forecast $q_T$ only) is resolution-blind — it discards the shape of the probability path, so trajectories with identical $q_T$ but different intermediate risk profiles compare equal; TPS is the full-resolution view and must be read alongside it.

**Outcome-verified pass$^k$ with false-success correction.** Trials are outcome-verified booleans: a success is credited only when the recorded actions reproduce the gold terminal state. pass@$k$ is best-of-$k$ capability (at least one successful trial among the first $k$); pass$^k$ is reliability (all $k$ trials must succeed, so a single failure fails the task). Severity-corrected credit weights each task by $w_i = K(\sigma_i)/K_{\max} \in [0,1]$, so failing a HIGH task forfeits more credit than failing a LOW task. The false-success rate is the share of trajectories that end in an ALLOW or VERIFY claim without a recorded outcome and whose recorded actions fail to reproduce the gold state — the "agent claims done, state unchanged" detector. The rate counts safe-by-construction trajectories (escalated, denied, or outcome-recorded) in the denominator, so it cannot be gamed by refusing to claim.

**Cost-sensitivity.** For each ratio $r \in \{1, 2, 5, 10, 100\}$, the HIGH and CRITICAL severity costs are scaled by $r$ (LOW/MEDIUM unchanged) and per-model losses are recomputed, producing cost-sensitivity curves, per-ratio loss rankings, and crossover ratios: the first ratio at which a pair of models swaps loss ranking, or never. These make C1 measurable — the loss ranking diverges from the accuracy ranking exactly when the curves cross — and they convert the contested-input problem into an empirical one: a conclusion that changes inside the published ratio range is a conclusion about the cost assumption, not the model.

**Integrity metrics.** Unauthorized mutation rate, policy-gate bypass rate, state consistency after tool execution, evidence completeness, and audit-record completeness are reported but not merged into the loss score, keeping integrity concerns orthogonal to cost.

## 6. Benchmark

The v1 suite is finance back-office only. Three domains, all generated from seeds with verifier-as-oracle ground truth.

**Reconciliation.** Ledger-statement pairs classified MATCH or EXCEPTION, with exception classification, severity classification, evidence extraction, and escalation decision. The track carries forward the nine-class exception taxonomy with an explicit severity mapping: HIGH — AMOUNT_MISMATCH, FX_CONVERSION_ERROR, BENEFICIARY_MISMATCH, COUNTERPARTY_MISMATCH; MEDIUM — VALUE_DATE_MISMATCH, MISSING_MESSAGE, PARTIAL_MATCH; LOW — DUPLICATE, FIELD_CORRUPTION. Severity is derived from the exception type by code, not by the model, so misclassification of severity is itself an observable error mode.

**Payment-exception repair.** Failed or returned payment triage, repair proposals, missing/contradictory field handling, duplicate and replay detection, approval-before-mutation, and final ledger-state verification. Signals include fraud-hold exceptions that must route to human review; the domain's cost profile emphasizes principal risk.

**Settlement-risk escalation.** FX and value-date exceptions, counterparty/beneficiary mismatch, delayed or missing settlement, exposure-sensitive severity, and mandatory human review for high-tail-risk actions: HITL is *required* whenever the exposure class is HIGH or CRITICAL, so failing to escalate is a scored failure mode rather than a stylistic choice.

**Verifier-as-oracle.** Every candidate task is submitted to an independent domain verifier that recomputes the gold outcome from the task's initial state alone, without reading the generator's intended label. A task enters the suite only when the verifier agrees with the generated gold outcome; agreement of 100% is enforced by construction of the release pipeline, not asserted post hoc. The verifier is sealed per release revision.

**Seeded determinism.** Same seed, byte-identical suite. Generation, task serialization, canonical JSON, and hashing are deterministic; suites are distributed as frozen revisions, and the response cache is keyed by model, prompt, parameters, seed, and input hash.

**Contamination signatures.** Every task carries a SHA-256 signature. The contamination monitor computes overlap between the benchmark suite and any candidate training/eval corpus; the release certificate requires a train/eval overlap of 0, and leak-injection tests must be detected with recall 1.0. The flagship adapter is never trained on the public test set.

## 7. Falsifiability

The paper's claims are executable tests, shipped with the benchmark.

**H0 — flat-cost theorem.** With $K$ flat, the loss ranking equals the accuracy ranking. This is asserted as an identity over the formal model (§4) and instantiated as a property test over the real scoring code: run any set of error patterns with the `flat.yaml` cost profile and assert the loss ordering is the error-rate ordering. If it ever fails, the implementation contradicts the model; the test is part of CI.

**H1 — metric divergence.** As cost asymmetry grows, the loss ranking diverges from the accuracy ranking. The divergence is measurable via cost-sensitivity curves and crossover ratios (§5): a divergence claim is accepted only when a crossover ratio exists within the published range $r \in [1, 100]$ for some pair of policies, and rejected otherwise. H1 is therefore not a slogan but a quantity with an existence condition.

**Property test — monotonicity of escalation.** For the same calibrated risk ordering, raising $K(\mathrm{HIGH})$ never *lowers* the optimal escalation rate. This is a designed property test over the decision function: increasing the cost of the failure the escalation prevents cannot rationally reduce the set of states worth escalating; the test verifies the implementation respects the ordering across the full ratio grid.

Each falsifiable claim is reported with per-seed deltas, confidence intervals, and variance envelopes, so "the ranking flipped at $r=5$" is reported as a statement about the paired seeds, not a single lucky run.

## 8. Open Questions and Limitations

**Synthetic data only.** The benchmark is generated, not observed. The distributions, error patterns, and severity mixes are designed and seeded; they are representative of back-office workloads by construction and citation, not by sampling. Everything reported on synthetic data inherits this caveat, and the repository says so on every report.

**Severity costs are contested inputs.** No universal cost model exists. LossBench treats $K$ as a versioned, sourced input: an empirical registry (Fed payments statistics, ACH/wire operational costs, fraud-loss multipliers, Basel operational-risk event classes) feeds profiles including `reconciliation.yaml`, `principal_risk.yaml`, and `review_heavy.yaml`, and every conclusion must be shown stable across published sensitivity ranges. The risk of arbitrary-cost criticism is addressed by never publishing a single chosen profile as the answer, and by the crossover-ratio framing: rankings that move within the published range are reported as cost-dependent, not as findings.

**Small-n statistical honesty.** Model runs on seeded suites are expensive; $n$ per arm is small. We report paired per-seed deltas and variance envelopes rather than point estimates with false precision, and we state explicitly where a crossover ratio is within noise.

**Local judge calibration gap.** The severity registry cites institutional sources for operational costs, but the *judge-cost and review-cost* parameters — human review time, conditional judge invocation rates — are the least evidenced entries. They are measured against the same sensitivity discipline as severity costs.

**No production data yet.** The control plane is validated on recorded traces from the benchmark and synthetic workloads; no real tenant ledger has been scored. Runtime claims (C3–C5 in the design) are therefore out of scope of this paper's evidence, and deployment evidence is future work.

## 9. References

- Chow, C. K. On optimum recognition error and reject tradeoff. *IEEE Transactions on Information Theory*, 16(1):41–46, 1970.
- Geifman, Y. and El-Yaniv, R. Selective prediction for deep neural networks. *NeurIPS*, 2017.
- Mozannar, H. and Sontag, D. Consistent estimators for learning to defer to an expert. *ICML*, 2020.
- Kuhn, L., Gal, Y., and Farquhar, S. Uncertainty estimation for language models: An experiment-driven approach (semantic entropy). arXiv:2302.09664, 2023.
- Tian, K., Mitchell, E., Zhou, A., Sharma, A., Bao, A., Yao, H., Finn, C., and Liang, P. Just ask for calibration: Strategies for eliciting calibrated confidence scores from language models fine-tuned with human feedback. arXiv:2305.14975, 2023.
- Yao, S., Shinn, N., Razavi, P., and Narasimhan, K. tau-bench: A benchmark for tool-agent-user interaction in real-world domains. arXiv:2406.12045, 2024.
- Barres, V., Dong, H., Ray, S., Si, X., and Narasimhan, K. tau2-bench: Evaluating conversational agents in a dual-control environment. arXiv:2506.07982, 2025.
- Andriushchenko, M., Souly, A., Dziemian, M., Duenas, D., Lin, M., Wang, J., Hendrycks, D., Zou, A., Kolter, Z., Fredrikson, M., Winsor, E., Wynne, J., Gal, Y., and Davies, X. AgentHarm: A benchmark for measuring harmfulness of LLM agents. arXiv:2410.09024, 2024.
- Debenedetti, E., Zhang, J., Balunovic, M., Beurer-Kellner, L., Fischer, M., and Tramer, F. AgentDojo: A dynamic environment to evaluate prompt injection attacks and defenses for LLM agents. arXiv:2406.13352, 2024.
- Ong, I., Almahairi, A., Wu, V., Chiang, W.-L., Wu, T., Gonzalez, J. E., Kadous, M. W., and Stoica, I. RouteLLM: Learning to route LLMs with preference data. arXiv:2406.18665, 2024.
- Jimenez, C. E., Yang, J., Wettig, A., Yao, S., Pei, K., Press, O., and Narasimhan, K. SWE-bench: Can language models resolve real-world GitHub issues? arXiv:2310.06770, 2024.
- Trinh, T., Elfeki, M., Luo, G., Luu, K., Hunt, N., Hernandez, E., Marwaha, N., He, Y. Y., Wang, C., Carabedo, F., Castillo, A., and Liu, B. HiL-Bench (Human-in-Loop Benchmark): Do agents know when to ask for help? arXiv:2604.09408, 2026.
- Madeyski, L., Kitchenham, B., and Shepperd, M. LLM4SCREENLIT: Recommendations on assessing the performance of large language models for screening literature in systematic reviews (introduces the Weighted Matthews Correlation Coefficient, WMCC). arXiv:2511.12635, 2025; *Information and Software Technology* 198:108204, 2026.
- Guo, C., Pleiss, G., Sun, Y., and Weinberger, K. Q. On calibration of modern neural networks. *ICML*, 2017.
- Angelopoulos, A. N. and Bates, S. A gentle introduction to conformal prediction and distribution-free uncertainty quantification. arXiv:2107.07511, 2021.
- Angelopoulos, A. N., Bates, S., Fisch, A., Lei, L., and Schuster, T. Conformal risk control. arXiv:2208.02814, 2022.
- Board of Governors of the Federal Reserve System. The 2024 Federal Reserve Payments Study. federalreserve.gov. [AUTHORS, YEAR: exact release details not independently verified].
- LexisNexis Risk Solutions. True Cost of Fraud study. [AUTHORS, YEAR: edition and figures not independently verified].
- NVIDIA. Switchyard: model gateway and routing documentation. [AUTHORS, YEAR: product documentation, not an archival citation].
- LiteLLM project. Automatic routing documentation. [AUTHORS, YEAR: product documentation, not an archival citation].

---

Status: Draft v0.1 — companion code: https://github.com/caiotheodoro/lossbench
