"""The policy decision function: PolicyEngine.apply rules in fixed order."""

from __future__ import annotations

from lossbench.decision import bayes_route
from lossbench.schema import (
    CostProfile,
    DecisionKind,
    DecisionRequest,
    DecisionResponse,
    PolicyBundle,
    Severity,
)


class PolicyEngine:
    """Deterministic decision point applying a PolicyBundle over a CostProfile."""

    def __init__(self, bundle: PolicyBundle, cost_profile: CostProfile) -> None:
        self.bundle = bundle
        self.cost_profile = cost_profile

    def decide(self, request: DecisionRequest) -> DecisionResponse:
        """Apply policy rules in order: deny, allowlist, escalate, route, allow.

        Rule order: 1) DENY if proposed_action['tool'] is in bundle.deny;
        2) DENY if the tool is absent from a non-empty allowlist;
        3) ESCALATE if calibrated_p >= escalation_threshold OR if the
        expected loss of auto-approving exceeds the review cost
        (Bayes guard: p̂·K(σ) > escalate_cost — the severity-aware rule from
        the formal model; the threshold alone is severity-blind and would
        approve a CRITICAL case at low p); 4) ROUTE via bayes_route over
        request.available_models using bundle.model_tiers as model_cost
        (fall back to the first available model when tiers are empty);
        5) ALLOW otherwise. requires_human is True iff ESCALATE.
        expected_loss = calibrated_p * cost_profile.cost(severity) where
        severity comes from request.trajectory_state['severity'] or LOW.
        """
        calibrated_p = request.risk_features.get("calibrated_p", 0.0)
        severity = self._severity(request)
        k = self.cost_profile.cost(severity)
        expected_loss = calibrated_p * k
        tool = request.proposed_action.get("tool")
        if tool in self.bundle.deny:
            return self._respond(
                DecisionKind.DENY,
                request,
                expected_loss=expected_loss,
                rationale=f"denied: tool '{tool}' is on the deny list",
            )
        if self.bundle.allowlist and tool not in self.bundle.allowlist:
            return self._respond(
                DecisionKind.DENY,
                request,
                expected_loss=expected_loss,
                rationale=f"denied: tool '{tool}' is not on the allowlist",
            )
        bayes_guard = expected_loss > self.cost_profile.escalate_cost
        if calibrated_p >= self.bundle.escalation_threshold or bayes_guard:
            reason = (
                "escalated: expected loss exceeds review cost"
                if bayes_guard
                else "escalated: calibrated_p at or above escalation_threshold"
            )
            return self._respond(
                DecisionKind.ESCALATE,
                request,
                requires_human=True,
                expected_loss=expected_loss,
                rationale=reason,
            )
        tiers = {
            model: cost
            for model, cost in self.bundle.model_tiers.items()
            if model in request.available_models
        }
        if tiers:
            p_error = self._risk_by_tier(calibrated_p, tiers)
            best_model, routed_loss = bayes_route(p_error, severity, self.cost_profile, tiers)
            return self._respond(
                DecisionKind.ROUTE,
                request,
                selected_model=best_model,
                expected_loss=round(routed_loss, 4),
                rationale=(
                    f"routed to '{best_model}' by expected loss "
                    "(tiers are risk-scaled costs, not prices)"
                ),
            )
        if request.available_models:
            return self._respond(
                DecisionKind.ROUTE,
                request,
                selected_model=request.available_models[0],
                expected_loss=expected_loss,
                rationale="routed to the first available model",
            )
        return self._respond(
            DecisionKind.ALLOW,
            request,
            expected_loss=expected_loss,
            rationale="allowed: no policy rule applies",
        )

    def _severity(self, request: DecisionRequest) -> Severity:
        raw = request.trajectory_state.get("severity", Severity.LOW.value)
        try:
            return Severity(raw)
        except ValueError:
            return Severity.LOW

    def _risk_by_tier(self, calibrated_p: float, tiers: dict[str, float]) -> dict[str, float]:
        min_cost = min(tiers.values())
        return {
            model: calibrated_p * (min_cost / cost) if cost > 0 else calibrated_p
            for model, cost in tiers.items()
        }

    def _respond(
        self,
        decision: DecisionKind,
        request: DecisionRequest,
        *,
        rationale: str,
        selected_model: str | None = None,
        requires_human: bool = False,
        expected_loss: float | None = None,
    ) -> DecisionResponse:
        return DecisionResponse(
            decision=decision,
            selected_model=selected_model,
            requires_human=requires_human,
            expected_loss=expected_loss,
            rationale=rationale,
            policy_ref=request.policy_ref,
        )
