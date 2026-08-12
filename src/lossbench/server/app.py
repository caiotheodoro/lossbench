"""FastAPI application factory for the multitenant decision HTTP service."""

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from lossbench.ledger.store import AuditLedger
from lossbench.policy import PolicyEngine
from lossbench.schema import DecisionEvent, DecisionRequest, DecisionResponse
from lossbench.server.store import TenantConfig, TenantStore


class _DecideBody(BaseModel):
    tenant_id: str
    request: DecisionRequest


def create_app(store: TenantStore, ledger: AuditLedger | None = None) -> FastAPI:
    """Build the FastAPI app bound to a TenantStore and an optional AuditLedger."""

    app = FastAPI(title="lossbench-server")

    def _validation_error_handler(
        request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        return JSONResponse(status_code=400, content={"detail": "invalid request"})

    app.add_exception_handler(RequestValidationError, _validation_error_handler)

    def _require_tenant(tenant_id: str) -> TenantConfig:
        try:
            return store.get(tenant_id)
        except KeyError:
            raise HTTPException(
                status_code=403, detail=f"unknown tenant: {tenant_id}"
            ) from None

    @app.post("/v1/decide")
    def decide(body: _DecideBody) -> DecisionResponse:
        """Evaluate a DecisionRequest against the tenant's own policy."""
        config = _require_tenant(body.tenant_id)
        if body.request.tenant_id != body.tenant_id:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"tenant mismatch: path/body tenant '{body.tenant_id}' != "
                    f"request tenant '{body.request.tenant_id}'"
                ),
            )
        engine = PolicyEngine(config.policy, config.cost_profile)
        return engine.decide(body.request)

    @app.get("/v1/tenants/{tenant_id}/config")
    def get_config(tenant_id: str) -> dict[str, object]:
        """Return the tenant's active policy surface."""
        config = _require_tenant(tenant_id)
        return {
            "policy_id": config.policy.id,
            "cost_model_id": config.policy.cost_model_id,
            "escalation_threshold": config.policy.escalation_threshold,
            "allowlist": config.policy.allowlist,
        }

    @app.post("/v1/tenants/{tenant_id}/events", status_code=201)
    def post_event(tenant_id: str, event: DecisionEvent) -> dict[str, str]:
        """Append a DecisionEvent to the audit ledger; 501 when no ledger.

        A duplicate event_id returns 409 (idempotent retry semantics on an
        append-only endpoint), not a 500.
        """
        _require_tenant(tenant_id)
        if ledger is None:
            raise HTTPException(status_code=501, detail="no ledger configured")
        stored = event.model_copy(update={"tenant_id": tenant_id})
        try:
            chain_hash = ledger.append(stored)
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from None
        return {"event_id": stored.event_id, "chain_hash": chain_hash}

    @app.get("/v1/health")
    def health() -> dict[str, object]:
        """Return liveness and the registered tenant count."""
        return {"status": "ok", "tenants": len(store)}

    return app
