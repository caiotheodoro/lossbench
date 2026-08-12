"""Tenant configuration registry: per-tenant policy and cost configuration."""

from dataclasses import dataclass

from lossbench.schema import CostProfile, PolicyBundle


@dataclass
class TenantConfig:
    """Policy bundle and cost profile bound to one tenant."""

    tenant_id: str
    policy: PolicyBundle
    cost_profile: CostProfile


class TenantStore:
    """Mapping of tenant_id -> TenantConfig, fixed after registration."""

    def __init__(self) -> None:
        """Create an empty tenant registry."""
        self._configs: dict[str, TenantConfig] = {}

    def register(self, config: TenantConfig) -> None:
        """Register a tenant config; raises ValueError on duplicate tenant_id."""
        if config.tenant_id in self._configs:
            raise ValueError(f"duplicate tenant_id: {config.tenant_id}")
        self._configs[config.tenant_id] = config

    def get(self, tenant_id: str) -> TenantConfig:
        """Return the registered config; raises KeyError when unknown."""
        return self._configs[tenant_id]

    def __len__(self) -> int:
        """Number of registered tenants."""
        return len(self._configs)
