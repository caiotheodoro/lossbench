"""DeepSeek Harness (dsh) plugin contract: manifest, payloads, and bridge (P2.6)."""

from lossbench.adapters.dsh.plugin import (
    DshPluginBridge,
    ToolDeniedError,
    build_manifest,
    hook_payload,
    manifest_json,
)

__all__ = [
    "DshPluginBridge",
    "ToolDeniedError",
    "build_manifest",
    "hook_payload",
    "manifest_json",
]
