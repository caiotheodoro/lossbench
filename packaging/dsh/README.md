# lossbench-risk-control — dsh plugin

A DeepSeek Harness (dsh) plugin that places the LossBench policy engine in
front of model and tool calls. This directory ships the Python-side plugin
contract: a manifest template and the payload shapes the thin JS shim (added
in a later packaging step) translates into dsh hook behavior. The contract is
duck-typed and never requires Node in this repository.

## Publish

1. Package the JS shim as an npm package named `lossbench-risk-control` and
   tag it with the `dsh-plugin` topic so dsh's plugin registry discovers it.
2. Keep `plugin.manifest.json` as the manifest template: `id` matches the npm
   package name, `version` is semver, `hooks` lists the hooks the shim
   registers (`beforeModel`, `beforeTool`, `onResolution`), and `bridge`
   points at the Python HTTP endpoint the shim calls back into.
3. The bridge endpoint authenticates with the `bridge.auth_header` header
   (default `x-lossbench-key`) carrying the shared key; requests must finish
   within `bridge.timeout_s` (30 s).

## Wire the hooks

The shim maps each dsh hook to an HTTP POST against `bridge.url` with a
`hook_payload` JSON body:

- `beforeModel` — body `{"hook": "beforeModel", "event": null, "tool_name":
  null, "args": null, "messages": [...]}`. The Python side returns either
  `{"action": "continue", "payload": {"decision": ..., "event": ...}}` (pass
  the call through, optionally honoring the selected model), `{"action":
  "block", "reason": ...}` (deny the call), or `{"action": "escalate",
  "reason": ...}` (route the call to human review before it proceeds).
- `beforeTool` — same envelope with `tool_name` and `args` populated; a block
  means the tool is on the policy deny list or absent from the allowlist, and
  the shim surfaces it as a dsh ToolDeniedError.
- `onResolution` — body `{"hook": "onResolution", "event": null, "tool_name":
  null, "args": null, "messages": null}` plus the resolution in `args`:
  `{"decision_id", "resolution", "reviewer"}` where `resolution` is
  `APPROVE` | `REJECT` | `AMEND` (AMEND carries `amended_action`). The
  response is the recorded DecisionEvent.

## Driving the policy

The bridge is duck-typed: beforeModel `messages` may carry a `lossbench`
metadata dict on any message (`calibrated_p`, `severity`, `trajectory_id`,
`task_id`, `model_id`, `tenant_id`, `action`, `available_models`), and
beforeTool `args` may carry `calibrated_p`, `severity`, and trajectory
context at the top level. Anything the harness does not provide defaults to a
permissive policy stance.
