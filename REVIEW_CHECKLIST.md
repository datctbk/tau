# tau Review Checklist

Use this checklist for PR reviews to keep tau aligned with the minimal-core philosophy.

## Core Philosophy Guardrails

- [ ] No feature logic is added to tau core unless it is a reusable primitive.
- [ ] Product-specific behavior is implemented in extension packages (`tau-assistant`, `tau-agents`, `tau-memory`, `tau-web`).
- [ ] New core hooks are generic, stable, and documented.

## Safety and Reliability

- [ ] Policy path behavior is tested (allow/block paths).
- [ ] Audit/event output remains structured and backward-compatible.
- [ ] Error handling and retry/compaction behavior remain deterministic.

## Testing and Validation

- [ ] New behavior has unit/integration tests.
- [ ] Existing tests still pass.
- [ ] Compile/type checks pass for touched modules.

## Diff Discipline

- [ ] Keep core changes minimal and isolated.
- [ ] Avoid unrelated refactors in the same PR.
- [ ] Include short rationale for each core change.
