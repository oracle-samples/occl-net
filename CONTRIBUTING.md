# Contributing to OCCL-NET

We welcome contributions. For bugs or feature requests, open a GitHub issue.
Report security vulnerabilities through the process in [SECURITY.md](SECURITY.md)
instead of opening an issue.

## Oracle Contributor Agreement

Before submitting a pull request, sign the
[Oracle Contributor Agreement](https://oca.opensource.oracle.com). Each commit
must include a sign-off line using the name and email address used for the
agreement:

```text
Signed-off-by: Your Name <you@example.org>
```

Add it when committing with `git commit --signoff` (or `git commit -s`). Only
pull requests from contributors whose agreement can be verified can be accepted.

## OCCL-NET coding standard

OCCL-NET follows the upstream NCCL code style and transport conventions unless
an OCCL-NET-specific requirement is documented here. Existing NCCL compatibility
interfaces and ABI names are preserved where required; new OCCL-NET code must
not introduce `NCCL` into new function or variable names.

## Required conventions

### Files, licensing, and scope

- Every new source, header, script, and Markdown file begins with an Oracle
  copyright and UPL notice, using the file's comment syntax.
- Preserve upstream notices and add Oracle and/or its affiliates modification
  notices where a file is derived from upstream code.
- Keep the network-plugin ABI layer narrow. Transport behavior belongs in
  `src/transport/net_ib`; `src/plugin/net` should contain ABI adaptation only.
- New public behavior requires user documentation and tests.

### C and C++ style

- Follow the surrounding NCCL formatting: two-space indentation, braces on the
  same line as control statements/functions, and concise line wrapping.
- Use `struct nccl...`/existing compatibility types only when interacting with
  inherited APIs. Use clear OCCL-NET names for new functions and variables.
- Prefer `nullptr` in C++ code; use `NULL` only where surrounding C-compatible
  interface style requires it.
- Initialize objects explicitly and keep ownership and cleanup paths visible.
- Do not silently ignore return values that can report transport, allocation,
  or verbs failures.

### Naming conventions

Follow the established NCCL naming shape, but distinguish OCCL-NET-owned code
from inherited compatibility code.

| Kind | NCCL compatibility/inherited code | New OCCL-NET code |
|---|---|---|
| Exported plugin ABI, upstream callback, or upstream structure | Preserve the exact required `nccl...`, `NCCL_...`, or `ncclNet...` spelling. | Do not rename; ABI and source compatibility take precedence. |
| Internal function | Preserve existing upstream `ncclCamelCase` names when modifying that upstream implementation. | Use `occlCamelCase`, for example `occlIbResiliencyTryRejoinQp`. |
| Internal type | Preserve an upstream `struct nccl...` type where it is part of the inherited transport model. | Use `struct occl...` or `occl..._t` consistently, for example `struct occlFrmPlannerState`. |
| Local variable | Preserve existing names in a small upstream-compatible edit. | Use lower camel case without a vendor/trademark prefix, for example `rejoinEpoch`, `activeQpCount`, or `probeDeadlineNs`. |
| Internal constant/macro | Preserve required NCCL compatibility macros. | Use `OCCL_` upper snake case, for example `OCCL_FRM_MAX_REJOIN_ATTEMPTS`. Prefer `constexpr`/typed constants in C++ where practical. |
| New environment variable | Preserve documented `NCCL_*` compatibility variables. | Use `OCCL_NET_` upper snake case, for example `OCCL_NET_FRM_SPREAD_EPS_US`. |
| File | Retain upstream file names where they implement inherited code. | Use lower snake case and an OCCL-specific noun when creating a new module, for example `occl_frm_planner.cc`. |

Do not introduce the `NCCL` trademark in new functions or variables. A new
`nccl...` identifier is allowed only when it implements a required upstream ABI
or must exactly match an inherited compatibility interface. Record that reason
in the review description.

For a large feature added to an inherited `ncclIb...` file, place OCCL-NET
helpers behind an `occl...` boundary. The inherited function can call the new
helper, but new state-machine, steering, health, and telemetry logic should not
extend the `nccl...` namespace.

### Error handling and logging

- Use `NCCLCHECK`, `NCCLCHECKGOTO`, and a single cleanup label for fallible
  transport setup paths.
- Use `WARN(...)` for errors requiring operator attention or an error return.
- Use `INFO(NCCL_NET, ...)` for one-time lifecycle/configuration/state changes.
- Use `TRACE(NCCL_NET, ...)` for high-frequency request, QP, completion, and
  replay detail. Keep it useful for correlating `comm`, request ID, device,
  QP number, WR ID, and status.
- Do not use raw `printf`/`fprintf` in the normal transport path. Diagnostic
  output must be behind a documented compile-time or runtime gate and must not
  become a production data-path dependency.
- Do not log secret material, user payload, or unbounded high-frequency events
  at INFO or WARN.

### Concurrency and resiliency

- Document the owner, lock/atomic discipline, and lifetime of every shared
  recovery, async-event, or telemetry field.
- Use atomics with an explicit memory order where concurrent state crosses
  threads or device/proxy progress paths.
- Make recovery state transitions explicit and idempotent. Late CQEs and
  repeated async events must be safely recognized rather than treated as new
  failures.
- Preserve the separation between active failover routing and rejoined QPs;
  a QP returns to traffic only after local and peer readiness are established.

### Configuration, ABI, and tests

- Define configuration parameters through `NCCL_PARAM` and document their
  default, valid range, interaction, and rollback behavior.
- Treat `NCCL_*` names as compatibility-facing configuration. New
  OCCL-NET-only settings should use an OCCL-NET namespace unless an upstream
  compatibility reason requires otherwise.
- Preserve plugin ABI wrappers and symbol-version exports. Any export change
  requires compatibility testing and a versioning review.
- Add focused unit tests for every error/recovery branch and regression tests
  for multi-QP, peer-ready, and stale-CQE behavior.

## OCCL-NET conformance backlog

The following changes would bring the current tree closer to these standards.
They are recommendations, not behavior changes made by this document.

| Priority | Change needed | Rationale |
|---|---|---|
| High | Replace or formalize CIM/FRM raw stdout output. | `cim.cc` and `p2p.cc` use direct stdout diagnostics; route normal diagnostics through NCCL logging or retain them only behind documented debug gates. |
| High | Make `CIM_DEBUG` a build-system option instead of a source-local define. | A source-local `#define CIM_DEBUG 0` makes release/debug behavior less discoverable and harder to reproduce. |
| Medium | Make FRM debug controls explicit CMake options and record them in build metadata. | `FRM_DEBUG` and `FRM_LAT_PRINT_MS` materially affect observability and output volume. |
| Medium | Rename new internal CIM/FRM helpers, types, and globals to `occl...`/`OCCL_...` names. | Existing compatibility symbols remain necessary; new internal code needs a clear ownership boundary. |
| Medium | Standardize logging vocabulary and field order for failure, failover, and rejoin messages. | Stable `comm`, `devIndex`, `qp_num`, request ID, WR ID, status, and epoch fields improve automated incident analysis. |
| Medium | Add an automated source check for Oracle UPL notices and disallowed new identifier names. | This turns mandatory repository guardrails into enforceable review checks. |
| Low | Document direct diagnostic output format and retention expectations. | CIM/FRM stdout snapshots can be consumed by tools and should be deliberately versioned if supported. |

## Review checklist

Before submitting a change, confirm:

- new files have the Oracle UPL notice and required upstream notices;
- new functions/variables avoid the `NCCL` trademark unless they implement a
  required compatibility interface;
- error paths clean up every initialized resource;
- INFO and TRACE volume is appropriate for the path frequency;
- a feature flag has a safe default and documented rollback;
- relevant source/unit tests and the standalone plugin build pass;
- public ABI, environment-variable, and versioning implications were reviewed.
- Doxygen all functions and APIs
