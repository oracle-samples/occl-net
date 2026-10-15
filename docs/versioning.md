# OCCL-NET Versioning Strategy

OCCL-NET uses semantic versioning in `major.minor.patch` form. A release version
describes the externally shipped `liboccl-net-occl-ib.so` artifact, its supported
plugin ABI, documented environment variables, and documented behavior.

The current release tooling accepts `OCCL_NET_IB_PLUGIN_VERSION` and defaults to
`0.1.0`. The project should use `1.0.0` for its first explicitly supported
production compatibility baseline.

## Version components

| Change | Meaning | Examples |
|---|---|---|
| `1.x.x` to `2.0.0` | Major release | Removes a supported plugin ABI, changes a documented environment variable incompatibly, changes required deployment/runtime dependencies, changes a public configuration default with material behavior impact, or breaks supported NCCL-runtime compatibility. |
| `x.1.x` to `x.2.0` | Minor release | Adds backward-compatible features: a new supported NCCL range, optional steering/recovery capability, backward-compatible plugin ABI export, opt-in variable, diagnostic stream, or performance feature. |
| `x.x.1` to `x.x.2` | Patch release | Backward-compatible defect, correctness, security, portability, documentation, build, or performance fix with no public-contract change. |

A major version is not required for an internal refactor that leaves all public
contracts and supported behavior intact. A minor release is not appropriate for
a default change that can alter reliability, routing, security, or job outcome
without an operator opt-in.

## Public compatibility contract

The following are versioned contracts:

- the plugin filename and installation/loading instructions;
- exported `ncclNetPlugin_v*` symbols and their compatible NCCL ABI ranges;
- the dynamically visible plugin symbol surface;
- documented environment variables, accepted values, defaults, and semantics;
- platform, architecture, glibc, verbs, and CUDA-runtime compatibility claims;
- observability contracts that users rely on: stable warning categories,
  diagnostic environment variables, and documented machine-consumed output;
- recovery and steering behavior where a default can change failure handling or
  data-path selection.

## Release rules

1. Every release sets `OCCL_NET_IB_PLUGIN_VERSION` at build and verification
   time. The artifact must embed the identical version string.
2. A release notes the supported NCCL versions and exported plugin ABI symbols.
3. Additive ABI support is normally a minor release. Removing an export is a
   major release.
4. Add a new environment variable as opt-in with a safe default in a minor
   release. Reserve a major release for removal or incompatible semantic change.
5. A changed default requires a release note, migration guidance, and an
   isolation/rollback variable. Use a major release when it can change the
   expected correctness or availability contract.
6. Do not reuse a released version. Correct a released artifact with a new patch
   version.
7. Publish a compatibility matrix and checksums for every release artifact.

## Examples

| Proposed change | Version change |
|---|---|
| Fix QP-rejoin retry accounting without changing documented behavior. | `1.4.2` → `1.4.3` |
| Add a disabled-by-default optional telemetry variable. | `1.4.2` → `1.5.0` |
| Add an additional compatible plugin ABI export while retaining existing exports. | `1.4.2` → `1.5.0` |
| Remove support for an exported plugin ABI or a documented NCCL runtime range. | `1.4.2` → `2.0.0` |
| Rename or remove a documented variable without preserving an alias. | `1.4.2` → `2.0.0` |
| Change failover or steering default in a way that changes job behavior for existing users. | `1.4.2` → `2.0.0`, unless the old behavior remains the default or explicitly selectable. |

## Relationship to source-control and package versions

Git commits and build metadata identify provenance; they do not replace the
release version. A release should record the git commit, build platform,
compiler/toolchain, NCCL compatibility range, and SHA-256 checksum alongside
the semantic version.
