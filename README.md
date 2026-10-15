# OCCL-NET Shared-Library Plugin

This guide covers building and running the OCCL-NET network plugin. It leaves
the application's NCCL installation in place and loads `liboccl-net-ib.so` as
the external NET plugin.

## Installation

Choose either a prebuilt Linux binary or a build from source. Both options
produce the same plugin library, `liboccl-net-ib.so`, which you load alongside
your application's existing NCCL installation. Each execution host needs a
working NVIDIA driver, a compatible NCCL runtime, and the RDMA provider and
runtime libraries for its NICs, such as `libibverbs` and `libmlx5` for mlx5
devices. See [Prerequisites](#prerequisites) for NCCL compatibility requirements.

### Option 1: Install a prebuilt binary

Download the v1.1.4 binary that matches your Linux host's architecture
(`uname -m`). The prebuilt binaries do not require the source-build toolchain.

For **x86-64 / amd64** (`uname -m` reports `x86_64`):

```bash
curl --fail --location --output liboccl-net-ib.so \
  'https://objectstorage.us-phoenix-1.oraclecloud.com/n/ax8lpwectatc/b/occl/o/net-plugin%2Famd64%2F1.1.4%2Fliboccl-net-ib.so'
```

For **ARM64 / aarch64** (`uname -m` reports `aarch64`):

```bash
curl --fail --location --output liboccl-net-ib.so \
  'https://objectstorage.us-phoenix-1.oraclecloud.com/n/ax8lpwectatc/b/occl/o/net-plugin%2Faarch64%2F1.1.4%2Fliboccl-net-ib.so'
```

Continue with [Verify and use the plugin](#verify-and-use-the-plugin).

### Option 2: Build from source

Build on a Linux GPU host with a working NVIDIA driver, CUDA Toolkit, and
InfiniBand or RoCE stack. Install CMake 3.22.1 or newer, GCC or Clang, Python 3,
the CUDA Toolkit headers, and the RDMA development headers (including
`infiniband/verbs.h`, typically supplied by `libibverbs-dev` or
`rdma-core-devel`). The plugin uses CUDA headers during compilation but does
not link the CUDA runtime. The default build does not require the GoogleTest
submodule.

From the repository root, configure and build the standalone shared library:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --target occl-net-ib-plugin -j "$(nproc)"
```

If CMake cannot locate a nonstandard CUDA installation, add
`-DCUDAToolkit_ROOT=/path/to/cuda` to the configure command. The build produces
`build/lib/liboccl-net-ib.so`. `BUILD_IB_TRANSPORT_PLUGIN` is on by default;
keep it enabled for this target. The default `RDMA_CORE=OFF` and `MLX5DV=OFF`
settings load the verbs and mlx5 libraries at runtime. Set either option to
`ON` at configure time only when you want direct linkage to the corresponding
installed development library.

Use the built library directly by setting its absolute path:

```bash
export NCCL_NET_PLUGIN="$(pwd)/build/lib/liboccl-net-ib.so"
export NCCL_NET=IB
```

### Verify and use the plugin

After setting `NCCL_NET_PLUGIN` using either option above, check that the
file and NCCL NET-plugin entry points exist:

```bash
test -f "$NCCL_NET_PLUGIN"
readelf -Ws "$NCCL_NET_PLUGIN" | grep 'ncclNetPlugin_v'
ldd "$NCCL_NET_PLUGIN"
```

`readelf` should list `ncclNetPlugin_v6` through `ncclNetPlugin_v12`. Resolve
any dependencies reported as `not found` by `ldd` on each execution host.

Make the same compatible plugin build available on every participating host
and set `NCCL_NET_PLUGIN` and `NCCL_NET` on every rank. Follow [Enable the plugin](#enable-the-plugin) for
launcher examples and [Confirm that it loaded](#confirm-that-it-loaded) to
verify that NCCL selected the plugin.

## Prerequisites

Each participating host needs the same compatible plugin build, an NCCL runtime that supports the exported NET-plugin ABI, and access to the InfiniBand or RoCE devices selected for the job. The plugin exports ABI versions 6 through 12.
Use an absolute path to the library. Ensure the process can load the plugin and its runtime dependencies; for example, install it in a system library path or add its directory to LD_LIBRARY_PATH.
The library embeds an occl-net-ib version <version> string. Release tooling sets it with OCCL_NET_IB_PLUGIN_VERSION which follows the Major.Minor.Patch format.
Record that version with every production deployment and incident report.
NCCL 2.20.3 for Basic port/link failover/recovery
NCCL 2.24.3 for Full NIC failover/recovery

## Check the plugin version

To extract the embedded version from a downloaded plugin, run:

```bash
readelf -p .occl_net_ib_version "/path/to/liboccl-net-ib.so"
```

If `readelf` is unavailable, use `strings`:

```bash
strings "/path/to/liboccl-net-ib.so" | grep -F 'occl-net-ib version '
```

Replace the example path with the plugin library you will load. Look for
`occl-net-ib version <version>` in the output. This is the OCCL-NET plugin
version, not the application's NCCL runtime version.

## Enable the plugin

Set `NCCL_NET_PLUGIN` to the complete path of the shared object. Use
`NCCL_NET=IB` when the job must select the IB network transport explicitly.

```bash
export NCCL_NET_PLUGIN=/opt/occl-net/lib/liboccl-net-ib.so
export NCCL_NET=IB

mpirun -np 16 ./your_application
```

For a launcher that propagates environment variables explicitly:

```bash
mpirun -np 16 \
  -x NCCL_NET_PLUGIN=/opt/occl-net/lib/liboccl-net-ib.so \
  -x NCCL_NET=IB \
  ./your_application
```

Set the same values on every rank. Do not mix plugin builds or incompatible
NCCL runtimes within one job.

## Confirm that it loaded

Enable NET and INIT information for one diagnostic run:

```bash
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=INIT,NET
```

The NCCL host runtime, not the shared object by itself, prints the authoritative
plugin-load records. With the settings above, expect these records, with the
configured library path substituted for `<path>`:

```text
NCCL_NET_PLUGIN set by environment to <path>
Successfully loaded external network plugin <path>
Initialized NET plugin IB
Assigned NET plugin IB to comm
```

The second record proves that the runtime opened the library and found a
supported exported NET-plugin ABI. `Initialized` proves that plugin
initialization and device enumeration succeeded; `Assigned` proves that the
`IB` plugin was selected for that communicator. The plugin then prints one or
more device records in this form:

```text
NET/IB: [<index>] <device>:<port>/<link-layer> provider=<provider>
  speed=<speed> context=<pointer> pciPath=<path> ar=<0|1>
```

Treat all four runtime records plus at least one `NET/IB` device record as a
successful load-and-selection confirmation. A useful minimum capture is
stdout/stderr from every rank plus the plugin path, NCCL version, host name,
and selected NIC list.

## Trace build

For per-request and completion tracing, configure a separate build with
`-DTRACE=ON` and build the same target:

```bash
cmake -S . -B build-trace -DCMAKE_BUILD_TYPE=Release -DTRACE=ON
cmake --build build-trace --target occl-net-ib-plugin -j "$(nproc)"
export NCCL_NET_PLUGIN="$(pwd)/build-trace/lib/liboccl-net-ib.so"
```

Then use:

```bash
export NCCL_DEBUG=TRACE
export NCCL_DEBUG_SUBSYS=NET
```

`NCCL_DEBUG=TRACE` alone does not restore normal `TRACE(...)` statements if the
plugin was built without `ENABLE_TRACE`.

## Runtime configuration

The following variables are implemented by OCCL-NET's portable NET/IB code.
Start with defaults unless a workload or troubleshooting procedure calls for a
change.

| Variable | Default | Purpose |
|---|---:|---|
| `NCCL_IB_HCA` | unset | Include or exclude the IB HCAs and ports visible to the job. Use it to constrain runs to known-good NICs. |
| `NCCL_IB_MERGE_NICS` | `0` | Merge compatible NICs into virtual devices when enabled. The default keeps NICs as separate devices. |
| `NCCL_IB_MERGE_VFS` | `1` | Merge compatible virtual functions. |
| `NCCL_IB_QPS_PER_CONNECTION` | `1` | Number of QPs per connection. |
| `NCCL_IB_SPLIT_DATA_ON_QPS` | `0` | Split data across QPs. |
| `NCCL_IB_ADAPTIVE_ROUTING` | auto | Override IB adaptive-routing selection. `0` disables it; `1` enables it where supported. |
| `NCCL_IB_AR_THRESHOLD` | `8192` | Message-size threshold for adaptive-routing behavior. |
| `NCCL_IB_TIMEOUT` | `20` | Verbs timeout used for QP connection configuration. |
| `NCCL_IB_RETRY_CNT` | `7` | Verbs retry count used for QP connection configuration. |
| `NCCL_IB_PKEY` | `0` | InfiniBand partition key. |
| `NCCL_IB_SL` | `-1` | Service-level override; otherwise traffic class/default policy applies. |
| `NCCL_IB_TC` | `-1` | Traffic-class override. |
| `NCCL_IB_FIFO_TC` | `-1` | FIFO traffic-class override. |
| `NCCL_IB_GID_INDEX` | `-1` | Force a GID index when automatic GID selection is unsuitable. |
| `NCCL_IB_ROCE_VERSION_NUM` | `2` | Requested RoCE version. |
| `NCCL_IB_ADDR_FAMILY` | unset | Limit bootstrap-address selection to an address family. |
| `NCCL_IB_ADDR_RANGE` | unset | Limit bootstrap-address selection to an IP range. |
| `NCCL_IB_ECE_ENABLE` | `1` | Enable ECE negotiation when available. |
| `NCCL_IB_PREPOST_RECEIVE_WORK_REQUESTS` | `0` | Pre-post receive WQEs. Resiliency enables required pre-posting automatically. |
| `NCCL_IB_RETURN_ASYNC_EVENTS` | `1` | Return fatal IB async events to the communicator. |
| `NCCL_IB_RESILIENCY_PORT_FAILOVER` | `1` | Enable QP/device failover and QP rejoin. |
| `NCCL_IB_RESILIENCY_PORT_FAILOVER_MAX_ATTEMPTS` | `1` | Maximum failed-request probe attempts. |
| `NCCL_IB_RESILIENCY_PORT_FAILOVER_PROBE_DELAY` | `1000` ms | Delay before posting a failover probe. |
| `OCI_FRM_ENABLE` | `1` | Enable OCCL-NET CIM/FRM congestion-aware steering. |
| `FRM_SPREAD_EPS_US` | `4` | FRM latency-spread threshold in microseconds. |
| `FRM_ZERO_STAMP_HOLDOFF` | `2` | FRM holdoff for zero latency stamps. |
| `NCCL_IB_WARN_RAIL_LOCAL` | `0` | Emit an INFO warning for local/remote rail mismatch. |

Values are process environment values. Prefer a job launcher, scheduler
environment, or a controlled NCCL configuration file over shell profiles so a
run is reproducible.

## Recommended starting profiles

Default resilient/steered deployment:

```bash
export NCCL_NET_PLUGIN=/opt/occl-net/lib/liboccl-net-ib.so
export NCCL_NET=IB
export NCCL_IB_RESILIENCY_PORT_FAILOVER=1
export OCI_FRM_ENABLE=1
```

Simple compatibility profile with OCCL-NET still loaded:

```bash
export NCCL_NET_PLUGIN=/opt/occl-net/lib/liboccl-net-ib.so
export NCCL_NET=IB
export OCI_FRM_ENABLE=0
export NCCL_IB_RESILIENCY_PORT_FAILOVER=0
export NCCL_IB_QPS_PER_CONNECTION=1
export NCCL_IB_SPLIT_DATA_ON_QPS=0
export NCCL_IB_MERGE_NICS=0
unset NCCL_NET_FORCE_MERGE
```

See [troubleshooting.md](troubleshooting.md) for an ordered isolation plan.

## Full NIC Backup

Full NIC Backup uses `NCCL_NET_FORCE_MERGE` to place multiple physical devices
in a backup group. OCCL-NET exposes GPU-local aliases for normal traffic and
keeps the group's other members as warm standby paths. With
`NCCL_IB_RESILIENCY_PORT_FAILOVER=1`, failed requests can move to those paths;
the existing probe/rejoin mechanism can return traffic to a recovered primary.

Use NCCL 2.24.3 or newer and the same updated OCCL-NET build on every rank.
NCCL 2.24.3–2.29 supports up to four members per group through NET ABI v9–v11;
NCCL 2.30+ supports up to eight through v12. A backup group needs at least two
members, all using the same link layer (InfiniBand or RoCE). Choose members
with working peer connectivity and include a separate physical NIC so a
whole-NIC failure leaves a usable path.

For a four-device group, start with:

```bash
export NCCL_NET_PLUGIN=/opt/occl-net/lib/liboccl-net-ib.so
export NCCL_NET=IB
export NCCL_NET_FORCE_MERGE="mlx5_0,mlx5_1,mlx5_2,mlx5_3"
export NCCL_IB_RESILIENCY_PORT_FAILOVER=1
export NCCL_IB_QPS_PER_CONNECTION=1
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=INIT,NET
```

Replace the example names with devices visible on each host and propagate the
configuration to every rank. `NCCL_IB_HCA`, if set, must allow the intended
members. 

The host runtime parses comma-separated members and semicolon-separated groups,
for example `NCCL_NET_FORCE_MERGE="mlx5_0,mlx5_1;mlx5_2,mlx5_3"` creates two
separate backup groups. Keep the quotes so the shell does not interpret the
semicolon. Devices omitted from the list can still be selected or automatically
merged; the variable is not a device filter. See NVIDIA's
[NIC Fusion syntax](https://developer.nvidia.com/blog/networking-reliability-and-observability-at-scale-with-nccl-2-24/).
Avoid duplicate members and overlapping groups.


The supported failure model is sequential/one-sided N-1: a group can retain a
usable path as members fail, provided the corresponding peer paths remain
reachable. Arbitrary simultaneous failures leaving unrelated survivors at
opposite endpoints are outside this contract. Standby paths consume QP and
registration resources; cross-NUMA backup traffic can reduce performance.
In other words, Full NIC Backup is designed to handle successive device 
failures on one host, down to one surviving device in the group, while the 
peer’s corresponding network paths remain available. Recovery is not guaranteed 
when both hosts lose devices and their surviving devices lack a configured path 
between them.

## Documentation

This README documents the standalone build, plugin loading, runtime settings,
and failover behavior. Contributors should also read the
[contribution guide](CONTRIBUTING.md).

## Examples

See [Enable the plugin](#enable-the-plugin) for a launch example and
[Recommended starting profiles](#recommended-starting-profiles) for example
runtime configurations.

## Help

For non-security questions or bug reports, open an issue in this repository.
For security vulnerabilities, use the process in [SECURITY.md](SECURITY.md).

## Contributing

Contributions are welcome. Before submitting a pull request, review the
[contribution guide](CONTRIBUTING.md), including the Oracle Contributor
Agreement requirement.

## Security

See [SECURITY.md](SECURITY.md) for responsible vulnerability disclosure.

## License

Copyright (c) 2026 Oracle and/or its affiliates.

OCCL-NET original code and modifications are licensed under the [Universal
Permissive License (UPL), Version 1.0](LICENSE.txt).

This project includes code derived from NVIDIA NCCL, Copyright (c) 2015-2026
NVIDIA CORPORATION & AFFILIATES. NVIDIA-origin portions retain their original
copyright and license notices and remain subject to the
[Apache License, Version 2.0](third_party/licenses/Apache-2.0.txt). The
[retained NCCL license text](third_party/licenses/NCCL-LICENSE.txt) also
contains the BSD terms applicable to some upstream portions. Other third-party
files retain their own license notices.
