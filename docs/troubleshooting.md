# OCCL-NET Troubleshooting Guide

Use this guide to isolate a problem without changing multiple variables at
once. Apply one step to every rank, rerun the smallest reproducer, and record
the exact result before moving to the next step.

## Before changing configuration

Capture this information:

- plugin path and embedded OCCL-NET version;
- NCCL version, application version, host names, rank-to-host mapping, and
  launcher command;
- `NCCL_*`, `OCI_FRM_*`, and `FRM_*` variables used by the job;
- selected IB devices/ports, link state, link layer, and MTU;
- all rank logs with `NCCL_DEBUG=INFO` and `NCCL_DEBUG_SUBSYS=INIT,NET`.

Do not begin with TRACE on a large production workload. It can make a failure
harder to read and can materially increase output volume.

## Progressive isolation sequence

The first successful step identifies the feature boundary crossed by the
previous step. Keep all earlier changes in place while progressing.

| Step | Change | What it isolates |
|---:|---|---|
| 0 | Run the known workload with the current settings and INFO NET logging. | Establishes the failure signature. |
| 1 | `OCI_FRM_ENABLE=0` | CIM/FRM steering and its latency-driven QP selection. |
| 2 | `NCCL_IB_RESILIENCY_PORT_FAILOVER=0` | Request replay, QP failover, probing, and QP rejoin. |
| 3 | `NCCL_IB_QPS_PER_CONNECTION=1` and `NCCL_IB_SPLIT_DATA_ON_QPS=0` | Multi-QP placement and split-data behavior. |
| 4 | Unset `NCCL_NET_FORCE_MERGE` and set `NCCL_IB_MERGE_NICS=0`. | Merged/virtual NIC topology and cross-rail selection. |
| 5 | `NCCL_IB_ADAPTIVE_ROUTING=0` | IB adaptive-routing policy. |
| 6 | `NCCL_IB_ECE_ENABLE=0` | ECE negotiation and reapplication during QP rejoin. |
| 7 | Set `NCCL_IB_HCA` to one known-good HCA/port. | Device/port or rail-specific failures. |
| 8 | `NCCL_IB_RETURN_ASYNC_EVENTS=0` for a diagnostic comparison only. | Effect of propagating IB asynchronous fatal events. |
| 9 | `NCCL_NET_PLUGIN=none` | Entire OCCL-NET plugin. |

Step 8 is diagnostic only. Do not use it to mask a known device, CQ, QP, or GID
health event in production.

## Example isolation commands

Start with the plugin explicitly enabled:

```bash
export NCCL_NET_PLUGIN=/opt/occl-net/lib/liboccl-net-occl-ib.so
export NCCL_NET=IB
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=INIT,NET
```

Disable steering:

```bash
export OCI_FRM_ENABLE=0
```

Disable failover/rejoin after that:

```bash
export NCCL_IB_RESILIENCY_PORT_FAILOVER=0
```

Reduce to one non-merged, non-split QP configuration:

```bash
export NCCL_IB_QPS_PER_CONNECTION=1
export NCCL_IB_SPLIT_DATA_ON_QPS=0
unset NCCL_NET_FORCE_MERGE
export NCCL_IB_MERGE_NICS=0
```

Finally, stop loading the plugin:

```bash
export NCCL_NET_PLUGIN=none
unset NCCL_NET
```

If the problem persists after Step 9, it is outside OCCL-NET's loaded plugin
path. Continue with the application's NCCL, launcher, host networking, and
hardware support procedures.

## Symptom-to-evidence map

| Symptom | Look for | First isolation step |
|---|---|---:|
| Plugin does not load | Missing shared-object path, incompatible plugin ABI, loader errors. | Verify path and library dependencies before Step 1. |
| Wrong NIC or incompatible link layer | `NCCL_IB_HCA` messages; incompatible-device WARNs. | 7 |
| Connection or QP setup failure | QP-created records, GID/RoCE settings, `ibv_modify_qp` errors. | 5, then 6, then 7 |
| CQE/QP fatal event | `async fatal event on QP`, `Got CQE with error`, peer/vendor/HCA fields. | 2, then 7 |
| Failover loop or permanent failure | Failed QP, replacement QP, probe/rejoin retry budget, permanent-error records. | 2 |
| Rejoin stalls | `Local port is not active`, `Locally rebuilt QP`, `Waiting for peer-ready`, peer-ready backoff. | 2, then 7 |
| Throughput regression or imbalance | FRM placement/latency output, per-QP skew, rail selection. | 1, then 3–5 |
| ECE-related connection or rejoin issue | ECE configuration or cached-ECE rejoin records. | 6 |

## Full NIC Backup using NCCL_NET_FORCE_MERGE

Start with the [Full NIC Backup profile](plugin-usage.md#full-nic-backup-wip)
and collect `NCCL_DEBUG=INFO`, `NCCL_DEBUG_SUBSYS=INIT,NET` logs from every rank.
Confirm the plugin loaded before diagnosing backup behavior. Record the exact
forced groups, HCA filter, QPs per connection, split-data setting, failover
setting, plugin build, and NCCL runtime on both endpoints.

For four members arranged as two pairs with matching normalized PCI paths
within each pair, expect alias creation like:

```text
NET/IB : Made 2 GPU-local aliases for forced merge beginning at virtual device <index>
```

A connection between paired aliases with QPs per connection set to one reports:

```text
NET/IB: Force-merge negotiated activeWidth=2 Q=1 logicalQps=2 physicalQps=8 localPrivateDevices=4 peerPrivateDevices=4
```

`activeWidth` is the number of active ports. `logicalQps` describes the traffic
lanes; `physicalQps` includes their standby paths. `localPrivateDevices` and
`peerPrivateDevices` show the backup membership at each endpoint. Alias counts
and widths depend on PCI topology, so a single-port alias is not itself an
error. The alias's name and speed do not list or sum its backup members.

| Symptom or log | Check and next action |
|---|---|
| `Skipping makeVDevice, NCCL_IB_MERGE_NICS=0` | Enable `NCCL_IB_MERGE_NICS=1` along with the forced list on every rank. |
| No forced-alias creation or negotiation records | Check runtime/ABI support, a nonempty forced list, host device names, and whether `NCCL_IB_HCA` excluded members. Confirm the application selected the intended group. |
| Duplicate-device, overlapping-group, or incompatible-device warning | Use distinct members in disjoint groups with one link layer; respect the four-member v9–v11 or eight-member v12 limit. |
| `Incompatible force-merge connection protocol or mode`, `Incompatible legacy force-merge peer`, or invalid geometry offer | Align plugin builds, forced-merge mode, and `NCCL_IB_QPS_PER_CONNECTION` on both peers. Older forced-merge builds are not wire-compatible with the new negotiation. |
| `Force-merge activeWidth=1 fallback: localPublicWidth=... peerPublicWidth=...` | At least one alias has one active port. Check the device PCI paths and expected pairing; this fallback preserves backup membership. |
| `Force-merge activeWidth=1 fallback at 128-QP limit` | Two active ports would exceed the per-connection cap. The plugin preserves configured Q and backup paths using one active port. Reduce Q if two-port operation is needed. |
| `Requested QP geometry exceeds the 128-QP connection limit` | Even one active port exceeds the cap. Reduce Q or group size; `activeWidth × Q × max(localPrivateDevices, peerPrivateDevices)` must fit in 128. |
| Traffic fails instead of using a backup | Verify `NCCL_IB_RESILIENCY_PORT_FAILOVER=1`, negotiated membership, and reachable surviving paths. Capture CQE/async errors and replacement/probe/rejoin records. A second port on the failed physical NIC is not an independent backup. |
| Recovery stalls or bandwidth drops after failover | Follow the rejoin evidence in the symptom map. Check surviving-path connectivity, port state, and cross-NUMA placement. Full NIC Backup does not guarantee unchanged degraded bandwidth or arbitrary independent survivors at both endpoints. |

To isolate forced grouping while retaining failover, first unset
`NCCL_NET_FORCE_MERGE` on every rank and rerun with the other settings unchanged.
To remove merging entirely, also set `NCCL_IB_MERGE_NICS=0`. Remove persistent
forced-merge entries from launcher/configuration files too. Setting the forced
list to `0` is not an off switch. Restore the backup profile before evaluating
failover; the progressive isolation sequence disables failover at step 2.

Keep configuration and log comparisons separate from hardware fault injection.
For a backup incident, add the failure/recovery timeline and surviving devices
on both endpoints to the escalation bundle. The supported model is sequential/
one-sided N-1 with reachable peer paths, not arbitrary simultaneous bilateral
NIC loss.

## Escalation bundle

For a reproducible issue, include the smallest reproducer, complete logs from
all ranks, the environment before and after each isolation step, plugin version,
NCCL version, `ibv_devinfo`/port-state output, and whether the issue follows a
specific HCA, port, host, rank, or message size.
