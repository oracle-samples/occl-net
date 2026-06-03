<!--
Copyright (c) 2026 OCCL-net.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->

# Portable Net Plugin Release

Build the vendored `liboccl-net-ib.so` from the single Linux devcontainer in `.devcontainer/linux/devcontainer.json`. The interactive VS Code container uses a modern Ubuntu userspace because current VS Code Server cannot run on CentOS 7. The release artifact is still built inside the CentOS 7 CUDA builder target by `scripts/build-linux-release-plugin.sh`.

The release builder target is based on `nvidia/cuda:12.4.0-devel-centos7` so the plugin is compiled against a glibc 2.17 environment. That libc floor is the compatibility target for stock Oracle Linux 7 through 10 and Ubuntu 14.04 or newer. Newer distributions can usually run binaries built against older glibc, but older distributions cannot run binaries that require newer `GLIBC_*` symbols.

CUDA 12.4 is used as the builder because it is the newest CUDA line still suitable for the CentOS 7 / glibc 2.17 build floor. CUDA 13 runtime support is validated separately on CUDA 13-class drivers and hosts; do not switch the release builder to CUDA 13 unless the OL7 target is explicitly dropped.

CPU architecture is the remaining hard boundary. This release devcontainer is pinned to `linux/amd64`; publish a separate artifact if another architecture becomes required.

Build from VS Code or a shell in the devcontainer:

```bash
bash scripts/build-linux-release-plugin.sh
```

Inspect the artifact before publishing:

```bash
bash scripts/verify-linux-release-plugin.sh build/linux-release/lib/liboccl-net-ib.so
```

Release gates:

- `NEEDED` must not include `libcudart`, `libstdc++`, `libibverbs`, or `libmlx5`.
- The highest `GLIBC_*` requirement must be no newer than `GLIBC_2.17`.
- Exported dynamic symbols must include `ncclNetPlugin_v6` through `ncclNetPlugin_v12`.
- Static C++ runtime symbols should not appear as public dynamic symbols.
- The same artifact must load and initialize on CUDA 12.x and CUDA 13-class runtime hosts before publication.
