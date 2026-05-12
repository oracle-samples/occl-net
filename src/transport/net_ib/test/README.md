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

# Net-IB Tests

The net-IB tests are opt-in. A default project build does not create test
binaries or coverage targets.

## Test Structure

`src/transport/net_ib/test` contains source-oriented unit tests. These tests
are named `*_test.cc`, are discovered automatically by CMake, and produce one
GoogleTest binary per test source:

```text
build/test/unit/net_ib/net_ib_<source>_tests
```

Unit tests include the net-IB production source they cover directly and use
test-local stubs for logging, parameter loading, OS helpers, verbs wrappers,
and other hardware-facing dependencies. They are intended to run in the dev
container without GPU or InfiniBand hardware.

`src/transport/net_ib/test/single_node` contains GPU-local single-node tests.
These tests link the net-IB plugin implementation with the real runtime helper
sources and exercise device discovery, listen/connect/accept, memory
registration, host send/receive, flush, and CUDA-buffer smoke coverage. The
single-node binary is:

```text
build/test/single_node/net_ib/net_ib_single_node_tests
```

Single-node tests require a GPU/IB-capable host and are gated at runtime. Set
`OCCL_NET_IB_SINGLE_NODE_TESTS=1` to run them; otherwise they skip cleanly.

## Build And Distribution

To build both test classes from a normal build tree:

```bash
cmake -S . -B build-net-ib-tests \
  -DCMAKE_BUILD_TYPE=Debug \
  -DOCCL_NET_IB_ENABLE_TESTS=ON \
  -DOCCL_NET_IB_ENABLE_SINGLE_NODE_TESTS=ON
cmake --build build-net-ib-tests \
  --target net_ib_unit_tests net_ib_single_node_tests \
  --parallel
```

The unified runner performs the same configure/build steps and then runs tests.
Use `--unit-only` for the dev-container gate before upload:

```bash
bash scripts/run_net_ib_tests.sh --unit-only build-net-ib-tests
```

Package tested artifacts from the dev container with:

```bash
bash scripts/run_net_ib_tests.sh --unit-only \
  --build-dir build-net-ib-dev \
  --package-dir build-net-ib-artifacts
```

The package contains the tested plugin shared object, all net-IB test
binaries, and the test runner script:

```text
build-net-ib-artifacts/
build-net-ib-artifacts/lib/liboccl-net-ib.so
build-net-ib-artifacts/bin/net_ib_*_tests
build-net-ib-artifacts/scripts/run_net_ib_tests.sh
```

To upload from a dev-container task, set `OCCL_HOST_USER`, `OCCL_HOST`,
`OCCL_FOLDER`, and optionally `OCCL_NET_IB_REMOTE_DIR` in `.env.local`, then
run `Build and Deploy to GPU VM`. That task does not run GPU-side tests by
default; it only uploads the tested artifacts.

On a GPU/IB-capable test bed, run the uploaded single-node binary directly:

```bash
cd /path/to/uploaded/net-ib-tested-artifacts
OCCL_NET_IB_SINGLE_NODE_TESTS=1 NCCL_DEBUG=WARN \
  ./bin/net_ib_single_node_tests --gtest_brief=1
```

## Running Tests

Run all net-IB tests without coverage:

```bash
bash scripts/run_net_ib_tests.sh build-net-ib-tests
```

Run only unit tests:

```bash
bash scripts/run_net_ib_tests.sh --unit-only build-net-ib-tests
```

Run only unit tests from an existing build tree:

```bash
ctest --test-dir build-net-ib-tests --output-on-failure \
  -R '^net_ib\.' \
  -E '^net_ib\.single_node\.'
```

Run only single-node tests from an existing build tree on a GPU/IB-capable
host:

```bash
OCCL_NET_IB_SINGLE_NODE_TESTS=1 \
  ctest --test-dir build-net-ib-tests --output-on-failure -R '^net_ib\.single_node\.'
```

Run only single-node tests through the runner on a staged GPU/IB source
workspace:

```bash
cd /path/to/occl-net-workspace
bash scripts/run_net_ib_tests.sh --gpu-testbed --single-node-only
```

## Running With Coverage

Coverage mode requires `lcov`, `python3`, CMake, and CTest. It builds the same
unit and single-node test binaries with coverage instrumentation, runs tests,
and writes the filtered report under:

```text
build/test/coverage/net_ib/
```

Run all tests with coverage:

```bash
bash scripts/run_net_ib_tests.sh --coverage build-net-ib-coverage
```

On a GPU/IB-capable host, include the single-node runtime gate:

```bash
OCCL_NET_IB_SINGLE_NODE_TESTS=1 \
  bash scripts/run_net_ib_tests.sh --coverage build-net-ib-gpu-coverage
```

On a staged GPU/IB test bed workspace, use GPU test-bed mode with coverage:

```bash
cd /path/to/occl-net-workspace
bash scripts/run_net_ib_tests.sh --gpu-testbed --coverage
```

That coverage command requires native `lcov` on the host.

## Coverage Artifacts

Coverage artifacts are written to `build/test/coverage/net_ib/`.

`summary.txt` is the overall LCOV summary. It reports aggregate line and
function coverage for filtered net-IB production sources. Branch coverage is
currently not collected, so `branches...: no data found` is expected.

`files.tsv` is the per-file breakdown generated by the runner. Columns are:

```text
file
covered_lines
total_lines
line_coverage_pct
covered_functions
total_functions
function_coverage_pct
uncovered_lines
first_uncovered_line_ranges
```

Use `files.tsv` to find the source files and line ranges that need more tests.

`coverage.info` is the filtered LCOV tracefile. It contains the detailed
machine-readable records used by LCOV tools. The runner filters the tracefile
to `src/transport/net_ib` production files and excludes `test/` plus `gdaki/`.
You can pass it to tools such as `lcov --summary` or `genhtml`.

`tests.log` is the CTest output captured during coverage mode. It is not part
of the coverage math, but it records which tests ran, passed, failed, or
skipped for that coverage run.
