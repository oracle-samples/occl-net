<!--
 Copyright (c) 2026 Oracle and/or its affiliates.
 SPDX-License-Identifier: UPL-1.0

 Licensed under the Universal Permissive License (UPL), Version 1.0.
 See LICENSE.txt.
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

The unified runner performs the same configure/build steps before running the selected tests. On
a host, or in a container with an incomplete toolchain, `--unit-only` automatically uses the
dedicated unit-test image:

```bash
bash scripts/tests/run_net_ib_tests.sh --unit-only build-net-ib-tests
```

Package tested artifacts with:

```bash
bash scripts/tests/run_net_ib_tests.sh --unit-only \
  --build-dir build-net-ib-dev \
  --package-dir build-net-ib-artifacts
```

The `--unit-only` package contains the tested plugin shared object, source/unit test binaries, and
the test runner script. It does not contain the single-node binary:

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
bash scripts/tests/run_net_ib_tests.sh build-net-ib-tests
```

Run only unit tests:

```bash
bash scripts/tests/run_net_ib_tests.sh --unit-only build-net-ib-tests
```

When no custom CTest filter is used, `run_net_ib_tests.sh` prints structural source-level function
coverage after the unit tests. The inventory contains production definitions from translation units
owned by `occl-net-ib-plugin` and from the project headers they include. Direct coverage counts
production functions that Clang resolves as direct `CALL_EXPR` targets in a GoogleTest body or in a
test-only helper reachable from one.
Inclusive coverage adds production functions reachable through the production call graph. These
metrics do not measure line or branch execution coverage. `--show-uncovered` lists the exact
complement of direct coverage and groups it into `Indirect-only` functions reached through that
graph and `Fully uncovered` functions not reached at all. Both sections show their own count and
group entries by production source file.

Use either list option independently, or combine them:

```bash
bash scripts/tests/run_net_ib_tests.sh --unit-only --show-covered build-net-ib-tests
bash scripts/tests/run_net_ib_tests.sh --unit-only --show-uncovered build-net-ib-tests
```

The dedicated unit-test image includes the `python3-clang-14` binding required by the reporter. The
reporter fingerprints its selected project files, compilation database, and analyzer script. A
cold report parses independent translation units in up to four isolated processes. It also caches
each translation unit with its project-header dependencies, so a normal source edit reparses only
affected units. When the complete fingerprint is unchanged, the reporter reuses the persisted result
without parsing `compile_commands.json` or an AST. The unit-test binaries still run on every
invocation.

The standard net-IB CTest selection includes the reporter's Python regression suite, but it does not
generate the project function report. In addition, `--output-on-failure` only displays a test
program's stdout when that test fails. Use the runner command above when both the CTest results and
function report are required.

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
bash scripts/tests/run_net_ib_tests.sh --gpu-testbed --single-node-only
```

## Running With Runtime Coverage

Runtime coverage mode requires `lcov`, `python3`, CMake, and CTest. It builds the selected test
binaries with coverage instrumentation, runs them, prints the aggregate line and function
percentages, and writes the filtered report under:

```text
build/test/coverage/net_ib/
```

Run CPU unit tests with runtime coverage from a host or container; the runner uses the dedicated
unit-test image automatically when the local toolchain is incomplete:

```bash
bash scripts/tests/run_net_ib_tests.sh --unit-only --coverage build-net-ib-coverage
```

Run all tests with runtime coverage in a configured environment:

```bash
bash scripts/tests/run_net_ib_tests.sh --coverage build-net-ib-coverage
```

On a GPU/IB-capable host, include the single-node runtime gate:

```bash
OCCL_NET_IB_SINGLE_NODE_TESTS=1 \
  bash scripts/tests/run_net_ib_tests.sh --coverage build-net-ib-gpu-coverage
```

On a staged GPU/IB test bed workspace, use GPU test-bed mode with coverage:

```bash
cd /path/to/occl-net-workspace
bash scripts/tests/run_net_ib_tests.sh --gpu-testbed --coverage
```

GPU and combined coverage modes require native `lcov` in their configured environment.

## Runtime Coverage Artifacts

Runtime coverage artifacts are written to `build/test/coverage/net_ib/`.

`summary.txt` is the overall LCOV summary. It reports aggregate line and function coverage for
filtered net-IB production sources. Branch coverage is currently not collected, so
`branches...: no data found` is expected.

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

Use `files.tsv` to find production files and line ranges that need more runtime execution.

`coverage.info` is the filtered LCOV tracefile. The runner limits it to
`src/transport/net_ib` production files and excludes `test/` plus `gdaki/`. It can be passed to
tools such as `lcov --summary` or `genhtml`.

`tests.log` is the CTest output captured during coverage mode. It records which tests ran, passed,
failed, or skipped, but does not contribute to coverage calculations.
