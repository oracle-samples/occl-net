# Copyright (c) 2026 Oracle and/or its affiliates.
# SPDX-License-Identifier: UPL-1.0
#
# Licensed under the Universal Permissive License (UPL), Version 1.0.
# See LICENSE.txt.

set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Build type for release plugin artifacts")

set(BUILD_IB_TRANSPORT_PLUGIN ON CACHE BOOL "Build the standalone IB transport plugin")
set(OCCL_PLUGIN_STATIC_CUDART ON CACHE BOOL "Prefer the static CUDA runtime for the standalone IB transport plugin")
set(OCCL_PLUGIN_STATIC_CXX_RUNTIME ON CACHE BOOL "Prefer static libstdc++ for the standalone IB transport plugin when supported")

set(OCCL_NET_IB_ENABLE_TESTS OFF CACHE BOOL "Disable unit tests for release plugin builds")
set(OCCL_NET_IB_ENABLE_SINGLE_NODE_TESTS OFF CACHE BOOL "Disable single-node tests for release plugin builds")
set(OCCL_NET_IB_ENABLE_COVERAGE OFF CACHE BOOL "Disable coverage instrumentation for release plugin builds")

set(RDMA_CORE OFF CACHE BOOL "Use runtime loading instead of direct RDMA core linkage")
set(MLX5DV OFF CACHE BOOL "Use runtime loading instead of direct mlx5 linkage")
