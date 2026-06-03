#
# Copyright (c) 2026 OCCL-net.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Build type for release plugin artifacts")

set(BUILD_IB_TRANSPORT_PLUGIN ON CACHE BOOL "Build the standalone IB transport plugin")
set(OCCL_PLUGIN_STATIC_CUDART ON CACHE BOOL "Prefer the static CUDA runtime for the standalone IB transport plugin")
set(OCCL_PLUGIN_STATIC_CXX_RUNTIME ON CACHE BOOL "Prefer static libstdc++ for the standalone IB transport plugin when supported")

set(OCCL_NET_IB_ENABLE_TESTS OFF CACHE BOOL "Disable unit tests for release plugin builds")
set(OCCL_NET_IB_ENABLE_SINGLE_NODE_TESTS OFF CACHE BOOL "Disable single-node tests for release plugin builds")
set(OCCL_NET_IB_ENABLE_COVERAGE OFF CACHE BOOL "Disable coverage instrumentation for release plugin builds")

set(RDMA_CORE OFF CACHE BOOL "Use runtime loading instead of direct RDMA core linkage")
set(MLX5DV OFF CACHE BOOL "Use runtime loading instead of direct mlx5 linkage")
