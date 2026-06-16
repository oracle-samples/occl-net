# Copyright (c) 2026 OCCL-net.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

file(GLOB NET_IB_SINGLE_NODE_TEST_LOCAL_SOURCES CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_LIST_DIR}/*.cc"
)
list(SORT NET_IB_SINGLE_NODE_TEST_LOCAL_SOURCES)

file(GLOB NET_IB_SINGLE_NODE_TRANSPORT_SOURCES CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_LIST_DIR}/../../*.cc"
)
list(FILTER NET_IB_SINGLE_NODE_TRANSPORT_SOURCES EXCLUDE REGEX "/(cim|gin)\\.cc$")
list(SORT NET_IB_SINGLE_NODE_TRANSPORT_SOURCES)

add_executable(net_ib_single_node_tests
    ${NET_IB_SINGLE_NODE_TEST_LOCAL_SOURCES}
    ${NET_IB_SINGLE_NODE_TRANSPORT_SOURCES}
    ${CMAKE_SOURCE_DIR}/src/misc/socket.cc
    ${CMAKE_SOURCE_DIR}/src/misc/ibvwrap.cc
    ${CMAKE_SOURCE_DIR}/src/misc/ibvsymbols.cc
    ${CMAKE_SOURCE_DIR}/src/misc/mlx5dvwrap.cc
    ${CMAKE_SOURCE_DIR}/src/misc/mlx5dvsymbols.cc
    ${CMAKE_SOURCE_DIR}/src/misc/cudawrap.cc
    ${CMAKE_SOURCE_DIR}/src/misc/param.cc
    ${CMAKE_SOURCE_DIR}/src/misc/utils.cc
    ${CMAKE_SOURCE_DIR}/src/misc/nvmlwrap.cc
    ${CMAKE_SOURCE_DIR}/src/misc/gdrwrap.cc
    ${CMAKE_SOURCE_DIR}/src/debug.cc
    ${CMAKE_SOURCE_DIR}/src/os/linux.cc
)

target_include_directories(net_ib_single_node_tests
    PRIVATE
        ${CMAKE_CURRENT_LIST_DIR}
        ${CMAKE_CURRENT_LIST_DIR}/..
        ${CMAKE_CURRENT_LIST_DIR}/../..
        ${CMAKE_SOURCE_DIR}/src
        ${CMAKE_SOURCE_DIR}/src/include
        ${CMAKE_SOURCE_DIR}/src/include/plugin
        ${CMAKE_SOURCE_DIR}/src/transport/net_ib
        ${CMAKE_SOURCE_DIR}/src/transport/net_ib/gdaki
        ${CMAKE_BINARY_DIR}/include
        ${CMAKE_BINARY_DIR}/obj/include
        ${CUDAToolkit_INCLUDE_DIRS}
        ${CUDAToolkit_INCLUDE_DIRS}/cccl
)

target_link_libraries(net_ib_single_node_tests
    PRIVATE
        GTest::gtest_main
        Threads::Threads
        ${CMAKE_DL_LIBS}
        CUDA::cudart
        ${EXTRA_LIBS}
)

if(OCCL_NET_IB_ENABLE_COVERAGE)
    target_compile_options(net_ib_single_node_tests PRIVATE -O0 -g --coverage)
    target_link_options(net_ib_single_node_tests PRIVATE --coverage)
endif()

set_target_properties(net_ib_single_node_tests PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
    POSITION_INDEPENDENT_CODE ON
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/test/single_node/net_ib"
)

add_dependencies(net_ib_single_node_tests nccl_header generate_git_version)

include(GoogleTest)
gtest_discover_tests(net_ib_single_node_tests
    TEST_PREFIX net_ib.single_node.
)
