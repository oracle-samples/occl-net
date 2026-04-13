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

cmake_minimum_required(VERSION 3.22.1)

get_filename_component(OCCL_ROOT "${CMAKE_CURRENT_LIST_DIR}/../../.." ABSOLUTE)

file(READ "${OCCL_ROOT}/makefiles/version.mk" OCCL_VERSION_CONTENT)
string(REGEX REPLACE ".*NCCL_MAJOR[ ]*:=[ ]*([0-9]+).*" "\\1" NCCL_MAJOR "${OCCL_VERSION_CONTENT}")
string(REGEX REPLACE ".*NCCL_MINOR[ ]*:=[ ]*([0-9]+).*" "\\1" NCCL_MINOR "${OCCL_VERSION_CONTENT}")
string(REGEX REPLACE ".*NCCL_PATCH[ ]*:=[ ]*([0-9]+).*" "\\1" NCCL_PATCH "${OCCL_VERSION_CONTENT}")
string(REGEX REPLACE ".*NCCL_SUFFIX[ ]*:=[ ]*([a-zA-Z0-9]*).*" "\\1" NCCL_SUFFIX "${OCCL_VERSION_CONTENT}")
math(EXPR NCCL_VERSION_CODE "(${NCCL_MAJOR} * 10000) + (${NCCL_MINOR} * 100) + ${NCCL_PATCH}")

project(OCCLNetIbRccl VERSION ${NCCL_MAJOR}.${NCCL_MINOR}.${NCCL_PATCH} LANGUAGES C CXX)

if(NOT CMAKE_SYSTEM_NAME STREQUAL "Linux")
  message(FATAL_ERROR "The occl-net-ib RCCL build entrypoint supports Linux hosts only.")
endif()

include(GNUInstallDirs)

find_package(Threads REQUIRED)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" OR CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC -fvisibility=hidden -Wall -Wno-unused-function -Wno-sign-compare -Wvla -g")
endif()

file(MAKE_DIRECTORY "${CMAKE_BINARY_DIR}/include")
file(READ "${OCCL_ROOT}/src/nccl.h.in" OCCL_NCCL_HEADER_TEMPLATE)
string(REPLACE "\${nccl:Major}" "${NCCL_MAJOR}" OCCL_NCCL_HEADER_TEMPLATE "${OCCL_NCCL_HEADER_TEMPLATE}")
string(REPLACE "\${nccl:Minor}" "${NCCL_MINOR}" OCCL_NCCL_HEADER_TEMPLATE "${OCCL_NCCL_HEADER_TEMPLATE}")
string(REPLACE "\${nccl:Patch}" "${NCCL_PATCH}" OCCL_NCCL_HEADER_TEMPLATE "${OCCL_NCCL_HEADER_TEMPLATE}")
string(REPLACE "\${nccl:Suffix}" "${NCCL_SUFFIX}" OCCL_NCCL_HEADER_TEMPLATE "${OCCL_NCCL_HEADER_TEMPLATE}")
string(REPLACE "\${nccl:Version}" "${NCCL_VERSION_CODE}" OCCL_NCCL_HEADER_TEMPLATE "${OCCL_NCCL_HEADER_TEMPLATE}")
file(WRITE "${CMAKE_BINARY_DIR}/include/nccl.h" "${OCCL_NCCL_HEADER_TEMPLATE}")

set(OCCL_IB_RCCL_PLUGIN_SOURCES
    "${OCCL_ROOT}/src/plugin/net/net_ib_plugin_v11.cc"
    "${OCCL_ROOT}/src/plugin/net/net_ib_plugin_stubs.cc"
    "${OCCL_ROOT}/src/transport/net_ib/common.cc"
    "${OCCL_ROOT}/src/transport/net_ib/connect.cc"
    "${OCCL_ROOT}/src/transport/net_ib/gdr.cc"
    "${OCCL_ROOT}/src/transport/net_ib/init.cc"
    "${OCCL_ROOT}/src/transport/net_ib/p2p.cc"
    "${OCCL_ROOT}/src/transport/net_ib/p2p_resiliency.cc"
    "${OCCL_ROOT}/src/transport/net_ib/reg.cc"
    "${OCCL_ROOT}/src/misc/socket.cc"
    "${OCCL_ROOT}/src/misc/ibvwrap.cc"
    "${OCCL_ROOT}/src/misc/ibvsymbols.cc"
    "${OCCL_ROOT}/src/misc/mlx5dvwrap.cc"
    "${OCCL_ROOT}/src/misc/mlx5dvsymbols.cc"
    "${OCCL_ROOT}/src/misc/param.cc"
    "${OCCL_ROOT}/src/misc/utils.cc"
    "${OCCL_ROOT}/src/debug.cc"
    "${OCCL_ROOT}/src/os/linux.cc"
)

add_library(occl-net-ib-rccl-plugin SHARED ${OCCL_IB_RCCL_PLUGIN_SOURCES})

target_include_directories(occl-net-ib-rccl-plugin
  PRIVATE
    "${OCCL_ROOT}/src"
    "${OCCL_ROOT}/src/include"
    "${OCCL_ROOT}/src/include/plugin"
    "${OCCL_ROOT}/src/transport/net_ib"
    "${CMAKE_BINARY_DIR}/include"
)

target_compile_definitions(occl-net-ib-rccl-plugin PRIVATE
  OCCL_DISABLE_CUDA_RUNTIME=1
  NCCL_USE_CMAKE
  NCCL_OS_LINUX
  NCCL_MAJOR=${NCCL_MAJOR}
  NCCL_MINOR=${NCCL_MINOR}
  NCCL_PATCH=${NCCL_PATCH}
  NCCL_VERSION_CODE=${NCCL_VERSION_CODE}
)

if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" OR CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
  target_compile_definitions(occl-net-ib-rccl-plugin PRIVATE NCCL_COMPILER_GCC)
endif()

target_link_libraries(occl-net-ib-rccl-plugin
  PRIVATE
    Threads::Threads
    ${CMAKE_DL_LIBS}
)

target_link_options(occl-net-ib-rccl-plugin PRIVATE "-Wl,--no-undefined")

set_target_properties(occl-net-ib-rccl-plugin PROPERTIES
    OUTPUT_NAME "rccl-net-occl-ib"
    PREFIX "lib"
    POSITION_INDEPENDENT_CODE ON
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
)

set(OCCL_IB_RCCL_PLUGIN_ALIASES
    "${CMAKE_BINARY_DIR}/lib/libnccl-net-occl-ib.so"
    "${CMAKE_BINARY_DIR}/lib/liboccl-net-ib.so"
)

add_custom_command(TARGET occl-net-ib-rccl-plugin POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E copy_if_different
    "$<TARGET_FILE:occl-net-ib-rccl-plugin>"
    "${CMAKE_BINARY_DIR}/lib/libnccl-net-occl-ib.so"
  COMMAND ${CMAKE_COMMAND} -E copy_if_different
    "$<TARGET_FILE:occl-net-ib-rccl-plugin>"
    "${CMAKE_BINARY_DIR}/lib/liboccl-net-ib.so"
  VERBATIM
)

install(TARGETS occl-net-ib-rccl-plugin
  RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}"
  LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}"
  ARCHIVE DESTINATION "${CMAKE_INSTALL_LIBDIR}"
)

install(FILES ${OCCL_IB_RCCL_PLUGIN_ALIASES}
  DESTINATION "${CMAKE_INSTALL_LIBDIR}"
  OPTIONAL
)
