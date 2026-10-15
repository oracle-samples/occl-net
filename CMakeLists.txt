# Copyright (c) 2026 Oracle and/or its affiliates.
# SPDX-License-Identifier: UPL-1.0
#
# Licensed under the Universal Permissive License (UPL), Version 1.0.
# See LICENSE.txt.

cmake_minimum_required(VERSION 3.22.1)

file(READ "${CMAKE_CURRENT_SOURCE_DIR}/makefiles/version.mk" occl_version_content)
string(REGEX REPLACE ".*NCCL_MAJOR[ ]*:=[ ]*([0-9]+).*" "\\1" occl_major "${occl_version_content}")
string(REGEX REPLACE ".*NCCL_MINOR[ ]*:=[ ]*([0-9]+).*" "\\1" occl_minor "${occl_version_content}")
string(REGEX REPLACE ".*NCCL_PATCH[ ]*:=[ ]*([0-9]+).*" "\\1" occl_patch "${occl_version_content}")
string(REGEX REPLACE ".*NCCL_SUFFIX[ ]*:=[ ]*([a-zA-Z0-9]*).*" "\\1" occl_suffix "${occl_version_content}")
math(EXPR occl_version_code "(${occl_major} * 10000) + (${occl_minor} * 100) + ${occl_patch}")

project(OCCLNetIbPlugin
  VERSION "${occl_major}.${occl_minor}.${occl_patch}"
  LANGUAGES CXX C
)

include(GNUInstallDirs)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "Release")
endif()

option(BUILD_IB_TRANSPORT_PLUGIN "Build the standalone IB transport plugin" ON)
option(TRACE "Enable tracing" OFF)
option(WERROR "Treat warnings as errors" OFF)
option(NET_PROFILER "Enable network profiler" OFF)
option(RDMA_CORE "Enable RDMA core" OFF)
option(MLX5DV "Enable MLX5DV" OFF)
option(MAX_EXT_NET_PLUGINS "Maximum external network plugins" 0)
option(OCCL_PLUGIN_STATIC_CUDART "Compatibility knob for older standalone plugin builds; ignored because this build does not link a GPU runtime" ON)
option(OCCL_PLUGIN_STATIC_CXX_RUNTIME "Link the standalone IB transport plugin against static libstdc++ when supported while keeping libgcc shared" ON)
option(OCCL_NET_IB_ENABLE_TESTS "Build OCCL Net IB GoogleTest targets" OFF)
option(OCCL_NET_IB_ENABLE_SINGLE_NODE_TESTS "Build OCCL Net IB GPU-local single-node GoogleTest targets" OFF)
option(OCCL_NET_IB_ENABLE_COVERAGE "Build net-IB transport sources with gcov/lcov coverage instrumentation" OFF)
set(OCCL_NET_IB_PLUGIN_VERSION "0.1.0" CACHE STRING "Version string embedded in the standalone IB transport plugin")

if(OCCL_NET_IB_ENABLE_COVERAGE)
  set(OCCL_NET_IB_ENABLE_TESTS ON CACHE BOOL "Build OCCL Net IB GoogleTest targets" FORCE)
  set(OCCL_NET_IB_ENABLE_SINGLE_NODE_TESTS ON CACHE BOOL "Build OCCL Net IB GPU-local single-node GoogleTest targets" FORCE)
endif()

if(OCCL_NET_IB_ENABLE_TESTS OR OCCL_NET_IB_ENABLE_SINGLE_NODE_TESTS)
  enable_testing()
  include(CTest)
endif()

if(NOT CMAKE_SYSTEM_NAME STREQUAL "Linux")
  message(FATAL_ERROR "The standalone IB transport plugin build currently supports Linux only")
endif()

if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" OR CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
  set(occl_compiler_gcc ON)
else()
  message(FATAL_ERROR "The standalone IB transport plugin build currently requires GCC or Clang")
endif()

set(stale_cache_file_vars
  CUDAToolkit_rt_LIBRARY
  RT_LIBRARY
)
foreach(stale_cache_file_var IN LISTS stale_cache_file_vars)
  get_property(stale_cache_file_type CACHE "${stale_cache_file_var}" PROPERTY TYPE)
  if(stale_cache_file_type)
    get_property(stale_cache_file_path CACHE "${stale_cache_file_var}" PROPERTY VALUE)
    if(stale_cache_file_path AND NOT EXISTS "${stale_cache_file_path}")
      message(STATUS "Clearing stale cache entry ${stale_cache_file_var}=${stale_cache_file_path}")
      unset(${stale_cache_file_var} CACHE)
    endif()
  endif()
endforeach()

find_package(CUDAToolkit REQUIRED)
find_package(Threads REQUIRED)
find_package(Python3 REQUIRED COMPONENTS Interpreter)

string(REGEX MATCH "([0-9]+\\.[0-9]+)" cuda_version "${CUDAToolkit_VERSION}")
string(REGEX MATCH "([0-9]+)" cuda_major "${cuda_version}")
string(REGEX REPLACE ".*\\.([0-9]+)$" "\\1" cuda_minor "${cuda_version}")

if(${cuda_major} GREATER_EQUAL 13)
  set(CMAKE_CXX_STANDARD 17)
else()
  set(CMAKE_CXX_STANDARD 14)
endif()
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_compile_definitions(
  NCCL_USE_CMAKE
  NCCL_OS_LINUX
  NCCL_COMPILER_GCC
  NCCL_MAJOR=${occl_major}
  NCCL_MINOR=${occl_minor}
  NCCL_PATCH=${occl_patch}
  NCCL_VERSION_CODE=${occl_version_code}
  CUDA_MAJOR=${cuda_major}
  CUDA_MINOR=${cuda_minor}
  DOCA_VERBS_USE_CUDA_WRAPPER
  DOCA_VERBS_USE_NET_WRAPPER
  NCCL_GIN_PROXY_ENABLE=1
)
set(ENV{NCCL_USE_CMAKE} "1")

if(TRACE)
  add_compile_definitions(ENABLE_TRACE)
endif()

if(NET_PROFILER)
  add_compile_definitions(NCCL_ENABLE_NET_PROFILING=1)
endif()

if(MAX_EXT_NET_PLUGINS GREATER 0)
  add_compile_definitions(NCCL_NET_MAX_PLUGINS=${MAX_EXT_NET_PLUGINS})
endif()

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC -fvisibility=hidden -Wall -Wno-unused-function -Wno-sign-compare -Wvla -g")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS} -O0")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS} -O3")

if(WERROR)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Werror")
endif()

set(extra_libs)
if(RDMA_CORE)
  add_compile_definitions(NCCL_BUILD_RDMA_CORE=1)
  find_library(VERBS_LIBRARY NAMES verbs)
  if(VERBS_LIBRARY)
    list(APPEND extra_libs verbs)
  endif()
endif()

if(MLX5DV)
  add_compile_definitions(NCCL_BUILD_MLX5DV=1)
  find_library(MLX5_LIBRARY NAMES mlx5)
  if(MLX5_LIBRARY)
    list(APPEND extra_libs mlx5)
  endif()
endif()

find_library(RT_LIBRARY NAMES rt)
if(RT_LIBRARY)
  list(APPEND extra_libs rt)
endif()
set(EXTRA_LIBS ${extra_libs})

set(generated_include_dir "${CMAKE_BINARY_DIR}/include")
set(generated_obj_include_dir "${CMAKE_BINARY_DIR}/obj/include")
file(MAKE_DIRECTORY "${generated_include_dir}" "${generated_obj_include_dir}")

file(READ "${CMAKE_CURRENT_SOURCE_DIR}/src/nccl.h.in" generated_api_header)
string(REPLACE "\${nccl:Major}" "${occl_major}" generated_api_header "${generated_api_header}")
string(REPLACE "\${nccl:Minor}" "${occl_minor}" generated_api_header "${generated_api_header}")
string(REPLACE "\${nccl:Patch}" "${occl_patch}" generated_api_header "${generated_api_header}")
string(REPLACE "\${nccl:Suffix}" "${occl_suffix}" generated_api_header "${generated_api_header}")
string(REPLACE "\${nccl:Version}" "${occl_version_code}" generated_api_header "${generated_api_header}")
file(WRITE "${generated_include_dir}/nccl.h" "${generated_api_header}")

add_custom_target(nccl_header
  DEPENDS "${generated_include_dir}/nccl.h"
)

set(git_version_file "${generated_obj_include_dir}/nccl_git_version.h")
add_custom_command(
  OUTPUT "${git_version_file}"
  COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/src/misc/generate_git_version.py" "${git_version_file}"
  DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/src/misc/generate_git_version.py"
  COMMENT "Checking git version"
)
add_custom_target(generate_git_version
  DEPENDS "${git_version_file}"
)

if(BUILD_IB_TRANSPORT_PLUGIN)
  include(CMakePushCheckState)
  include(CheckCXXSourceCompiles)

  set(occl_plugin_cxx_runtime_linkage "dynamic")
  set(occl_plugin_cxx_runtime_link_options)
  if(OCCL_PLUGIN_STATIC_CXX_RUNTIME)
    unset(OCCL_PLUGIN_CAN_STATIC_LINK_CXX_RUNTIME CACHE)
    unset(OCCL_PLUGIN_CAN_STATIC_LINK_CXX_RUNTIME)
    cmake_push_check_state(RESET)
    list(APPEND CMAKE_REQUIRED_LINK_OPTIONS
      "-static-libstdc++"
    )
    check_cxx_source_compiles("
      #include <string>
      int main() {
        std::string s = \"occl\";
        return s.size() == 4 ? 0 : 1;
      }
    " OCCL_PLUGIN_CAN_STATIC_LINK_CXX_RUNTIME)
    cmake_pop_check_state()
    if(OCCL_PLUGIN_CAN_STATIC_LINK_CXX_RUNTIME)
      set(occl_plugin_cxx_runtime_linkage "static-libstdc++/shared-libgcc")
      list(APPEND occl_plugin_cxx_runtime_link_options
        "-static-libstdc++"
      )
    else()
      message(WARNING "OCCL_PLUGIN_STATIC_CXX_RUNTIME=ON but the current toolchain cannot link a trivial C++ program with -static-libstdc++; using dynamic C++ runtime for occl-net-ib-plugin")
    endif()
  endif()

  if(OCCL_PLUGIN_STATIC_CUDART)
    message(STATUS "occl-net-ib-plugin GPU runtime linkage: disabled (OCCL_PLUGIN_STATIC_CUDART is ignored for the host-only standalone build)")
  else()
    message(STATUS "occl-net-ib-plugin GPU runtime linkage: disabled (host-side verbs plus DMA-BUF)")
  endif()
  message(STATUS "occl-net-ib-plugin C++ runtime linkage: ${occl_plugin_cxx_runtime_linkage}")

  set(ib_transport_plugin_net_ib_sources
    src/transport/net_ib/common.cc
    src/transport/net_ib/connect.cc
    src/transport/net_ib/gdr.cc
    src/transport/net_ib/init.cc
    src/transport/net_ib/p2p.cc
    src/transport/net_ib/p2p_resiliency.cc
    src/transport/net_ib/reg.cc
  )

  set(ib_transport_plugin_sources
    src/plugin/net/net_ib_plugin_v11.cc
    src/plugin/net/net_ib_plugin_stubs.cc
    src/transport/net_ib/cim.cc
    ${ib_transport_plugin_net_ib_sources}
    src/misc/socket.cc
    src/misc/ibvwrap.cc
    src/misc/ibvsymbols.cc
    src/misc/mlx5dvwrap.cc
    src/misc/mlx5dvsymbols.cc
    src/misc/param.cc
    src/misc/utils.cc
    src/misc/nvmlwrap.cc
    src/misc/gdrwrap.cc
    src/debug.cc
    src/os/linux.cc
  )

  add_library(occl-net-ib-plugin SHARED
    ${ib_transport_plugin_sources}
  )

  target_include_directories(occl-net-ib-plugin
    PRIVATE
      "${CMAKE_CURRENT_SOURCE_DIR}/src"
      "${CMAKE_CURRENT_SOURCE_DIR}/src/include"
      "${CMAKE_CURRENT_SOURCE_DIR}/src/include/plugin"
      "${CMAKE_CURRENT_SOURCE_DIR}/src/transport/net_ib"
      "${generated_include_dir}"
      "${generated_obj_include_dir}"
      "${CUDAToolkit_INCLUDE_DIRS}"
      "${CUDAToolkit_INCLUDE_DIRS}/cccl"
  )

  target_compile_definitions(occl-net-ib-plugin PRIVATE
    OCCL_DISABLE_CUDA_RUNTIME=1
    OCCL_NET_IB_PLUGIN_VERSION="${OCCL_NET_IB_PLUGIN_VERSION}"
  )

  target_link_libraries(occl-net-ib-plugin
    PRIVATE
      Threads::Threads
      ${CMAKE_DL_LIBS}
      ${extra_libs}
  )

  set(occl_net_ib_plugin_version_script
    "${CMAKE_CURRENT_SOURCE_DIR}/src/plugin/net/occl-net-ib-plugin.map"
  )
  target_link_options(occl-net-ib-plugin PRIVATE
    "-Wl,--no-undefined"
    "-Wl,--exclude-libs,ALL"
    "-Wl,--version-script=${occl_net_ib_plugin_version_script}"
    ${occl_plugin_cxx_runtime_link_options}
  )
  set_property(TARGET occl-net-ib-plugin APPEND PROPERTY
    LINK_DEPENDS "${occl_net_ib_plugin_version_script}"
  )

  if(OCCL_NET_IB_ENABLE_COVERAGE)
    foreach(ib_transport_plugin_net_ib_source IN LISTS ib_transport_plugin_net_ib_sources)
      set_source_files_properties(${ib_transport_plugin_net_ib_source} PROPERTIES
        COMPILE_OPTIONS "-O0;-g;--coverage"
      )
    endforeach()
    target_link_options(occl-net-ib-plugin PRIVATE --coverage)
  endif()

  add_dependencies(occl-net-ib-plugin nccl_header generate_git_version)

  set_target_properties(occl-net-ib-plugin PROPERTIES
    OUTPUT_NAME "occl-net-ib"
    PREFIX "lib"
    POSITION_INDEPENDENT_CODE ON
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
  )

  install(TARGETS occl-net-ib-plugin
    RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}"
    LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}"
    ARCHIVE DESTINATION "${CMAKE_INSTALL_LIBDIR}"
  )
endif()

if(OCCL_NET_IB_ENABLE_TESTS OR OCCL_NET_IB_ENABLE_SINGLE_NODE_TESTS)
  if(NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/third-party/googletest/CMakeLists.txt")
    message(FATAL_ERROR "Net-IB tests require the third-party/googletest submodule. Run `git submodule update --init third-party/googletest`.")
  endif()

  set(BUILD_GMOCK OFF CACHE BOOL "" FORCE)
  set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
  add_subdirectory(
    "${CMAKE_CURRENT_SOURCE_DIR}/third-party/googletest"
    "${CMAKE_BINARY_DIR}/third-party/googletest"
    EXCLUDE_FROM_ALL
  )
  add_subdirectory(src/transport/net_ib/test)
endif()
