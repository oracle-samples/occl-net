# Copyright (c) 2026 Oracle and/or its affiliates.
# SPDX-License-Identifier: UPL-1.0
#
# Licensed under the Universal Permissive License (UPL), Version 1.0.
# See LICENSE.txt.

if(OCCL_NET_IB_ENABLE_TESTS)
    set(NET_IB_SOURCE_TEST_TARGETS)

    function(add_net_ib_source_test test_source)
        get_filename_component(test_stem "${test_source}" NAME_WE)
        string(REGEX REPLACE "_test$" "" source_stem "${test_stem}")
        set(target_name "net_ib_${source_stem}_tests")

        # These tests include and isolate their production source without the
        # common doubles provided by test_support.cc.
        if(source_stem STREQUAL "debug" OR
           source_stem STREQUAL "linux" OR
           source_stem STREQUAL "net_ib_plugin_stubs" OR
           source_stem STREQUAL "param")
            add_executable(${target_name} ${test_source})
        else()
            add_executable(${target_name}
                ${test_source}
                test_support.cc
            )
        endif()

        # Disable CUDA-runtime dependencies for tests whose code paths do not require them.
        if(source_stem STREQUAL "checks" OR
           source_stem STREQUAL "debug" OR
           source_stem STREQUAL "linux" OR
           source_stem STREQUAL "utils")
            target_compile_definitions(${target_name} PRIVATE OCCL_DISABLE_CUDA_RUNTIME=1)
        endif()

        # Let ibvwrap_test use libibverbs completion-status text instead of the
        # deterministic formatter from test_support.cc.
        if(source_stem STREQUAL "ibvwrap")
            target_compile_definitions(${target_name} PRIVATE OCCL_TEST_USE_PRODUCTION_IBV_TEXT=1)
        endif()

        target_include_directories(${target_name}
            PRIVATE
                ${CMAKE_CURRENT_LIST_DIR}
                ${CMAKE_CURRENT_LIST_DIR}/..
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

        target_link_libraries(${target_name}
            PRIVATE
                GTest::gtest_main
                Threads::Threads
                CUDA::cudart
                # Link the platform library required by dynamic-loader calls.
                ${CMAKE_DL_LIBS}
        )

        target_compile_options(${target_name}
            PRIVATE
                -ffunction-sections
                -fdata-sections
        )

        if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
            target_link_options(${target_name} PRIVATE -Wl,--gc-sections)
        endif()

        if(OCCL_NET_IB_ENABLE_COVERAGE)
            target_compile_options(${target_name} PRIVATE -O0 -g --coverage)
            target_link_options(${target_name} PRIVATE --coverage)
        endif()

        set_target_properties(${target_name} PROPERTIES
            CXX_STANDARD 17
            CXX_STANDARD_REQUIRED ON
            RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/test/unit/net_ib"
        )

        add_dependencies(${target_name} nccl_header generate_git_version)

        include(GoogleTest)
        gtest_discover_tests(${target_name}
            TEST_PREFIX net_ib.
        )

        set(NET_IB_SOURCE_TEST_TARGETS ${NET_IB_SOURCE_TEST_TARGETS} ${target_name} PARENT_SCOPE)
    endfunction()

    file(GLOB NET_IB_SOURCE_TEST_SOURCES CONFIGURE_DEPENDS
        "${CMAKE_CURRENT_LIST_DIR}/*_test.cc"
    )
    list(SORT NET_IB_SOURCE_TEST_SOURCES)

    foreach(net_ib_source_test_source IN LISTS NET_IB_SOURCE_TEST_SOURCES)
        add_net_ib_source_test("${net_ib_source_test_source}")
    endforeach()

    add_custom_target(net_ib_unit_tests
        DEPENDS ${NET_IB_SOURCE_TEST_TARGETS}
    )

    # Register the reporter's Python regression suite alongside the C++ unit tests.
    # Suppress bytecode so importing the reporter does not dirty the source tree.
    add_test(NAME net_ib.report_unit_function_coverage
        COMMAND "${CMAKE_COMMAND}" -E env
            "PYTHONDONTWRITEBYTECODE=1"
            "${Python3_EXECUTABLE}"
            "${CMAKE_SOURCE_DIR}/scripts/tests/test_report_unit_function_coverage.py"
    )
endif()

if(OCCL_NET_IB_ENABLE_SINGLE_NODE_TESTS)
    add_subdirectory(single_node)
endif()
