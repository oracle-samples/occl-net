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

if(OCCL_NET_IB_ENABLE_TESTS)
    set(NET_IB_SOURCE_TEST_TARGETS)

    function(add_net_ib_source_test test_source)
        get_filename_component(test_stem "${test_source}" NAME_WE)
        string(REGEX REPLACE "_test$" "" source_stem "${test_stem}")
        set(target_name "net_ib_${source_stem}_tests")

        add_executable(${target_name}
            ${test_source}
            test_support.cc
        )

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
        )

        target_link_libraries(${target_name}
            PRIVATE
                GTest::gtest_main
                Threads::Threads
                CUDA::cudart
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
endif()

if(OCCL_NET_IB_ENABLE_SINGLE_NODE_TESTS)
    add_subdirectory(single_node)
endif()
