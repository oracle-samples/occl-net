# Transport sources
set(NET_IB_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/cim.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/common.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/connect.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/gdr.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/init.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/p2p.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/p2p_resiliency.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/reg.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/gin.cc
)

if(OCCL_NET_IB_ENABLE_TESTS OR OCCL_NET_IB_ENABLE_SINGLE_NODE_TESTS)
    if(NOT EXISTS "${CMAKE_SOURCE_DIR}/third-party/googletest/CMakeLists.txt")
        message(FATAL_ERROR "Net-IB tests require third-party/googletest. Run: git submodule update --init third-party/googletest")
    endif()
    include(CTest)
    set(BUILD_GMOCK OFF CACHE BOOL "" FORCE)
    set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
    add_subdirectory(${CMAKE_SOURCE_DIR}/third-party/googletest ${CMAKE_BINARY_DIR}/third-party/googletest EXCLUDE_FROM_ALL)
    add_subdirectory(test)
endif()

add_subdirectory(gdaki)

# Add transport sources to parent scope
set(TRANSPORT_SOURCES ${TRANSPORT_SOURCES} ${NET_IB_SOURCES} PARENT_SCOPE)

# Add DOCA device headers to parent scope
set(DEVICE_DOCA_HEADERS ${DEVICE_DOCA_HEADERS} PARENT_SCOPE)

# Add DOCA sources to parent scope
set(DOCA_SOURCES ${DOCA_SOURCES} PARENT_SCOPE)

# Add DOCA_HOME to parent scope
set(DOCA_HOME ${DOCA_HOME} PARENT_SCOPE)
