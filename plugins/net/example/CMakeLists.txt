set(SRC_FILES
    ${CMAKE_CURRENT_SOURCE_DIR}/plugin.c
)

# Create the default OCCL net plugin shared library.
add_library(occl-net SHARED ${SRC_FILES})

# Set include directories
target_include_directories(occl-net PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/nccl
)

# Emit liboccl-net.so by default.
set_target_properties(occl-net PROPERTIES
    OUTPUT_NAME "occl-net"
    PREFIX "lib"
    POSITION_INDEPENDENT_CODE ON
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib
)

# Backward-compatible target name for existing build scripts.
add_library(nccl-net-example ALIAS occl-net)
