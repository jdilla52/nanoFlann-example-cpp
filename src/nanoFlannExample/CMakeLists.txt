### config for s3
add_library(nanoFlannExample STATIC
        nanoFlannExample.cpp
        nanoFlannExample.h
        )

target_link_libraries(nanoFlannExample
        nanoflann::nanoflann)

target_include_directories(nanoFlannExample PUBLIC src/nanoFlannExample)

target_include_directories(nanoFlannExample PRIVATE "${CMAKE_BINARY_DIR}" "${PROJECT_BINARY_DIR}")
