#
# Create a target for executing the Catch unit tests.
#

add_executable(nanoFlannExample_test
        main.cpp
        nano_flann_example_test.cpp
        )

target_include_directories(nanoFlannExample_test
        PUBLIC
        test/
        "${PROJECT_BINARY_DIR}"
        "${CMAKE_BINARY_DIR}"
        )
set(BUILD_TESTING OFF)
set(EIGEN_BUILD_PKGCONFIG OFF)
target_link_libraries(nanoFlannExample_test
        Catch2::Catch2
        Eigen3::Eigen
        nanoflann::nanoflann
        nanoFlannExample
        )

cmake_policy(SET CMP0110 NEW)
list(APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake/Modules")




ParseAndAddCatchTests(nanoFlannExample_test)