cmake_minimum_required(VERSION 3.3)
set(CMAKE_CXX_STANDARD 17)
project(nanoFlannExample LANGUAGES CXX)
#set(ALLOW_DUPLICATE_CUSTOM_TARGETS true)

add_subdirectory(thirdparty)

# library
add_subdirectory(src/nanoFlannExample)
target_include_directories(nanoFlannExample PUBLIC src)

# tests
#if($ENV{TEST_FLAN_EXAMPLE})
#    message("Generating tests")
    add_subdirectory(test)
#elseif(${TEST_FLAN_EXAMPLE})
#    message("Generating tests")
#    add_subdirectory(test)
#else()
#    message("Building without tests")
#endif()