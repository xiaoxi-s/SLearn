cmake_minimum_required(VERSION 3.8)
# as add_subdirectory() enters, this file is what the function is looking for

set(This SLearnTests)

set(Sources
    test_linear_regression.cpp
    CSVReader.cpp
    test_activation.cpp
)

add_executable(${This} ${Sources})

target_link_libraries(
    ${This} PUBLIC
    gtest_main
    SLearn
)

add_test(
    NAME ${This}
    COMMAND ${This}
)
