cmake_minimum_required(VERSION 3.8)

project(SLearn C CXX)

set(This SLearn)

set(CMAKE_C_STANDARD 99)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(PYBIND11_CPP_STANDARD -std=c++14)

# google test
#add_subdirectory(googletest)
#add_subdirectory(test)

# file(GLOB SOURCES "src/*.cpp")
# file(GLOB HEADERS "src/*.hpp")

# include_directories(/usr/include/python3.8/)

# add_subdirectory(pybind11)

# add_library(${This} SHARED ${SOURCES} ${HEADERS})


# eigen and pybind11
find_package( PkgConfig )
pkg_check_modules( EIGEN3 REQUIRED eigen3 )
include_directories( ${EIGEN3_INCLUDE_DIRS} )


find_package(pybind11 REQUIRED)

# Python.h
include_directories(/usr/include/python3.8)

# google test
add_subdirectory(googletest)
add_subdirectory(test)


file(GLOB SOURCES "src/*.cpp")
file(GLOB HEADERS "src/*.hpp")

add_library(${This} SHARED ${SOURCES} ${HEADERS})
pybind11_add_module(slearn ${SOURCES} ${HEADERS})