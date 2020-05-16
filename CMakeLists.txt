cmake_minimum_required(VERSION 3.8)

project(SLearn C CXX)

set(This SLearn)

# set version requirements
set(CMAKE_C_STANDARD 99)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(PYBIND11_CPP_STANDARD -std=c++14)

set(TEST FALSE)

# eigen
find_package( PkgConfig )
pkg_check_modules( EIGEN3 REQUIRED eigen3 )
include_directories( ${EIGEN3_INCLUDE_DIRS} )

# pybind11
find_package(pybind11 REQUIRED)
# Python.h
include_directories(/usr/include/python3.8)

# collect all the files
file(GLOB SOURCES "src/*.cpp")
file(GLOB HEADERS "src/*.hpp")

pybind11_add_module(slearn ${SOURCES} ${HEADERS})

# google test
if(TEST)
    add_subdirectory(googletest)
    add_subdirectory(test)

    add_library(${This} SHARED ${SOURCES} ${HEADERS})
endif(TEST)

