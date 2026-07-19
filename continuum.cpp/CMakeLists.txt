cmake_minimum_required(VERSION 3.14)
project(continuum VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native -ffast-math -funroll-loops")
set(CMAKE_CXX_FLAGS_DEBUG "-O0 -g")

# Find BLAS (optional, for optimized matmul)
find_package(BLAS QUIET)
if(BLAS_FOUND)
    add_definitions(-DHAVE_BLAS)
    message(STATUS "BLAS found: ${BLAS_LIBRARIES}")
else()
    message(STATUS "BLAS not found — using reference matmul (slower)")
endif()

# Source files
set(SOURCES
    src/tensor.cpp
    src/model.cpp
    src/sampler.cpp
    src/continuum.cpp
)

# Headers
set(HEADERS
    include/tensor.h
    include/model.h
    include/sampler.h
)

# Executable
add_executable(continuum ${SOURCES})

target_include_directories(continuum PRIVATE include)

if(BLAS_FOUND)
    target_link_libraries(continuum PRIVATE ${BLAS_LIBRARIES})
endif()

# Install
install(TARGETS continuum DESTINATION bin)
