# Continuum SLM — Android CMakeLists.txt
# Builds the shared library (.so) for Android arm64-v8a using NDK.
#
# Usage:
#   export ANDROID_NDK=/path/to/ndk
#   cmake -DCMAKE_TOOLCHAIN_FILE=$ANDROID_NDK/build/cmake/android.toolchain.cmake \
#         -DANDROID_ABI=arm64-v8a \
#         -DANDROID_PLATFORM=android-28 \
#         -DCMAKE_BUILD_TYPE=Release \
#         -B build-android
#   cmake --build build-android
#
# Or use the convenience script:  ./build.sh

cmake_minimum_required(VERSION 3.10)
project(continuum_slm LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ============================================================================
# Optimization flags for ARM Cortex-A (phone CPUs)
# ============================================================================
if(ANDROID)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -ffast-math -funroll-loops")
    # Use bare armv8-a for maximum compatibility (works on all arm64 phones)
    if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=armv8-a")
    endif()
endif()

# ============================================================================
# Source files
# ============================================================================
set(SOURCES
    ${CMAKE_SOURCE_DIR}/../src/tensor.cpp
    ${CMAKE_SOURCE_DIR}/../src/model.cpp
    ${CMAKE_SOURCE_DIR}/../src/sampler.cpp
    ${CMAKE_SOURCE_DIR}/../jni/continuum_jni.cpp
)

set(HEADERS
    ${CMAKE_SOURCE_DIR}/../include/tensor.h
    ${CMAKE_SOURCE_DIR}/../include/model.h
    ${CMAKE_SOURCE_DIR}/../include/sampler.h
)

# ============================================================================
# Build shared library
# ============================================================================
add_library(continuum_jni SHARED ${SOURCES} ${HEADERS})

target_include_directories(continuum_jni PRIVATE
    ${CMAKE_SOURCE_DIR}/../include
)

target_link_libraries(continuum_jni
    log     # Android logging
    m       # math library (for fmaxf, sqrtf, etc.)
)

# ============================================================================
# Strip symbols for smaller .so size
# ============================================================================
if(CMAKE_BUILD_TYPE STREQUAL "Release")
    set_target_properties(continuum_jni PROPERTIES
        LINK_FLAGS "-Wl,--gc-sections -Wl,--strip-all"
    )
endif()
