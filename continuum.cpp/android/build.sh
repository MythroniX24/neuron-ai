#!/bin/bash
# build.sh — One-command Android .so build for Continuum SLM
#
# Prerequisites:
#   1. Install Android NDK (r25 or newer)
#   2. Set ANDROID_NDK or edit the path below
#
# Usage:
#   cd continuum.cpp/android
#   ./build.sh
#
# Output:
#   build-android/lib/arm64-v8a/libcontinuum_jni.so  (~2-5 MB)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

# ─── NDK Detection ───
if [ -z "${ANDROID_NDK:-}" ]; then
    # Common NDK locations
    for candidate in \
        "$HOME/Android/Sdk/ndk/"* \
        "$HOME/android-ndk-r"* \
        "/usr/local/lib/android/sdk/ndk/"* \
        "$ANDROID_HOME/ndk/"*; do
        if [ -d "$candidate" ]; then
            ANDROID_NDK="$candidate"
            break
        fi
    done
fi

if [ -z "${ANDROID_NDK:-}" ] || [ ! -d "$ANDROID_NDK" ]; then
    echo "ERROR: Android NDK not found."
    echo "  Set ANDROID_NDK environment variable:"
    echo "  export ANDROID_NDK=/path/to/ndk"
    echo ""
    echo "  Download from: https://developer.android.com/ndk/downloads"
    exit 1
fi

echo "Using NDK: $ANDROID_NDK"

# ─── Build type ───
BUILD_TYPE="${1:-Release}"
BUILD_DIR="$SCRIPT_DIR/build-android"
TOOLCHAIN="$ANDROID_NDK/build/cmake/android.toolchain.cmake"

if [ ! -f "$TOOLCHAIN" ]; then
    echo "ERROR: Toolchain not found: $TOOLCHAIN"
    echo "  Make sure you're using NDK r23+"
    exit 1
fi

# ─── Clean ───
rm -rf "$BUILD_DIR"

# ─── Configure ───
echo ""
echo "============================================"
echo " Configuring (arm64-v8a, $BUILD_TYPE)..."
echo "============================================"

cmake \
    -DCMAKE_TOOLCHAIN_FILE="$TOOLCHAIN" \
    -DANDROID_ABI=arm64-v8a \
    -DANDROID_PLATFORM=android-28 \
    -DCMAKE_BUILD_TYPE="$BUILD_TYPE" \
    -B "$BUILD_DIR" \
    -S "$SCRIPT_DIR"

# ─── Build ───
echo ""
echo "============================================"
echo " Building..."
echo "============================================"

cmake --build "$BUILD_DIR" --config "$BUILD_TYPE" -j$(nproc 2>/dev/null || echo 4)

# ─── Show result ───
SO_FILE="$BUILD_DIR/lib/arm64-v8a/libcontinuum_jni.so"

echo ""
echo "============================================"
echo " Build Complete!"
echo "============================================"

if [ -f "$SO_FILE" ]; then
    SIZE=$(ls -lh "$SO_FILE" | awk '{print $5}')
    echo "  Output: $SO_FILE"
    echo "  Size:   $SIZE"
    echo ""
    echo "✅ Copy this .so to your Android project:"
    echo "   app/src/main/jniLibs/arm64-v8a/libcontinuum_jni.so"
else
    # CMake might put it in a different location
    SO_FILE=$(find "$BUILD_DIR" -name "*.so" -type f | head -1)
    if [ -f "$SO_FILE" ]; then
        SIZE=$(ls -lh "$SO_FILE" | awk '{print $5}')
        echo "  Output: $SO_FILE"
        echo "  Size:   $SIZE"
    else
        echo "❌ .so file not found in build directory"
        exit 1
    fi
fi

# ─── APK lib directory ───
APK_LIB_DIR="$SCRIPT_DIR/app/src/main/jniLibs/arm64-v8a"
mkdir -p "$APK_LIB_DIR"
cp "$SO_FILE" "$APK_LIB_DIR/libcontinuum_jni.so"
echo ""
echo "✅ Copied to: $APK_LIB_DIR/libcontinuum_jni.so"
echo "   Ready for Gradle build!"
