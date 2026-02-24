#!/bin/bash

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
cd "$SCRIPT_DIR"

if [ -f "config.cfg" ]; then
    while IFS='=' read -r key value; do
        [[ $key =~ ^#.* ]] || [[ -z $key ]] && continue
        export "$key=$value"
    done < config.cfg
fi

if [ -f "$ONEAPI_ROOT/setvars.sh" ]; then
    source "$ONEAPI_ROOT/setvars.sh" --force > /dev/null 2>&1
fi

cd ../..

CMAKE_ARGS="-B build -G Ninja -DCMAKE_BUILD_TYPE=Release"

if [ "$USE_VCPKG" = "true" ]; then
    echo "[INFO] Using vcpkg from $VCPKG_ROOT"
    CMAKE_ARGS="$CMAKE_ARGS -DCMAKE_TOOLCHAIN_FILE=$VCPKG_ROOT/scripts/buildsystems/vcpkg.cmake"
else
    echo "[INFO] Using system packages (apt/yum/etc.)"
fi

if [ ! -f "build/build.ninja" ]; then
    echo "[INFO] Configuring CMake..."
    rm -rf build
    cmake $CMAKE_ARGS \
        -DCMAKE_CXX_COMPILER=icpx \
        -DCMAKE_C_COMPILER=icx \
        -DCMAKE_CXX_FLAGS="-fsycl"
fi

echo "[INFO] Building..."
cmake --build build --config Release

if [ $? -eq 0 ]; then
    echo "[SUCCESS] Build completed successfully!"
else
    echo "[ERROR] Build failed."
fi
