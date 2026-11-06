# Runtime.cmake
# SushiBLAS - https://github.com/SushiSystems/SushiBLAS
# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under the Apache License 2.0. See LICENSE.

# Runtime.cmake: SushiRuntime as SushiRuntime::SushiRuntime.
#
# Single responsibility: find the SushiRuntime package. The CLI builds or reuses it and passes
# SushiRuntime_DIR; nothing is built in this tree and nothing is downloaded.

# SushiRuntime as a package. The CLI builds or reuses it and passes SushiRuntime_DIR.
find_package(SushiRuntime CONFIG REQUIRED)
message(STATUS "Found SushiRuntime ${SushiRuntime_VERSION}: ${SushiRuntime_DIR}")
