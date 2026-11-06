# SushiBLASSushiLog.cmake
# SushiBLAS - https://github.com/SushiSystems/SushiBLAS
# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under the Apache License 2.0. See LICENSE.

# SushiBLASSushiLog.cmake: SushiLog, the one logger of the process, as SushiLog::SushiLog.
#
# Single responsibility: find the SushiLog package, so SushiBLAS and SushiRuntime link the one
# SushiLog the CLI pinned in cli/sushistack.deps.toml.

# SushiLog, the one logger of the process. The CLI builds or reuses its package and passes
# SushiLog_DIR; nothing is built in this tree and nothing is downloaded.
find_package(SushiLog CONFIG REQUIRED)
message(STATUS "Found SushiLog ${SushiLog_VERSION}: ${SushiLog_DIR}")
