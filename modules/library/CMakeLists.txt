# CMakeLists.txt
# SushiBLAS - https://github.com/SushiSystems/SushiBLAS
# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.
# Commercial use requires a licence from Sushi Systems.

# The library tier: the project umbrella, then the one library a consumer links. The umbrella
# comes first, so its include root is recorded before sushiblas reads the list.
add_subdirectory(umbrella)
add_subdirectory(sushiblas)
