# CMakeLists.txt
# SushiBLAS - https://github.com/SushiSystems/SushiBLAS
# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.
# Commercial use requires a licence from Sushi Systems.

# The foundation tier: what every other module is written in.
add_subdirectory(core)
add_subdirectory(support)
add_subdirectory(geometry)
add_subdirectory(graph_primitives)
