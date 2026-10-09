# CMakeLists.txt
# SushiBLAS - https://github.com/SushiSystems/SushiBLAS
# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.
# Commercial use requires a licence from Sushi Systems.

# The tiers of the library, lowest first. The order is the one
# cmake/SushiBLASLayers.cmake declares; a tier is added there before it is added here.
add_subdirectory(foundation)
add_subdirectory(data)
add_subdirectory(kernels)
add_subdirectory(engine)
add_subdirectory(operations)
add_subdirectory(library)
