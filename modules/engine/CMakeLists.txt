# CMakeLists.txt
# SushiBLAS - https://github.com/SushiSystems/SushiBLAS
# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.
# Commercial use requires a licence from Sushi Systems.

# The engine tier: the recorder, the execution mode and what every operation family shares.
add_subdirectory(engine)
add_subdirectory(operands)
add_subdirectory(functor_map)
