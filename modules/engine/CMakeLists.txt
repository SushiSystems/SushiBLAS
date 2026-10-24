# CMakeLists.txt
# SushiBLAS - https://github.com/SushiSystems/SushiBLAS
# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.
# Commercial use requires a licence from Sushi Systems.

# The engine tier: the recorder, the execution mode and what every operation family shares.
add_subdirectory(engine)
add_subdirectory(operands)
add_subdirectory(functor_map)
if(SB_ENABLE_DISTRIBUTED)
    add_subdirectory(cluster)
endif()

# Test-only modules: a test program includes them, the library never does.
# The recording graph serves the test fixtures and the benchmark programs, and nothing else.
if(SB_BUILD_TESTS OR SB_BUILD_BENCHMARKS)
    add_subdirectory(recording_graph)
endif()

if(SB_BUILD_TESTS)
    add_subdirectory(engine_fixture)
    add_subdirectory(execution_mode_sweep)
    add_subdirectory(graph_probe)
endif()
