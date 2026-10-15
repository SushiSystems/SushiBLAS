# CMakeLists.txt
# SushiBLAS - https://github.com/SushiSystems/SushiBLAS
# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.
# Commercial use requires a licence from Sushi Systems.

# Test-only: GraphProbe, a graph a test owns so it can read what its record calls added.
sushiblas_add_module(
    NAME graph_probe
    LAYER engine
    INTERNAL
    DEPENDS
        engine
        test_runner)
