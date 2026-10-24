/**************************************************************************/
/* sushiblas_adapter.cpp                                                  */
/**************************************************************************/
/*                          This file is part of:                         */
/*                                SushiBLAS                               */
/*                https://github.com/SushiSystems/SushiBLAS               */
/*                         https://sushisystems.io                        */
/**************************************************************************/
/* Copyright (c) 2026-present Mustafa Garip & Sushi Systems               */
/*                                                                        */
/* Licensed under the PolyForm Noncommercial License 1.0.0 (the           */
/* "License"); you may not use this file except in compliance with the    */
/* License. You may obtain a copy of the License at                       */
/*                                                                        */
/*     https://polyformproject.org/licenses/noncommercial/1.0.0           */
/*                                                                        */
/* Noncommercial use is free. Commercial use requires a separate licence  */
/* from Sushi Systems; see COMMERCIAL.md. The software is provided        */
/* "as is", without warranty of any kind.                                 */
/**************************************************************************/

/**
 * @file sushiblas_adapter.cpp
 * @brief Implements the host program's backend and its operations on SushiBLAS.
 * @author Mustafa Garip
 */

#include "host_program.hpp"
#include "status_mapping.hpp"

#include <cstdint>
#include <utility>

#include <SushiBLAS/SushiBLAS.hpp>

namespace HostProgram
{
    /** @brief Owns the runtime and the engine the host's operations record on. */
    class Backend final
    {
        public:
            /** @brief Takes the runtime and builds the engine on it. */
            explicit Backend(SushiRuntime::Runtime&& created)
                : runtime(std::move(created)), engine(runtime)
            {

            }

            /// @brief The runtime the engine is built on; declared first, destroyed last.
            SushiRuntime::Runtime runtime;
            /// @brief The engine every operation of the host records on.
            SushiBLAS::Engine engine;
    };

    namespace
    {
        using Detail::guarded;
        using Detail::status_of_error;

        /** @brief Returns a tensor of the matrix's shape that holds its elements. */
        [[nodiscard]] SushiBLAS::Tensor tensor_of(SushiBLAS::Engine& engine, const Matrix& matrix)
        {
            const auto rows = static_cast<std::int64_t>(matrix.rows);
            const auto columns = static_cast<std::int64_t>(matrix.columns);
            SushiBLAS::Tensor tensor = engine.create_tensor({rows, columns});
            SushiBLAS::IO().from_vector(tensor, matrix.values);
            return tensor;
        }
    } // namespace

    void BackendDeleter::operator()(Backend* backend) const noexcept
    {
        delete backend;
    }

    Status create_backend(BackendHandle& backend)
    {
        return guarded([&]()
        {
            SushiRuntime::Result<SushiRuntime::Runtime> created = SushiRuntime::Runtime::create();
            if (!created)
                return status_of_error(created.error());

            backend.reset(new Backend(std::move(created.value())));
            return Status{};
        });
    }

    Status multiply_add(Backend& backend, const Matrix& a, const Matrix& b, const Matrix& offset,
                        Matrix& result, float& result_sum)
    {
        SushiBLAS::Engine& engine = backend.engine;
        const Status status = guarded([&]()
        {
            // The graph lives for this call, so a record call that throws leaves nothing behind.
            SushiRuntime::Graph graph = backend.runtime.graph();
            SushiBLAS::LateCountHold late_counts(backend.runtime);
            SushiBLAS::TaskRecorder recorder(graph.recorder(), engine.scratch_pool(),
                                             late_counts, engine.mode());
            const SushiBLAS::Tensor left = tensor_of(engine, a);
            const SushiBLAS::Tensor right = tensor_of(engine, b);
            const SushiBLAS::Tensor shift = tensor_of(engine, offset);
            const auto rows = static_cast<std::int64_t>(a.rows);
            const auto columns = static_cast<std::int64_t>(b.columns);
            SushiBLAS::Tensor product = engine.create_tensor({rows, columns});
            SushiBLAS::Tensor shifted = engine.create_tensor({rows, columns});
            SushiBLAS::Tensor total = engine.create_tensor({1});

            SushiBLAS::BLASOps(recorder).gemm(left, right, product);
            SushiBLAS::ElementwiseOps(recorder).add(product, shift, shifted);
            SushiBLAS::ReductionOps(recorder).sum(shifted, total);

            const SushiRuntime::Result<SushiRuntime::RunReport> ran = graph.run();
            if (!ran)
                return status_of_error(ran.error());

            SushiBLAS::IO io;
            result = Matrix{a.rows, b.columns, io.to_vector<float>(shifted)};
            result_sum = io.to_vector<float>(total)[0];
            return Status{};
        });

        return status;
    }
} // namespace HostProgram
