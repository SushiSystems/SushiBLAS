/**************************************************************************/
/* blas_task.hpp                                                          */
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
 * @file blas_task.hpp
 * @brief Declares execute_blas_task, which records one dtype-dispatched BLAS task.
 * @author Mustafa Garip
 */

#pragma once

#include <utility>
#include <SushiBLAS/core/common.hpp>
#include <SushiBLAS/engine/dispatch.hpp>
#include <SushiBLAS/engine/operand_set.hpp>
#include <SushiBLAS/engine/task.hpp>
#include <SushiBLAS/engine/task_recorder.hpp>

namespace SushiBLAS
{
    namespace Internal
    {
        /**
         * @brief Records one dtype-dispatched BLAS task whose body runs a reference and SYCL pair.
         * @param op_func Callable taking a value of the element type, a const
         *        KernelPairDispatcher&, the queue and the dependencies.
         * @throws std::invalid_argument at record time on a dtype outside Dom.
         * @throws UnsupportedExecution at record time, as TaskRecorder::record_pair states.
         * @see modules/operations/blas/README.md
         */
        template<Dispatch::Domain Dom = Dispatch::Domain::RealAndComplex, typename Func>
        TaskHandle execute_blas_task(TaskRecorder& recorder,
                                     const SushiRuntime::Internal::Graph::TaskMetadata& meta,
                                     const OperandSet& operands, Core::DataType dtype,
                                     Func&& op_func)
        {
            return execute_typed_task<Dom>(recorder, meta, operands, dtype, "BLAS operation",
                                           std::forward<Func>(op_func));
        }
    } // namespace Internal
} // namespace SushiBLAS
