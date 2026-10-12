/**************************************************************************/
/* rotg.cpp                                                               */
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
 * @file rotg.cpp
 * @brief Defines Level1::rotg, which records the BLAS ROTG task.
 * @author Mustafa Garip
 */

#include <SushiBLAS/blas/level1.hpp>

#include <vector>
#include <complex>
#include <SushiBLAS/blas_kernels/blas.hpp>
#include <SushiBLAS/blas/blas_task.hpp>
#include <SushiBLAS/operands/scalar_result.hpp>
#include <SushiBLAS/engine/operand_set.hpp>
#include <SushiBLAS/operands/blas_metadata.hpp>
#include <SushiBLAS/blas/scalar_operand.hpp>
#include <SushiBLAS/support/complex_arithmetic.hpp>
#include <SushiRuntime/core/operation_id.hpp>

namespace SushiBLAS
{
    using namespace SushiRuntime::Literals;

    TaskHandle Level1::rotg(Tensor& a, Tensor& b, Tensor& c, Tensor& s)
    {
        const char* const op = "blas.lvl1.rotg";

        Dispatch::require_scalar_operand(a, a.dtype, op, "a");
        Dispatch::require_scalar_operand(b, a.dtype, op, "b");
        Dispatch::require_scalar_operand(c, Internal::real_result_dtype(a.dtype), op, "c");
        Dispatch::require_scalar_operand(s, a.dtype, op, "s");

        // A complex ROTG reads b and leaves it; a real one overwrites it with z.
        OperandSet operands;
        operands.updates(a);
        if (Core::is_complex_dtype(a.dtype))
            operands.reads(b);
        else
            operands.updates(b);
        operands.writes(c).writes(s);

        const Internal::BLASMetadata metadata(op, "blas.lvl1.rotg"_op);

        void* rw_a = a.data_or_null();
        void* rw_b = b.data_or_null();
        void* write_c = c.data_or_null();
        void* write_s = s.data_or_null();

        return Internal::execute_blas_task(recorder_, metadata.get(), operands, a.dtype,
            [=](auto scalar_type, const KernelPairDispatcher& dispatcher,
                sycl::queue& q, const std::vector<sycl::event>& deps) -> sycl::event
            {
                using T = decltype(scalar_type);
                auto pa = static_cast<T*>(rw_a);
                auto pb = static_cast<T*>(rw_b);
                auto pc = static_cast<Kernels::RealType<T>*>(write_c);
                auto ps = static_cast<T*>(write_s);

                return dispatcher.run(q, deps,
                    [=] { Kernels::Reference::rotg<T>(pa, pb, pc, ps); },
                    [=](sycl::queue& q2, const std::vector<sycl::event>& deps2) -> sycl::event
                    {
                        return Kernels::SYCL::rotg<T>(q2, pa, pb, pc, ps, deps2);
                    });
            });
    }
} // namespace SushiBLAS
