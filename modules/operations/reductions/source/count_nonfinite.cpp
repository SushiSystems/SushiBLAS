/**************************************************************************/
/* count_nonfinite.cpp                                                    */
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
 * @file count_nonfinite.cpp
 * @brief Defines ReductionOps::count_nonfinite, which records the non-finite count reduction.
 * @author Mustafa Garip
 */

#include <vector>
#include <cstddef>
#include <cstdint>
#include <SushiBLAS/reduction/reduce_1d.hpp>
#include <SushiBLAS/reference/serial_reduce.hpp>
#include <SushiBLAS/reductions/reductions.hpp>
#include <SushiBLAS/reductions/dispatch.hpp>

namespace SushiBLAS
{
    using namespace SushiRuntime::Literals;

    namespace
    {
        /** @brief Maps an element to 1 when it is NaN or infinite, else 0. */
        struct NonFiniteIndicator
        {
            /** @brief Returns 1 for a NaN or an infinity; isfinite is false for both, see modules/operations/README.md. */
            template<typename T>
            [[nodiscard]] int64_t operator()(T x) const noexcept
            {
                return sycl::isfinite(x) ? int64_t{0} : int64_t{1};
            }
        };

        /** @brief Counts the non-finite elements of a tensor, on the device or on the host. */
        class CountNonFiniteDispatch final
        {
            public:
                /** @brief Submits the fixed-order SYCL count and returns its last event. */
                template<typename T>
                [[nodiscard]] sycl::event operator()(sycl::queue& q, const T* in, int64_t* out,
                                                     int64_t size,
                                                     Dispatch::ReductionScratch<int64_t> scratch,
                                                     const std::vector<sycl::event>& deps) const
                {
                    return Reduce::reduce<Reduce::ExactSum>(
                        q, in, out, static_cast<std::size_t>(size), scratch.data,
                        scratch.capacity, NonFiniteIndicator{}, Dispatch::ReduceIdentityFinalize{},
                        deps);
                }

                /** @brief Counts by one serial host scan in index order. */
                template<typename T>
                void reference(const T* in, int64_t* out, int64_t size) const
                {
                    Serial::reduce<Reduce::ExactSum>(in, out, static_cast<std::size_t>(size),
                                                     NonFiniteIndicator{},
                                                     Dispatch::ReduceIdentityFinalize{});
                }
        };
    } // namespace

    // The count goes through the counting seam with a predicate transform; see modules/operations/README.md.
    TaskHandle ReductionOps::count_nonfinite(const Tensor& t, Tensor& result)
    {
        Dispatch::require_storage(t, "math.count_nonfinite", "t");

        // The partial count is planned from the device the data lives on; see modules/operations/README.md.
        const std::size_t partials = Dispatch::planned_partials_for(t);

        return Dispatch::execute_counting_reduction(recorder_, t, result,
            "math.count_nonfinite", "math.count_nonfinite"_op, CountNonFiniteDispatch{}, partials);
    }
} // namespace SushiBLAS
