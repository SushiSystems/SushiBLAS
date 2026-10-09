/**************************************************************************/
/* centred_reduction.hpp                                                  */
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
 * @file centred_reduction.hpp
 * @brief Declares the dispatcher of a reduction that centres on a first fold: variance, standard deviation and logsumexp.
 * @author Mustafa Garip
 */

#pragma once

#include <vector>
#include <cstddef>
#include <cstdint>
#include <sycl/sycl.hpp>
#include <SushiBLAS/support/reduce_1d.hpp>
#include <SushiBLAS/support/serial_reduce.hpp>
#include <SushiBLAS/reductions/dispatch.hpp>

namespace SushiBLAS
{
    namespace Dispatch
    {
        /** @brief Finalizes a sum into a mean. */
        struct MeanOfSum
        {
            /** @brief Returns sum divided by n. */
            template<typename W>
            [[nodiscard]] W operator()(W sum, std::size_t n) const noexcept
            {
                return sum / static_cast<W>(n);
            }
        };

        /** @brief Maps an element to its squared distance from a centre read on the device. */
        template<typename W>
        struct SquaredDeviation
        {
            /** @brief The centre; stage one writes it before stage two reads it. */
            const W* centre;

            /** @brief Returns `(x - *centre)^2` in the accumulation type. */
            template<typename X>
            [[nodiscard]] W operator()(X x) const noexcept
            {
                const W deviation = static_cast<W>(x) - *centre;
                return deviation * deviation;
            }
        };

        /** @brief Finalizes a sum of squared deviations into the population variance. */
        template<typename W>
        struct MeanOfDeviations
        {
            /** @brief The centre; not read here, held so every finalizer is built the same way. */
            const W* centre;

            /** @brief Returns sum divided by n. */
            [[nodiscard]] W operator()(W sum, std::size_t n) const noexcept
            {
                return sum / static_cast<W>(n);
            }
        };

        /** @brief Finalizes a sum of squared deviations into the population standard deviation. */
        template<typename W>
        struct RootOfMeanDeviations
        {
            /** @brief The centre; not read here, held so every finalizer is built the same way. */
            const W* centre;

            /** @brief Returns the square root of sum divided by n. */
            [[nodiscard]] W operator()(W sum, std::size_t n) const noexcept
            {
                return sycl::sqrt(sum / static_cast<W>(n));
            }
        };

        /** @brief Maps an element to `exp(x - *centre)`, and to exactly 1 where x equals the centre. */
        template<typename W>
        struct ExpBelowCentre
        {
            /** @brief The maximum; stage one writes it before stage two reads it. */
            const W* centre;

            /** @brief Returns 1 for the maximum itself, so an infinite maximum does not give NaN. */
            template<typename X>
            [[nodiscard]] W operator()(X x) const noexcept
            {
                const W value = static_cast<W>(x);
                const W top = *centre;
                return value == top ? W(1) : sycl::exp(value - top);
            }
        };

        /** @brief Finalizes a sum of exponentials into `*centre + log(sum)`. */
        template<typename W>
        struct LogPlusCentre
        {
            /** @brief The maximum stage one wrote. */
            const W* centre;

            /** @brief Returns the centre plus the logarithm of sum. */
            [[nodiscard]] W operator()(W sum, std::size_t) const noexcept
            {
                return *centre + sycl::log(sum);
            }
        };

        /**
         * @brief Runs a reduction in two fixed-order stages: a centre, then a compensated sum of
         *        terms taken against it.
         * @tparam CentreAccum The Reduce accumulator of stage one.
         * @tparam CentreFinalize Callable `(W value, std::size_t n) -> W` that yields the centre.
         * @tparam Term Functor template built as `Term<W>{centre}`, the transform of stage two.
         * @tparam Finalize Functor template built as `Finalize<W>{centre}`, the finalizer of stage two.
         */
        template<template<typename> class CentreAccum, typename CentreFinalize,
                 template<typename> class Term, template<typename> class Finalize>
        class CentredReductionDispatch final
        {
            public:
                /**
                 * @brief Submits both stages and returns the event of the second.
                 * @pre scratch.capacity is at least 1: one centre, then the partials of both stages.
                 */
                template<typename T, typename W>
                [[nodiscard]] sycl::event operator()(sycl::queue& q, const T* in, T* out,
                                                     int64_t size, ReductionScratch<W> scratch,
                                                     const std::vector<sycl::event>& deps) const
                {
                    const std::size_t n = static_cast<std::size_t>(size);
                    W* centre = scratch.data;
                    W* partials = scratch.data + 1;
                    const std::size_t capacity = scratch.capacity - 1;

                    const sycl::event centred = Reduce::reduce<CentreAccum>(
                        q, in, centre, n, partials, capacity,
                        ReduceIdentityTransform{}, CentreFinalize{}, deps);

                    return Reduce::reduce<Reduce::CompensatedSum>(
                        q, in, out, n, partials, capacity,
                        Term<W>{centre}, Finalize<W>{centre},
                        std::vector<sycl::event>{centred});
                }

                /** @brief Runs both stages as serial host folds, the centre held in the type stage two sums in. */
                template<typename T>
                void reference(const T* in, T* out, int64_t size) const
                {
                    using Wide = typename Serial::Fold<Reduce::CompensatedSum, T>::Wide;

                    const std::size_t n = static_cast<std::size_t>(size);
                    Wide centre{};

                    Serial::reduce<CentreAccum>(in, &centre, n, ReduceIdentityTransform{},
                                                CentreFinalize{});
                    Serial::reduce<Reduce::CompensatedSum>(in, out, n, Term<Wide>{&centre},
                                                           Finalize<Wide>{&centre});
                }
        };
    } // namespace Dispatch
} // namespace SushiBLAS
