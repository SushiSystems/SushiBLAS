/**************************************************************************/
/* ordered_sum_element.hpp                                                */
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
 * @file ordered_sum_element.hpp
 * @brief Declares the element functor of ElementwiseOps::ordered_sum and the type its weights take.
 * @author Mustafa Garip
 */

#pragma once

#include <array>
#include <complex>
#include <cstddef>
#include <type_traits>
#include <sycl/sycl.hpp>
#include <SushiBLAS/elementwise/elementwise.hpp>
#include <SushiBLAS/elementwise/wrapping_math.hpp>

namespace SushiBLAS
{
    namespace Dispatch
    {
        namespace Detail
        {
            /** @brief Holds how many inputs an ordered_sum kernel carries by value. */
            inline constexpr std::size_t K_ORDERED_SUM_CAPACITY =
                ElementwiseOps::K_MAX_ORDERED_SUM_INPUTS;

            /**
             * @brief Names the type an ordered_sum weight is multiplied in for element type T.
             *
             * HALF multiplies in FLOAT32, a complex type in its real part's type, a real type in
             * itself; an integer type takes no weight and stores T. Only FLOAT64 and COMPLEX64
             * give double, so no other type puts a double in device code.
             * @tparam T One of the seven element types ordered_sum serves.
             */
            template<typename T>
            struct OrderedSumWeight
            {
                /** @brief The weight type. */
                using type = T;
            };

            /** @brief Multiplies a HALF weight in FLOAT32, the type a HALF sum accumulates in. */
            template<>
            struct OrderedSumWeight<sycl::half>
            {
                /** @brief The weight type. */
                using type = float;
            };

            /** @brief Multiplies a complex weight per component, in the real part's type. */
            template<typename Real>
            struct OrderedSumWeight<std::complex<Real>>
            {
                /** @brief The weight type. */
                using type = Real;
            };

            /** @brief Names OrderedSumWeight<T>::type. */
            template<typename T>
            using OrderedSumWeightType = typename OrderedSumWeight<T>::type;

            /**
             * @brief Computes `output[i] = ((0 + w0 * x0[i]) + w1 * x1[i]) + ...` for one type.
             * @tparam T One of the seven element types.
             * @see modules/operations/elementwise/README.md
             */
            template<typename T>
            struct OrderedSumElement
            {
                /** @brief Names the type each weight is held and multiplied in. */
                using Weight = OrderedSumWeightType<T>;

                /** @brief Holds the first element of each input, in index order. */
                std::array<const T*, K_ORDERED_SUM_CAPACITY> inputs;
                /** @brief Holds the weight of each input, rounded once to Weight on the host. */
                std::array<Weight, K_ORDERED_SUM_CAPACITY> weights;
                /** @brief Holds the first element of the output. */
                T* output;
                /** @brief Holds the number of inputs. */
                std::size_t input_count;

                /** @brief Writes element i, reading input 0 to input_count - 1 in that order. */
                void operator()(std::size_t i) const
                {
                    // Contraction is off: a multiply and an add round apart on host and device.
                    #pragma clang fp contract(off)
                    if constexpr (std::is_integral_v<T>)
                    {
                        T sum = T(0);
                        for (std::size_t k = 0; k < input_count; ++k)
                            sum = wrapping_add(sum, inputs[k][i]);
                        output[i] = sum;
                    }
                    else if constexpr (std::is_same_v<T, std::complex<float>> ||
                                       std::is_same_v<T, std::complex<double>>)
                    {
                        Weight real_sum = Weight(0);
                        Weight imaginary_sum = Weight(0);
                        for (std::size_t k = 0; k < input_count; ++k)
                        {
                            real_sum += weights[k] * inputs[k][i].real();
                            imaginary_sum += weights[k] * inputs[k][i].imag();
                        }
                        output[i] = T(real_sum, imaginary_sum);
                    }
                    else
                    {
                        Weight sum = Weight(0);
                        for (std::size_t k = 0; k < input_count; ++k)
                            sum += weights[k] * static_cast<Weight>(inputs[k][i]);
                        output[i] = static_cast<T>(sum);
                    }
                }
            };
        } // namespace Detail
    } // namespace Dispatch
} // namespace SushiBLAS
