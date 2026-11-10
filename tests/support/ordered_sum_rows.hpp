/**************************************************************************/
/* ordered_sum_rows.hpp                                                   */
/**************************************************************************/
/*                          This file is part of:                         */
/*                                SushiBLAS                               */
/*                https://github.com/SushiSystems/SushiBLAS               */
/*                         https://sushisystems.io                        */
/**************************************************************************/
/* Copyright (c) 2026-present Mustafa Garip & Sushi Systems               */
/*                                                                        */
/* Licensed under the Apache License, Version 2.0 (the "License"); you    */
/* may not use this file except in compliance with the License. You may   */
/* obtain a copy of the License at                                        */
/*                                                                        */
/*     https://www.apache.org/licenses/LICENSE-2.0                        */
/*                                                                        */
/* Unless required by applicable law or agreed to in writing, software    */
/* distributed under the License is distributed on an "AS IS" BASIS,      */
/* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or        */
/* implied. See the License for the specific language governing           */
/* permissions and limitations under the License.                         */
/**************************************************************************/

/**
 * @file ordered_sum_rows.hpp
 * @brief Declares the rows of the ordered_sum value tests: inputs, weights and expected sum.
 * @author Mustafa Garip
 */

#pragma once

#include <array>
#include <vector>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <complex>
#include <type_traits>
#include <gtest/gtest.h>
#include <sycl/sycl.hpp>
#include <SushiBLAS/elementwise/elementwise.hpp>

namespace SushiBLAS
{
    namespace Test
    {
        namespace OrderedSumRows
        {
            /// @brief Rows of every tensor; the 3 x 347 shape holds 1041 elements, a multiple of no
            ///        work-group size.
            constexpr int64_t K_ROWS = 3;

            /// @brief Columns of every tensor; 347 is prime.
            constexpr int64_t K_COLUMNS = 347;

            /// @brief The element count of every tensor.
            constexpr std::size_t K_ELEMENTS = static_cast<std::size_t>(K_ROWS * K_COLUMNS);

            /// @brief Weights that no float type holds exactly, so each product rounds.
            constexpr std::array<double, 5> K_WEIGHTS = {0.37, -1.9, 2.5, 0.011, -0.73};

            /// @brief Seed of the value generator; every run draws the same values.
            constexpr std::uint64_t K_SEED = 0x5EEDu;

            /// @brief Names whether T is one of the two complex element types.
            template<typename T>
            inline constexpr bool K_IS_COMPLEX =
                std::is_same_v<T, std::complex<float>> || std::is_same_v<T, std::complex<double>>;

            /// @brief Advances a linear congruential generator and returns its high bits.
            inline std::uint64_t next_random(std::uint64_t& state)
            {
                state = state * 6364136223846793005ull + 1442695040888963407ull;
                return state >> 11;
            }

            /// @brief Draws a value of about 100 in magnitude with two decimal digits.
            inline double next_real(std::uint64_t& state)
            {
                return static_cast<double>(static_cast<std::int64_t>(next_random(state) % 20001u) -
                                           10000) /
                       97.0;
            }

            /**
             * @brief Draws K_ELEMENTS values of T from the generator.
             * @tparam T One of the seven element types.
             */
            template<typename T>
            std::vector<T> draw_values(std::uint64_t& state)
            {
                std::vector<T> values(K_ELEMENTS);
                for (T& value : values)
                {
                    if constexpr (std::is_integral_v<T>)
                        value = static_cast<T>(next_random(state) << 20);
                    else if constexpr (K_IS_COMPLEX<T>)
                    {
                        const double real = next_real(state);
                        const double imaginary = next_real(state);
                        value = T(static_cast<typename T::value_type>(real),
                                  static_cast<typename T::value_type>(imaginary));
                    }
                    else
                        value = static_cast<T>(next_real(state));
                }
                return values;
            }

            /**
             * @brief Sums the inputs element by element in index order, in the accumulation type
             *        the operation names.
             * @tparam T One of the seven element types.
             */
            template <typename T>
            std::vector<T> serial_sum(const std::vector<std::vector<T>>& inputs,
                                      const std::vector<double>& weights)
            {
                // Contraction is off so a multiply and an add round apart, as in the operation.
                #pragma clang fp contract(off)
                std::vector<T> sums(K_ELEMENTS);
                for (std::size_t i = 0; i < K_ELEMENTS; ++i)
                {
                    if constexpr (std::is_integral_v<T>)
                    {
                        using Unsigned = std::make_unsigned_t<T>;
                        Unsigned sum = 0;
                        for (const std::vector<T>& input : inputs)
                            sum = static_cast<Unsigned>(sum + static_cast<Unsigned>(input[i]));
                        sums[i] = static_cast<T>(sum);
                    }
                    else if constexpr (K_IS_COMPLEX<T>)
                    {
                        using Real = typename T::value_type;
                        Real real_sum = 0;
                        Real imaginary_sum = 0;
                        for (std::size_t k = 0; k < inputs.size(); ++k)
                        {
                            real_sum += static_cast<Real>(weights[k]) * inputs[k][i].real();
                            imaginary_sum += static_cast<Real>(weights[k]) * inputs[k][i].imag();
                        }
                        sums[i] = T(real_sum, imaginary_sum);
                    }
                    else
                    {
                        using Accumulation =
                            std::conditional_t<std::is_same_v<T, sycl::half>, float, T>;
                        Accumulation sum = 0;
                        for (std::size_t k = 0; k < inputs.size(); ++k)
                            sum += static_cast<Accumulation>(weights[k]) *
                                   static_cast<Accumulation>(inputs[k][i]);
                        sums[i] = static_cast<T>(sum);
                    }
                }
                return sums;
            }

            /// @brief Returns whether two equally long vectors hold the same bytes.
            template<typename T>
            bool same_bits(const std::vector<T>& a, const std::vector<T>& b)
            {
                return a.size() == b.size() &&
                       std::memcmp(a.data(), b.data(), a.size() * sizeof(T)) == 0;
            }

            /// @brief Returns value as T, with a zero imaginary part for a complex T.
            template<typename T>
            T make_value(double value)
            {
                if constexpr (K_IS_COMPLEX<T>)
                    return T(static_cast<typename T::value_type>(value), 0);
                else
                    return static_cast<T>(value);
            }

            /**
             * @brief Names an element type and an input count: a row whose inputs are drawn from
             *        the seeded generator and whose expected sum is the serial sum.
             * @tparam Element One of the seven element types.
             * @tparam Inputs How many inputs the sum takes.
             */
            template<typename Element, std::size_t Inputs>
            struct SerialCase
            {
                /// @brief The element type.
                using Type = Element;

                /// @brief The inputs of the row, each of K_ELEMENTS values.
                static std::vector<std::vector<Element>> inputs()
                {
                    std::uint64_t state = K_SEED + Inputs;
                    std::vector<std::vector<Element>> host;
                    for (std::size_t k = 0; k < Inputs; ++k)
                        host.push_back(draw_values<Element>(state));
                    return host;
                }

                /// @brief The weights of the row: K_WEIGHTS repeated to the input count, or all 1
                ///        for an integer type.
                static std::vector<double> weights()
                {
                    std::vector<double> values(Inputs, 1.0);
                    if constexpr (!std::is_integral_v<Element>)
                    {
                        for (std::size_t k = 0; k < Inputs; ++k)
                            values[k] = K_WEIGHTS[k % K_WEIGHTS.size()];
                    }
                    return values;
                }

                /// @brief The sum the row expects: the serial sum of its inputs.
                static std::vector<Element> expected()
                {
                    return serial_sum<Element>(inputs(), weights());
                }
            };

            /**
             * @brief Names an element type: the row whose inputs 1 and 4 and whose weights 2 and 3
             *        give the sum 2 * 1 + 3 * 4 = 14 by hand, or 1 + 4 = 5 for an integer type.
             *
             * The expected value is specified, not computed, so it stays an oracle apart from the
             * serial sum.
             * @tparam Element One of the seven element types.
             */
            template<typename Element>
            struct HandWorkedCase
            {
                /// @brief The element type.
                using Type = Element;

                /// @brief The inputs of the row: every element of the first is 1, of the second 4.
                static std::vector<std::vector<Element>> inputs()
                {
                    return {std::vector<Element>(K_ELEMENTS, make_value<Element>(1.0)),
                            std::vector<Element>(K_ELEMENTS, make_value<Element>(4.0))};
                }

                /// @brief The weights of the row: 2 and 3, or 1 and 1 for an integer type.
                static std::vector<double> weights()
                {
                    if constexpr (std::is_integral_v<Element>)
                        return {1.0, 1.0};
                    else
                        return {2.0, 3.0};
                }

                /// @brief The sum the row expects, written by hand.
                static std::vector<Element> expected()
                {
                    if constexpr (std::is_integral_v<Element>)
                        return std::vector<Element>(K_ELEMENTS, make_value<Element>(5.0));
                    else
                        return std::vector<Element>(K_ELEMENTS, make_value<Element>(14.0));
                }
            };

            /// @brief The rows of one element type: one, two, five and the limit of inputs drawn
            ///        from the generator, and the hand-worked row.
            template<typename Element>
            using RowsOf =
                ::testing::Types<SerialCase<Element, 1>, SerialCase<Element, 2>,
                                 SerialCase<Element, 5>,
                                 SerialCase<Element, ElementwiseOps::K_MAX_ORDERED_SUM_INPUTS>,
                                 HandWorkedCase<Element>>;

            /// @brief Joins two type lists.
            template<typename First, typename Second>
            struct Joined;

            /// @brief Joins two type lists.
            template<typename... A, typename... B>
            struct Joined<::testing::Types<A...>, ::testing::Types<B...>>
            {
                /// @brief The joined list.
                using type = ::testing::Types<A..., B...>;
            };

            /// @brief Every element type at every row.
            using AllRows = typename Joined<
                typename Joined<
                    typename Joined<RowsOf<sycl::half>, RowsOf<float>>::type,
                    typename Joined<RowsOf<double>, RowsOf<std::complex<float>>>::type>::type,
                typename Joined<
                    typename Joined<RowsOf<std::complex<double>>, RowsOf<int32_t>>::type,
                    RowsOf<int64_t>>::type>::type;
        } // namespace OrderedSumRows
    } // namespace Test
} // namespace SushiBLAS
