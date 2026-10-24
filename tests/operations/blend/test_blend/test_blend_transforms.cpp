/**************************************************************************/
/* test_blend_transforms.cpp                                              */
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

/**
 * @file test_blend_transforms.cpp
 * @brief Tests BlendOps::blend_transforms in both forced modes against hand values, an
 * independent formula and a rounding bound, and its refusals.
 * @author Mustafa Garip
 */

#include <cmath>
#include <limits>
#include <string>
#include <string_view>
#include <vector>
#include <cstdint>
#include <cstddef>
#include <cstring>
#include <stdexcept>
#include <type_traits>
#include <gtest/gtest.h>
#include <sycl/sycl.hpp>
#include <SushiBLAS/SushiBLAS.hpp>
#include <SushiBLAS/blend/blend.hpp>
#include <SushiBLAS/arithmetic/accumulator.hpp>
#include <SushiBLAS/engine_fixture/sushiblas_test.hpp>
#include <SushiBLAS/tensor_expectations/tensor_comparison.hpp>
#include <SushiBLAS/tensor_expectations/scalar_dtype.hpp>
#include <SushiBLAS/typed_suite_naming/default_type_names.hpp>
#include <SushiBLAS/test_runner/test_environment.hpp>
#include <SushiBLAS/blas_test_oracle/blas_truth.hpp>

namespace SushiBLAS
{
    namespace Test
    {
        namespace
        {
            /// @brief The forced modes every case runs in.
            constexpr ExecutionMode K_FORCED_MODES[] = {ExecutionMode::ForceSYCL,
                                                        ExecutionMode::ForceReference};

            /// @brief The value an output holds before its operation runs.
            constexpr double K_SENTINEL = 7.0;

            /// @brief The largest magnitude up to which HALF holds every integer.
            constexpr double K_HALF_EXACT_INTEGER_LIMIT = 2048.0;

            /// @brief The weights that sum to one and that every type holds exactly.
            constexpr double K_DYADIC_WEIGHTS[4] = {0.5, 0.25, 0.125, 0.125};

            /// @brief How the weights of a case are drawn.
            enum class WeightKind
            {
                Integer,    /**< Integers from -2 to 2, which do not sum to one. */
                SumToOne,   /**< Binary fractions of one influence or four, which sum to one. */
                Fractional  /**< Values no type holds exactly, which do not sum to one. */
            };

            /// @brief Holds one set of extents the operation is run on.
            struct BlendCase
            {
                const char* id = "";     /**< The case's name in a failure message. */
                int64_t points = 0;      /**< P. */
                int64_t components = 0;  /**< The values per point, 3 or 4. */
                int64_t rows = 0;        /**< The rows per transform, 3 or 4. */
                int64_t transforms = 0;  /**< T. */
                int64_t influences = 0;  /**< J. */
            };

            /// @brief Holds the host values of one case, as doubles.
            struct BlendData
            {
                std::vector<double> points;   /**< `[P][components]`. */
                std::vector<double> matrices; /**< `[T][rows][4]`. */
                std::vector<double> weights;  /**< `[P][J]`. */
                std::vector<int64_t> indices; /**< `[P][J]`, every one inside `[0, T)`. */
            };

            /**
             * @brief Returns the cases every dtype and mode runs: 3x4 and 4x4 transforms, J = 1 and
             *        J = 4.
             */
            std::vector<BlendCase> table_cases()
            {
                return {
                    {"position, 3x4, J = 1", 7, 3, 3, 3, 1},
                    {"position, 4x4, J = 4", 9, 3, 4, 4, 4},
                    {"homogeneous, 3x4, J = 4", 6, 4, 3, 5, 4},
                    {"homogeneous, 4x4, J = 1", 5, 4, 4, 2, 1},
                    {"position, 3x4, J = 4", 33, 3, 3, 6, 4},
                };
            }

            /** @brief Returns the name a failure message gives a case run in mode. */
            std::string trace_of(const BlendCase& c, ExecutionMode mode)
            {
                return std::string(c.id) +
                       (mode == ExecutionMode::ForceReference ? " ForceReference" : " ForceSYCL");
            }

            /** @brief Returns how many elements a tensor of dims holds. */
            std::size_t element_count(const std::vector<int64_t>& dims) noexcept
            {
                int64_t count = 1;
                for (const int64_t extent : dims)
                    count *= extent;
                return static_cast<std::size_t>(count);
            }

            /** @brief Returns `((i * multiplier) % modulus) - shift` for i below count. */
            std::vector<double> residue_pattern(std::size_t count, int64_t multiplier,
                                                int64_t modulus, int64_t shift)
            {
                std::vector<double> values(count);
                for (std::size_t i = 0; i < count; ++i)
                    values[i] = static_cast<double>(
                        (static_cast<int64_t>(i) * multiplier) % modulus - shift);
                return values;
            }

            /** @brief Returns `0.37 * v + 0.011` for every v; no element type holds it exactly. */
            std::vector<double> fractional(std::vector<double> values)
            {
                for (double& v : values)
                    v = 0.37 * v + 0.011;
                return values;
            }

            /** @brief Returns values rounded to T. */
            template<typename T>
            std::vector<T> elements_from(const std::vector<double>& values)
            {
                std::vector<T> elements(values.size());
                for (std::size_t i = 0; i < values.size(); ++i)
                {
                    if constexpr (std::is_same_v<T, sycl::half>)
                        elements[i] = sycl::half(static_cast<float>(values[i]));
                    else
                        elements[i] = static_cast<T>(values[i]);
                }
                return elements;
            }

            /** @brief Returns elements widened to double. */
            template<typename T>
            std::vector<double> widened(const std::vector<T>& elements)
            {
                std::vector<double> values(elements.size());
                for (std::size_t i = 0; i < elements.size(); ++i)
                    values[i] = Detail::to_double(elements[i]);
                return values;
            }

            /**
             * @brief Returns the host values of a case under a kind of weights.
             * @note Fractional weights come with fractional points and matrices, rounded to T.
             */
            template<typename T>
            BlendData data_for(const BlendCase& c, WeightKind kind)
            {
                const std::size_t point_values = static_cast<std::size_t>(c.points * c.components);
                const std::size_t matrix_values =
                    static_cast<std::size_t>(c.transforms * c.rows * 4);
                const std::size_t influence_count =
                    static_cast<std::size_t>(c.points * c.influences);

                BlendData data;
                data.points = residue_pattern(point_values, 37, 23, 11);
                data.matrices = residue_pattern(matrix_values, 13, 7, 3);

                if (kind == WeightKind::Integer)
                {
                    data.weights = residue_pattern(influence_count, 3, 5, 2);
                }
                else if (kind == WeightKind::SumToOne)
                {
                    data.weights.resize(influence_count);
                    for (std::size_t i = 0; i < influence_count; ++i)
                    {
                        // Point p = i / J takes the dyadic weights rotated by p.
                        const std::size_t rotation = i / static_cast<std::size_t>(c.influences);
                        data.weights[i] =
                            c.influences == 1 ? 1.0 : K_DYADIC_WEIGHTS[(i + rotation) % 4];
                    }
                }
                else
                {
                    data.points = widened(elements_from<T>(fractional(data.points)));
                    data.matrices = widened(elements_from<T>(fractional(data.matrices)));
                    data.weights = widened(elements_from<T>(
                        fractional(residue_pattern(influence_count, 3, 5, 2))));
                }

                data.indices.resize(influence_count);
                for (std::size_t i = 0; i < influence_count; ++i)
                    data.indices[i] = static_cast<int64_t>(
                        (i * 7 + 3) % static_cast<std::size_t>(c.transforms));
                return data;
            }

            /**
             * @brief Returns a dense tensor of dims holding values rounded to T.
             * @param values One value per element.
             */
            template<typename T>
            Tensor tensor_from(Engine& engine, const std::vector<int64_t>& dims,
                               const std::vector<double>& values)
            {
                Tensor t = engine.create_tensor(
                    SushiRuntime::span<const int64_t>(dims.data(), dims.data() + dims.size()),
                    sushiblas_dtype<T>());

                if (t.num_elements > 0)
                    SushiBLAS::IO().from_vector(t, elements_from<T>(values));
                return t;
            }

            /** @brief Returns a dense tensor of dims holding indices in the integer type Index. */
            template<typename Index>
            Tensor index_tensor(Engine& engine, const std::vector<int64_t>& dims,
                                const std::vector<int64_t>& values)
            {
                const Core::DataType dtype = sushiblas_dtype<Index>();
                Tensor t = engine.create_tensor(
                    SushiRuntime::span<const int64_t>(dims.data(), dims.data() + dims.size()),
                    dtype);

                if (t.num_elements > 0)
                {
                    std::vector<Index> converted(values.size());
                    for (std::size_t i = 0; i < values.size(); ++i)
                        converted[i] = static_cast<Index>(values[i]);
                    SushiBLAS::IO().from_vector(t, converted);
                }
                return t;
            }

            /** @brief Returns a dense tensor of dims with every element at K_SENTINEL. */
            template<typename T>
            Tensor sentinel_tensor(Engine& engine, const std::vector<int64_t>& dims)
            {
                return tensor_from<T>(engine, dims,
                                      std::vector<double>(element_count(dims), K_SENTINEL));
            }

            /** @brief Returns a dense tensor of dims whose elements are all one. */
            template<typename T>
            Tensor filled_tensor(Engine& engine, const std::vector<int64_t>& dims)
            {
                return tensor_from<T>(engine, dims, std::vector<double>(element_count(dims), 1.0));
            }

            /** @brief Returns a dense INT32 tensor of dims whose elements are all zero. */
            Tensor zero_indices(Engine& engine, const std::vector<int64_t>& dims)
            {
                return index_tensor<int32_t>(
                    engine, dims, std::vector<int64_t>(element_count(dims), 0));
            }

            /** @brief Holds the operands of one case as tensors. */
            struct BlendTensors
            {
                Tensor points;
                Tensor transforms;
                Tensor indices;
                Tensor weights;
            };

            /** @brief Returns the operands of a case, the indices stored as Index. */
            template<typename T, typename Index = int32_t>
            BlendTensors tensors_for(Engine& engine, const BlendCase& c, const BlendData& data)
            {
                return BlendTensors{
                    tensor_from<T>(engine, {c.points, c.components}, data.points),
                    tensor_from<T>(engine, {c.transforms, c.rows, 4}, data.matrices),
                    index_tensor<Index>(engine, {c.points, c.influences}, data.indices),
                    tensor_from<T>(engine, {c.points, c.influences}, data.weights)};
            }

            /** @brief Returns the values the library writes for the operands of a case. */
            template<typename T>
            std::vector<T> blend_of(Engine& engine, const BlendCase& c, const BlendTensors& in)
            {
                Tensor out = sentinel_tensor<T>(engine, {c.points, c.components});
                RecordingFixture graph(engine);
                SushiBLAS::BlendOps(graph.recorder())
                    .blend_transforms(in.points, in.transforms, in.indices, in.weights, out);
                graph.run().value();
                return SushiBLAS::IO().to_vector<T>(out);
            }

            /**
             * @brief Returns value `column` of a point as a homogeneous vector: a position has w =
             *        1.
             */
            double homogeneous_value(const BlendCase& c, const BlendData& data, int64_t point,
                                     int64_t column)
            {
                if (column >= c.components)
                    return 1.0;
                return data.points[static_cast<std::size_t>(point * c.components + column)];
            }

            /**
             * @brief Returns entry `[row][column]` of a transform; a 3-row transform ends in `(0,
             *        0, 0, 1)`.
             */
            double matrix_entry(const BlendCase& c, const BlendData& data, int64_t slot,
                                int64_t row, int64_t column)
            {
                if (row >= c.rows)
                    return column == 3 ? 1.0 : 0.0;
                return data.matrices[static_cast<std::size_t>((slot * c.rows + row) * 4 + column)];
            }

            /**
             * @brief Returns the term one influence adds to one output value: the weight times
             *        the row of the transform times the point.
             * @param slot The transform the influence names; the caller has checked it is in range.
             * @note It is written apart from the kernels, with no shortcut for the implicit row.
             */
            double influence_term(const BlendCase& c, const BlendData& data, int64_t point,
                                  int64_t value, int64_t slot, double weight)
            {
                double moved = 0.0;
                for (int64_t column = 0; column < 4; ++column)
                    moved += matrix_entry(c, data, slot, value, column) *
                             homogeneous_value(c, data, point, column);
                return weight * moved;
            }

            /** @brief Returns the blend of the case by the formula, one double per output value. */
            std::vector<double> blend_by_formula(const BlendCase& c, const BlendData& data)
            {
                std::vector<double> out(static_cast<std::size_t>(c.points * c.components), 0.0);

                for (int64_t p = 0; p < c.points; ++p)
                {
                    for (int64_t i = 0; i < c.components; ++i)
                    {
                        for (int64_t j = 0; j < c.influences; ++j)
                        {
                            const std::size_t at = static_cast<std::size_t>(p * c.influences + j);
                            const int64_t slot = data.indices[at];
                            if (slot < 0 || slot >= c.transforms)
                                continue;

                            out[static_cast<std::size_t>(p * c.components + i)] +=
                                influence_term(c, data, p, i, slot, data.weights[at]);
                        }
                    }
                }
                return out;
            }

            /**
             * @brief Returns the exact sum of one output value and the sum of its products'
             *        magnitudes.
             * @note A product is a weight-times-matrix-entry times a point value. For FLOAT64 the
             *       first factor rounds once in double, which the bound counts.
             */
            TruthValue truth_of(const BlendCase& c, const BlendData& data, int64_t p, int64_t i)
            {
                TruthSum sum;

                for (int64_t j = 0; j < c.influences; ++j)
                {
                    const std::size_t at = static_cast<std::size_t>(p * c.influences + j);
                    const int64_t slot = data.indices[at];
                    if (slot < 0 || slot >= c.transforms)
                        continue;

                    for (int64_t column = 0; column < 4; ++column)
                    {
                        const double m = matrix_entry(c, data, slot, i, column);
                        if (m != 0.0)
                            sum.add_product(data.weights[at] * m,
                                            homogeneous_value(c, data, p, column));
                    }
                }
                return sum.value();
            }

            /**
             * @brief Returns the bound on a value's distance from its exact sum in one mode.
             * @param influences J, the number of terms the value sums.
             * @param truth The value's exact sum and the sum of its products' magnitudes.
             * @see tests/operations/blend/README.md
             */
            template<typename T>
            double blend_bound(ExecutionMode mode, int64_t influences, const TruthValue& truth)
            {
                const double store = unit_roundoff<T>() * std::fabs(truth.value);
                const double unit = mode == ExecutionMode::ForceSYCL
                                        ? unit_roundoff<Kernels::Accumulator<T>>()
                                        : unit_roundoff<double>();

                return rounding_bound(static_cast<std::size_t>(influences) + 5, unit,
                                      truth.magnitude) + store;
            }

            /** @brief Returns true when the test device can compute in T. */
            template<typename T>
            bool device_computes()
            {
                const sycl::device device = TestEnvironment::queue().get_device();

                if constexpr (std::is_same_v<T, sycl::half>)
                    return device.has(sycl::aspect::fp16);
                else if constexpr (std::is_same_v<T, double>)
                    return device.has(sycl::aspect::fp64);
                else
                    return true;
            }

            /**
             * @brief Expects record to throw std::invalid_argument naming the operation and
             *        operand.
             * @tparam Record Callable `() -> void` that records one malformed call.
             */
            template<typename Record>
            void expect_refusal(Record&& record, const char* operand)
            {
                try
                {
                    record();
                    ADD_FAILURE() << "blend.blend_transforms accepted a malformed '" << operand
                                  << "'";
                }
                catch (const std::invalid_argument& error)
                {
                    const std::string text = error.what();
                    EXPECT_NE(text.find("blend.blend_transforms"), std::string::npos) << text;
                    EXPECT_NE(text.find("'" + std::string(operand) + "'"), std::string::npos)
                        << text;
                }
            }
        } // namespace

        /**
         * @brief Runs the blend in one element type, skipped where the device lacks it.
         */
        template<typename T>
        class Medium_BlendTransformsTyped : public SushiBLASTest
        {
            protected:
                /** @brief Builds the engine and skips the case on a device without T's aspect. */
                void SetUp() override
                {
                    SushiBLASTest::SetUp();

                    if (!device_computes<T>())
                        GTEST_SKIP() << "the device does not report the aspect this dtype needs";
                }
        };

        using BlendTransformsTypes = ::testing::Types<sycl::half, float, double>;
        SB_TYPED_TEST_SUITE(Medium_BlendTransformsTyped, BlendTransformsTypes);

        /**
         * @brief Runs the blend cases that fix one element type.
         */
        class Medium_BlendTransforms : public SushiBLASTest
        {
        };

        /**
         * @brief Tests that a translation moves a position, and moves a homogeneous point only when
         *        w is 1.
         */
        TYPED_TEST(Medium_BlendTransformsTyped, ATranslationMovesPositionsAndLeavesDirectionsAlone)
        {
            using T = TypeParam;
            Engine& engine = *this->engine;

            for (const ExecutionMode mode : K_FORCED_MODES)
            {
                engine.set_mode(mode);

                const Tensor matrices = tensor_from<T>(engine, {1, 3, 4},
                                                       {1, 0, 0, 1, 0, 1, 0, 2, 0, 0, 1, 3});
                const Tensor indices = index_tensor<int32_t>(engine, {2, 1}, {0, 0});
                const Tensor weights = tensor_from<T>(engine, {2, 1}, {1, 1});

                // A position takes the translation; its output holds three values.
                const BlendCase positions{"hand positions", 2, 3, 3, 1, 1};
                const BlendTensors as_positions{
                    tensor_from<T>(engine, {2, 3}, {1, 1, 1, 0, 0, 0}), matrices, indices,
                    weights};
                EXPECT_EQ(widened(blend_of<T>(engine, positions, as_positions)),
                          (std::vector<double>{2, 3, 4, 1, 2, 3}))
                    << trace_of(positions, mode);

                // A homogeneous point keeps w, so w = 0 ignores the translation and w = 1 takes it.
                const BlendCase homogeneous{"hand homogeneous", 2, 4, 3, 1, 1};
                const BlendTensors as_homogeneous{
                    tensor_from<T>(engine, {2, 4}, {1, 2, 3, 0, 1, 2, 3, 1}), matrices, indices,
                    weights};
                EXPECT_EQ(widened(blend_of<T>(engine, homogeneous, as_homogeneous)),
                          (std::vector<double>{1, 2, 3, 0, 2, 4, 6, 1}))
                    << trace_of(homogeneous, mode);
            }
        }

        /**
         * @brief Tests that a 4x4 transform scales w, and that a position takes no division by it.
         */
        TYPED_TEST(Medium_BlendTransformsTyped, AFourByFourTransformActsOnAllFourValues)
        {
            using T = TypeParam;
            Engine& engine = *this->engine;

            for (const ExecutionMode mode : K_FORCED_MODES)
            {
                engine.set_mode(mode);

                const Tensor matrices = tensor_from<T>(
                    engine, {1, 4, 4}, {1, 0, 0, 1, 0, 1, 0, 2, 0, 0, 1, 3, 0, 0, 0, 2});
                const Tensor indices = index_tensor<int32_t>(engine, {1, 1}, {0});
                const Tensor weights = tensor_from<T>(engine, {1, 1}, {1});

                const BlendCase positions{"hand 4x4 position", 1, 3, 4, 1, 1};
                EXPECT_EQ(widened(blend_of<T>(engine, positions,
                                              {tensor_from<T>(engine, {1, 3}, {1, 1, 1}),
                                               matrices, indices, weights})),
                          (std::vector<double>{2, 3, 4}))
                    << trace_of(positions, mode);

                const BlendCase homogeneous{"hand 4x4 homogeneous", 1, 4, 4, 1, 1};
                EXPECT_EQ(widened(blend_of<T>(engine, homogeneous,
                                              {tensor_from<T>(engine, {1, 4}, {1, 1, 1, 5}),
                                               matrices, indices, weights})),
                          (std::vector<double>{6, 11, 16, 10}))
                    << trace_of(homogeneous, mode);
            }
        }

        /**
         * @brief Tests that two influences of weight one half give the average of their results.
         */
        TYPED_TEST(Medium_BlendTransformsTyped, TwoInfluencesGiveTheirWeightedSum)
        {
            using T = TypeParam;
            Engine& engine = *this->engine;

            for (const ExecutionMode mode : K_FORCED_MODES)
            {
                engine.set_mode(mode);

                // Transform 0 moves x by 2 and transform 1 moves y by 4.
                const Tensor matrices = tensor_from<T>(
                    engine, {2, 3, 4},
                    {1, 0, 0, 2, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 4, 0, 0, 1, 0});
                const Tensor indices = index_tensor<int32_t>(engine, {1, 2}, {0, 1});
                const Tensor weights = tensor_from<T>(engine, {1, 2}, {0.5, 0.5});

                const BlendCase c{"hand blend", 1, 3, 3, 2, 2};
                EXPECT_EQ(widened(blend_of<T>(engine, c,
                                              {tensor_from<T>(engine, {1, 3}, {0, 0, 0}),
                                               matrices, indices, weights})),
                          (std::vector<double>{1, 2, 0}))
                    << trace_of(c, mode);
            }
        }

        /**
         * @brief Tests that integer weights and data give the formula's sums exactly in every shape
         *        and mode.
         */
        TYPED_TEST(Medium_BlendTransformsTyped, IntegerDataEqualsTheFormulaExactly)
        {
            using T = TypeParam;
            Engine& engine = *this->engine;

            for (const BlendCase& c : table_cases())
            {
                const BlendData data = data_for<T>(c, WeightKind::Integer);
                const std::vector<double> expected = blend_by_formula(c, data);

                for (const double truth : expected)
                    ASSERT_LT(std::fabs(truth), K_HALF_EXACT_INTEGER_LIMIT) << c.id;

                const BlendTensors in = tensors_for<T>(engine, c, data);
                for (const ExecutionMode mode : K_FORCED_MODES)
                {
                    SCOPED_TRACE(trace_of(c, mode));
                    engine.set_mode(mode);

                    EXPECT_EQ(widened(blend_of<T>(engine, c, in)), expected);
                }
            }
        }

        /**
         * @brief Tests that weights summing to one give the formula's sums exactly, every value a
         *        multiple of one eighth.
         */
        TYPED_TEST(Medium_BlendTransformsTyped, WeightsThatSumToOneEqualTheFormulaExactly)
        {
            using T = TypeParam;
            Engine& engine = *this->engine;

            for (const BlendCase& c : table_cases())
            {
                const BlendData data = data_for<T>(c, WeightKind::SumToOne);
                const std::vector<double> expected = blend_by_formula(c, data);

                // A multiple of 1/8 below 2048/8 in magnitude is exact in every type.
                for (const double truth : expected)
                    ASSERT_LT(std::fabs(truth) * 8.0, K_HALF_EXACT_INTEGER_LIMIT) << c.id;

                const BlendTensors in = tensors_for<T>(engine, c, data);
                for (const ExecutionMode mode : K_FORCED_MODES)
                {
                    SCOPED_TRACE(trace_of(c, mode));
                    engine.set_mode(mode);

                    EXPECT_EQ(widened(blend_of<T>(engine, c, in)), expected);
                }
            }
        }

        /**
         * @brief Tests that identity transforms and weights summing to one give each point back.
         */
        TYPED_TEST(Medium_BlendTransformsTyped, IdentityTransformsWithUnitWeightsKeepThePoint)
        {
            using T = TypeParam;
            Engine& engine = *this->engine;

            for (const BlendCase& c : table_cases())
            {
                BlendData data = data_for<T>(c, WeightKind::SumToOne);
                data.matrices.assign(static_cast<std::size_t>(c.transforms * c.rows * 4), 0.0);
                for (int64_t t = 0; t < c.transforms; ++t)
                    for (int64_t d = 0; d < c.rows; ++d)
                        data.matrices[static_cast<std::size_t>((t * c.rows + d) * 4 + d)] = 1.0;

                const BlendTensors in = tensors_for<T>(engine, c, data);
                for (const ExecutionMode mode : K_FORCED_MODES)
                {
                    SCOPED_TRACE(trace_of(c, mode));
                    engine.set_mode(mode);

                    EXPECT_EQ(widened(blend_of<T>(engine, c, in)), data.points);
                }
            }
        }

        /**
         * @brief Tests that non-integer data and weights that do not sum to one stay within the
         *        rounding bound of the exact sum, in every shape and mode.
         */
        TYPED_TEST(Medium_BlendTransformsTyped, FractionalDataStaysWithinTheRoundingBound)
        {
            using T = TypeParam;
            Engine& engine = *this->engine;

            for (const BlendCase& c : table_cases())
            {
                // The library sees the rounded inputs, so the exact sum is formed from those.
                const BlendData data = data_for<T>(c, WeightKind::Fractional);
                const BlendTensors in = tensors_for<T>(engine, c, data);

                for (const ExecutionMode mode : K_FORCED_MODES)
                {
                    SCOPED_TRACE(trace_of(c, mode));
                    engine.set_mode(mode);
                    const std::vector<double> got = widened(blend_of<T>(engine, c, in));

                    for (int64_t p = 0; p < c.points; ++p)
                    {
                        for (int64_t i = 0; i < c.components; ++i)
                        {
                            const TruthValue truth = truth_of(c, data, p, i);

                            EXPECT_NEAR(got[static_cast<std::size_t>(p * c.components + i)],
                                        truth.value, blend_bound<T>(mode, c.influences, truth))
                                << "point " << p << " value " << i;
                        }
                    }
                }
            }
        }

        /** @brief Tests that one device run repeats to the bit on non-integer data. */
        TYPED_TEST(Medium_BlendTransformsTyped, TheDeviceRepeatsToTheBit)
        {
            using T = TypeParam;
            Engine& engine = *this->engine;
            engine.set_mode(ExecutionMode::ForceSYCL);

            const BlendCase c{"repeat", 129, 4, 4, 9, 4};
            const BlendData data = data_for<T>(c, WeightKind::Fractional);
            const BlendTensors in = tensors_for<T>(engine, c, data);

            const std::vector<T> first = blend_of<T>(engine, c, in);
            const std::vector<T> second = blend_of<T>(engine, c, in);

            ASSERT_EQ(first.size(), second.size());
            EXPECT_EQ(std::memcmp(first.data(), second.data(), first.size() * sizeof(T)), 0);
        }

        /**
         * @brief Tests that an index outside [0, T) adds no term in either mode, for both index
         *        types.
         */
        TEST_F(Medium_BlendTransforms, AnIndexOutsideTheTransformsAddsNoTerm)
        {
            const BlendCase c{"out of range", 4, 3, 3, 2, 4};
            BlendData data = data_for<float>(c, WeightKind::Integer);

            // The indices run below 0 and at or above T; the last point has no valid one.
            constexpr int64_t K_LOW = std::numeric_limits<int32_t>::min();
            constexpr int64_t K_HIGH = std::numeric_limits<int32_t>::max();
            data.indices = {0, -1, 2, 1,
                            1, K_HIGH, 0, K_LOW,
                            K_LOW, K_HIGH, 2, 1,
                            -1, -1, -1, -1};
            const std::vector<double> expected = blend_by_formula(c, data);

            // The last point has no valid influence, so it comes out as zeros.
            for (std::size_t i = 9; i < 12; ++i)
                ASSERT_EQ(expected[i], 0.0);

            BlendData wide = data;
            for (int64_t& index : wide.indices)
            {
                if (index == K_HIGH)
                    index = int64_t{1} << 40;
                else if (index == K_LOW)
                    index = -(int64_t{1} << 40);
            }
            ASSERT_EQ(blend_by_formula(c, wide), expected);

            for (const ExecutionMode mode : K_FORCED_MODES)
            {
                SCOPED_TRACE(trace_of(c, mode));
                engine->set_mode(mode);

                const BlendTensors narrow_in = tensors_for<float, int32_t>(*engine, c, data);
                const BlendTensors wide_in = tensors_for<float, int64_t>(*engine, c, wide);

                EXPECT_EQ(widened(blend_of<float>(*engine, c, narrow_in)), expected) << "INT32";
                EXPECT_EQ(widened(blend_of<float>(*engine, c, wide_in)), expected) << "INT64";
            }
        }

        /**
         * @brief Tests that an INT32, an INT64 and a UINT32 index tensor of the same values give
         *        the same bits in both modes.
         */
        TEST_F(Medium_BlendTransforms, EveryIndexTypeGivesTheSameBitsForTheSameValues)
        {
            const BlendCase c{"index types", 9, 3, 4, 4, 4};
            const BlendData data = data_for<float>(c, WeightKind::Integer);
            const std::vector<double> expected = blend_by_formula(c, data);

            for (const ExecutionMode mode : K_FORCED_MODES)
            {
                SCOPED_TRACE(trace_of(c, mode));
                engine->set_mode(mode);

                const std::vector<float> signed_narrow =
                    blend_of<float>(*engine, c, tensors_for<float, int32_t>(*engine, c, data));
                const std::vector<float> signed_wide =
                    blend_of<float>(*engine, c, tensors_for<float, int64_t>(*engine, c, data));
                const std::vector<float> unsigned_narrow =
                    blend_of<float>(*engine, c, tensors_for<float, uint32_t>(*engine, c, data));

                EXPECT_EQ(widened(signed_narrow), expected) << "INT32";
                ASSERT_EQ(signed_wide.size(), signed_narrow.size());
                ASSERT_EQ(unsigned_narrow.size(), signed_narrow.size());
                EXPECT_EQ(std::memcmp(signed_wide.data(), signed_narrow.data(),
                                      signed_narrow.size() * sizeof(float)), 0) << "INT64";
                EXPECT_EQ(std::memcmp(unsigned_narrow.data(), signed_narrow.data(),
                                      signed_narrow.size() * sizeof(float)), 0) << "UINT32";
            }
        }

        /**
         * @brief Tests that a UINT32 index at or past T, including the largest one, adds no term in
         *        both modes.
         */
        TEST_F(Medium_BlendTransforms, AUInt32IndexPastTheTransformsAddsNoTerm)
        {
            const BlendCase c{"uint32 out of range", 3, 3, 3, 2, 3};
            BlendData data = data_for<float>(c, WeightKind::Integer);
            constexpr int64_t K_LARGEST = std::numeric_limits<uint32_t>::max();
            data.indices = {0, 2, K_LARGEST,
                            K_LARGEST, 1, 5,
                            K_LARGEST, K_LARGEST, K_LARGEST};
            const std::vector<double> expected = blend_by_formula(c, data);

            for (std::size_t i = 6; i < 9; ++i)
                ASSERT_EQ(expected[i], 0.0);

            for (const ExecutionMode mode : K_FORCED_MODES)
            {
                SCOPED_TRACE(trace_of(c, mode));
                engine->set_mode(mode);

                const BlendTensors in = tensors_for<float, uint32_t>(*engine, c, data);
                EXPECT_EQ(widened(blend_of<float>(*engine, c, in)), expected) << "UINT32";
            }
        }

        /**
         * @brief Tests that a call with options puts the name and the class on the node, and one
         *        without keeps the operation id and Unstated.
         */
        TEST_F(Medium_BlendTransforms, TheNodeCarriesTheNameAndTheDeterminismClass)
        {
            BlendOps ops(recorder());
            const Tensor points = filled_tensor<float>(*engine, {2, 3});
            const Tensor transforms = filled_tensor<float>(*engine, {1, 3, 4});
            const Tensor indices = zero_indices(*engine, {2, 1});
            const Tensor weights = filled_tensor<float>(*engine, {2, 1});
            Tensor stated = sentinel_tensor<float>(*engine, {2, 3});
            Tensor silent = sentinel_tensor<float>(*engine, {2, 3});

            ops.blend_transforms(
                points, transforms, indices, weights, stated,
                OperationOptions{"skin_pose", SushiRuntime::Determinism::Tolerant});
            ops.blend_transforms(points, transforms, indices, weights, silent);

            SushiRuntime::Result<SushiRuntime::PlanView> plan = recording->plan();
            ASSERT_TRUE(plan);
            ASSERT_EQ(plan->nodes.size(), std::size_t{2});
            EXPECT_EQ(plan->nodes[0].name, "skin_pose");
            EXPECT_EQ(plan->nodes[0].determinism, SushiRuntime::Determinism::Tolerant);
            EXPECT_EQ(plan->nodes[1].name, "blend.blend_transforms");
            EXPECT_EQ(plan->nodes[1].determinism, SushiRuntime::Determinism::Unstated);
        }

        /**
         * @brief Tests that an empty name and a class outside the enum are refused at record time.
         */
        TEST_F(Medium_BlendTransforms, AnEmptyNameOrAnUnknownClassIsRefused)
        {
            BlendOps ops(recorder());
            const Tensor points = filled_tensor<float>(*engine, {2, 3});
            const Tensor transforms = filled_tensor<float>(*engine, {1, 3, 4});
            const Tensor indices = zero_indices(*engine, {2, 1});
            const Tensor weights = filled_tensor<float>(*engine, {2, 1});
            Tensor out = sentinel_tensor<float>(*engine, {2, 3});

            EXPECT_THROW(
                ops.blend_transforms(points, transforms, indices, weights, out,
                                     OperationOptions{"", SushiRuntime::Determinism::Bitwise}),
                std::invalid_argument);
            EXPECT_THROW(ops.blend_transforms(points, transforms, indices, weights, out,
                                              OperationOptions{std::string_view(),
                                                           SushiRuntime::Determinism::Bitwise}),
                         std::invalid_argument);
            EXPECT_THROW(ops.blend_transforms(
                             points, transforms, indices, weights, out,
                             OperationOptions{"skin", static_cast<SushiRuntime::Determinism>(200)}),
                         std::invalid_argument);
        }

        /** @brief Tests that the operation refuses every malformed operand at record time. */
        TEST_F(Medium_BlendTransforms, RefusesMismatchedOperandsAtRecordTime)
        {
            RecordingFixture graph(*engine);
            BlendOps ops(graph.recorder());
            Engine& e = *engine;

            const Tensor points = filled_tensor<float>(e, {2, 3});
            const Tensor transforms = filled_tensor<float>(e, {3, 3, 4});
            const Tensor indices = zero_indices(e, {2, 2});
            const Tensor weights = filled_tensor<float>(e, {2, 2});
            Tensor out = sentinel_tensor<float>(e, {2, 3});

            // The control: this call is well-formed and records.
            EXPECT_NO_THROW(ops.blend_transforms(points, transforms, indices, weights, out));
            graph.run().value();

            // Records one call with a single operand swapped for a malformed one.
            const auto refuses = [&](const Tensor& p, const Tensor& t, const Tensor& i,
                                     const Tensor& w, Tensor& o, const char* operand)
            {
                expect_refusal([&] { ops.blend_transforms(p, t, i, w, o); }, operand);
            };

            const Tensor double_transforms = filled_tensor<double>(e, {3, 3, 4});
            refuses(points, double_transforms, indices, weights, out, "transforms");

            const Tensor double_weights = filled_tensor<double>(e, {2, 2});
            refuses(points, transforms, indices, double_weights, out, "weights");

            Tensor double_out = sentinel_tensor<double>(e, {2, 3});
            refuses(points, transforms, indices, weights, double_out, "out");

            const Tensor float_indices = filled_tensor<float>(e, {2, 2});
            refuses(points, transforms, float_indices, weights, out, "indices");

            refuses(filled_tensor<float>(e, {6}), transforms, indices, weights, out, "points");
            refuses(points, filled_tensor<float>(e, {12}), indices, weights, out, "transforms");
            refuses(points, transforms, zero_indices(e, {4}), weights, out, "indices");
            refuses(points, transforms, indices, filled_tensor<float>(e, {4}), out, "weights");

            Tensor flat_out = sentinel_tensor<float>(e, {6});
            refuses(points, transforms, indices, weights, flat_out, "out");

            refuses(filled_tensor<float>(e, {2, 2}), transforms, indices, weights, out, "points");
            refuses(filled_tensor<float>(e, {2, 5}), transforms, indices, weights, out, "points");
            refuses(points, filled_tensor<float>(e, {3, 2, 4}), indices, weights, out,
                    "transforms");
            refuses(points, filled_tensor<float>(e, {3, 3, 3}), indices, weights, out,
                    "transforms");
            refuses(points, filled_tensor<float>(e, {0, 3, 4}), indices, weights, out,
                    "transforms");
            refuses(points, transforms, zero_indices(e, {3, 2}), weights, out, "indices");
            refuses(points, transforms, indices, filled_tensor<float>(e, {2, 3}), out, "weights");

            Tensor wide_out = sentinel_tensor<float>(e, {2, 4});
            refuses(points, transforms, indices, weights, wide_out, "out");

            refuses(points, transforms, zero_indices(e, {2, 0}), filled_tensor<float>(e, {2, 0}),
                    out, "indices");

            Tensor no_point_out = sentinel_tensor<float>(e, {0, 3});
            refuses(filled_tensor<float>(e, {0, 3}), transforms, zero_indices(e, {0, 2}),
                    filled_tensor<float>(e, {0, 2}), no_point_out, "points");

            refuses(points, transforms, indices, weights.transpose(0, 1), out, "weights");
        }

        /** @brief Tests that the operation refuses an output that is also an input. */
        TEST_F(Medium_BlendTransforms, RefusesAnOutputThatOverlapsAnInput)
        {
            RecordingFixture graph(*engine);
            BlendOps ops(graph.recorder());

            Tensor points = tensor_from<float>(*engine, {2, 3}, std::vector<double>(6, 1.0));
            const Tensor transforms =
                tensor_from<float>(*engine, {1, 3, 4}, std::vector<double>(12, 1.0));
            const Tensor indices = index_tensor<int32_t>(*engine, {2, 1}, {0, 0});
            const Tensor weights = tensor_from<float>(*engine, {2, 1}, {1, 1});

            // The output has the shape of the points, so the same tensor passes every shape check.
            const auto overlapping = [&] {
                ops.blend_transforms(points, transforms, indices, weights, points);
            };
            expect_refusal(overlapping, "out");
        }
    } // namespace Test
} // namespace SushiBLAS
