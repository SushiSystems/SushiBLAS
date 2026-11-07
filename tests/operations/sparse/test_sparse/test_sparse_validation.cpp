/**************************************************************************/
/* test_sparse_validation.cpp                                             */
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
 * @file test_sparse_validation.cpp
 * @brief Tests the record-time operand checks of the sparse operations.
 * @author Mustafa Garip
 */

#include <array>
#include <string>
#include <vector>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <functional>
#include <initializer_list>
#include <stdexcept>
#include <gtest/gtest.h>
#include <sycl/sycl.hpp>
#include <SushiBLAS/SushiBLAS.hpp>
#include <SushiRuntime/core/sushi_ptr.hpp>
#include <SushiBLAS/engine_fixture/sushiblas_test.hpp>
#include <SushiBLAS/test_runner/test_environment.hpp>
#include <SushiBLAS/graph_probe/graph_probe.hpp>
#include <SushiBLAS/sparse_fixtures/sparse_samples.hpp>
#include "support/refusal_naming.hpp"

namespace SushiBLAS
{
    namespace Test
    {
        namespace
        {
            /// @brief Length of the dense vector the sparse vector of this file compresses.
            constexpr int64_t K_VECTOR_SIZE = 8;

            /// @brief An extent whose largest index an INT32 array cannot hold.
            constexpr int64_t K_PAST_INT32 = (int64_t{1} << 31) + 5;
        } // namespace

        /**
         * @brief Builds FLOAT32 sparse operands with INT32 indices and records on a probe.
         */
        class Medium_SparseValidation : public SushiBLASTest
        {
            public:
                /** @brief Returns the entries 10, 20, 30 at the positions 6, 1, 4 of eight. */
                [[nodiscard]] SparseVector sample_vector()
                {
                    const SparseBuilder builder(engine->tensor_factory());
                    return builder.vector_from_entries(K_VECTOR_SIZE,
                                                       std::vector<int32_t>{6, 1, 4},
                                                       std::vector<float>{10.0f, 20.0f, 30.0f});
                }

                /** @brief Returns the CSR matrix of FLOAT32 host triplets, zero-based. */
                [[nodiscard]] SparseMatrix build(const HostTriplets<float>& host)
                {
                    const Core::IndexBase base = Core::IndexBase::Zero;
                    const SparseBuilder builder(engine->tensor_factory());
                    return builder.matrix_from_triplets(
                        host.rows, host.cols, stored_indices<int32_t>(host.row_indices, base),
                        stored_indices<int32_t>(host.column_indices, base), host.values);
                }

                /** @brief Returns the 4 x 5 sample matrix. */
                [[nodiscard]] SparseMatrix sample()
                {
                    return build(sample_matrix<float>());
                }

                /** @brief Returns a 4 x 4 lower triangular matrix with a full diagonal. */
                [[nodiscard]] SparseMatrix square()
                {
                    HostTriplets<float> host;
                    host.rows = 4;
                    host.cols = 4;
                    host.row_indices = {0, 1, 1, 2, 3, 3};
                    host.column_indices = {0, 0, 1, 2, 0, 3};
                    host.values = {2.0f, 1.0f, 3.0f, 5.0f, 4.0f, 7.0f};
                    return build(host);
                }

                /** @brief Returns a FLOAT32 vector of length elements. */
                [[nodiscard]] Tensor dense(int64_t length)
                {
                    return engine->create_tensor({length});
                }

                /** @brief Returns a FLOAT32 tensor of the given extents. */
                [[nodiscard]] Tensor dense_of(std::initializer_list<int64_t> dims)
                {
                    return engine->create_tensor(dims);
                }

                /** @brief Returns a FLOAT32 tensor of the given extents holding values. */
                [[nodiscard]] Tensor floats(std::initializer_list<int64_t> dims,
                                            const std::vector<float>& values)
                {
                    return create_initialized_tensor<float>(dims, values);
                }

                /** @brief Returns an INT32 tensor of the given extents holding values. */
                [[nodiscard]] Tensor int32s(std::initializer_list<int64_t> dims,
                                            const std::vector<int32_t>& values)
                {
                    return create_initialized_tensor<int32_t>(dims, values);
                }

                /** @brief Returns an INT64 tensor of the given extents holding values. */
                [[nodiscard]] Tensor int64s(std::initializer_list<int64_t> dims,
                                            const std::vector<int64_t>& values)
                {
                    return create_initialized_tensor<int64_t>(dims, values);
                }

                /**
                 * @brief Returns the stride-2 view of the even elements of a new FLOAT32 vector.
                 * @param length Elements of the vector the view reads; the view has half as many.
                 */
                [[nodiscard]] Tensor strided_view(int64_t length)
                {
                    kept_.push_back(dense(length));
                    const Tensor& wide = kept_.back();
                    const int64_t dims[1] = {wide.num_elements / 2};
                    std::array<int64_t, Core::K_MAX_TENSOR_RANK> strides{};
                    strides[0] = 2;
                    return Tensor(wide.storage, SushiRuntime::span<const int64_t>(dims, 1),
                                  strides, wide.dtype, 0, wide.layout);
                }

                /**
                 * @brief Expects a record call to throw std::invalid_argument and add no node.
                 * @param rule Names the broken rule in a failure message.
                 * @param record Callable `(SparseOps&) -> void` that makes the one call.
                 */
                template<typename Record>
                void expect_refused(const char* rule, Record&& record)
                {
                    SCOPED_TRACE(rule);

                    GraphProbe probe;
                    SparseOps ops(probe.recorder());
                    EXPECT_THROW(record(ops), std::invalid_argument);
                    EXPECT_EQ(0u, probe.take_recorded_count());
                }

                /** @brief Expects axpyi to refuse a sparse vector against a dense one of eight. */
                void expect_vector_refused(const char* rule, const SparseVector& x)
                {
                    Tensor y = dense(K_VECTOR_SIZE);
                    expect_refused(rule, [&](SparseOps& ops) { ops.axpyi(2.0, x, y); });
                }

                /** @brief Expects spmv to refuse a matrix against dense vectors of 5 and 4. */
                void expect_matrix_refused(const char* rule, const SparseMatrix& A)
                {
                    const Tensor x = dense(5);
                    Tensor y = dense(4);
                    expect_refused(rule, [&](SparseOps& ops) { ops.spmv(A, x, y); });
                }

            private:
                /// @brief The wide vectors the strided views of a test read, kept alive for it.
                std::vector<Tensor> kept_;
        };

        namespace
        {
            /** @brief Holds a rule and the change that makes a sparse vector break it. */
            struct VectorRow
            {
                const char* name; /**< The rule, which is also the row suffix. */
                std::function<void(Medium_SparseValidation&, SparseVector&)> spoil; /**< The change. */
            };

            /** @brief Returns one row for each rule a sparse vector keeps. */
            std::vector<VectorRow> vector_rows()
            {
                using Fixture = Medium_SparseValidation;
                return {
                    {"AnIndexBaseOfZero",
                     [](Fixture&, SparseVector& x) { x.index_base = static_cast<Core::IndexBase>(0); }},
                    {"ANegativeSize", [](Fixture&, SparseVector& x) { x.size = -1; }},
                    {"Float32Indices",
                     [](Fixture& f, SparseVector& x) { x.indices = f.floats({3}, {6.0f, 1.0f, 4.0f}); }},
                    {"Int32IndicesTooNarrowForTheSize",
                     [](Fixture&, SparseVector& x) { x.size = K_PAST_INT32; }},
                    {"IndicesAndValuesOfDifferentLengths",
                     [](Fixture& f, SparseVector& x) { x.indices = f.int32s({2}, {6, 1}); }},
                    {"IndicesOfRank2",
                     [](Fixture& f, SparseVector& x) { x.indices = f.int32s({3, 1}, {6, 1, 4}); }},
                    {"AStridedValuesView",
                     [](Fixture& f, SparseVector& x) { x.values = f.strided_view(6); }},
                    {"ThreeValuesWithoutStorage",
                     [](Fixture&, SparseVector& x)
                     { x.values = Tensor(SushiRuntime::sushi_ptr<Storage>(), {3}); }},
                    {"MoreEntriesThanTheSize", [](Fixture&, SparseVector& x) { x.size = 2; }}};
            }

            /** @brief Holds a rule and the change that makes a CSR matrix break it. */
            struct MatrixRow
            {
                const char* name; /**< The rule, which is also the row suffix. */
                std::function<void(Medium_SparseValidation&, SparseMatrix&)> spoil; /**< The change. */
            };

            /** @brief Returns one row for each rule a CSR matrix keeps. */
            std::vector<MatrixRow> matrix_rows()
            {
                using Fixture = Medium_SparseValidation;
                return {
                    {"AsManyRowOffsetsAsRows",
                     [](Fixture& f, SparseMatrix& A) { A.row_offsets = f.int32s({4}, {0, 2, 2, 5}); }},
                    {"ColumnIndicesOfAnotherDtypeThanTheOffsets",
                     [](Fixture& f, SparseMatrix& A)
                     { A.column_indices = f.int64s({7}, {0, 3, 1, 2, 4, 0, 4}); }},
                    {"EntriesInAMatrixWithoutColumns", [](Fixture&, SparseMatrix& A) { A.cols = 0; }},
                    {"Int32IndicesTooNarrowForTheColumns",
                     [](Fixture&, SparseMatrix& A) { A.cols = K_PAST_INT32; }},
                    {"ADefaultConstructedMatrix", [](Fixture&, SparseMatrix& A) { A = SparseMatrix(); }}};
            }

            /** @brief Holds a rule and the spmm call that breaks it. */
            struct SpmmRow
            {
                const char* name; /**< The rule, which is also the row suffix. */
                std::function<void(Medium_SparseValidation&, SparseOps&)> call; /**< The call. */
            };

            /** @brief Returns one row for each rule the dense operands of spmm keep. */
            std::vector<SpmmRow> spmm_rows()
            {
                using Fixture = Medium_SparseValidation;
                return {
                    {"BOfRank3",
                     [](Fixture& f, SparseOps& ops)
                     {
                         const SparseMatrix A = f.sample();
                         const Tensor cube = f.dense_of({5, 3, 1});
                         Tensor C = f.dense_of({4, 3});
                         ops.spmm(A, cube, C);
                     }},
                    {"BAndCOfDifferentLayouts",
                     [](Fixture& f, SparseOps& ops)
                     {
                         const SparseMatrix A = f.sample();
                         const Tensor B = f.dense_of({5, 3});
                         Tensor C = f.dense_of({4, 3});
                         const Tensor column_major(B.storage, {5, 3}, Core::DataType::Float32, 0,
                                                   Core::Layout::ColumnMajor);
                         ops.spmm(A, column_major, C);
                     }},
                    {"BWithTheWrongRowCount",
                     [](Fixture& f, SparseOps& ops)
                     {
                         const SparseMatrix A = f.sample();
                         const Tensor short_B = f.dense_of({4, 3});
                         Tensor C = f.dense_of({4, 3});
                         ops.spmm(A, short_B, C);
                     }},
                    {"CWithTheWrongColumnCount",
                     [](Fixture& f, SparseOps& ops)
                     {
                         const SparseMatrix A = f.sample();
                         const Tensor B = f.dense_of({5, 3});
                         Tensor narrow_C = f.dense_of({4, 2});
                         ops.spmm(A, B, narrow_C);
                     }}};
            }

            /// @brief Names a row, which gtest takes as the test suffix.
            template<typename Row>
            std::string row_name(const ::testing::TestParamInfo<Row>& info)
            {
                return info.param.name;
            }
        } // namespace

        /** @brief Breaks one rule of a sparse vector at a time. */
        class Medium_SparseValidationOfVector : public Medium_SparseValidation,
                                                public ::testing::WithParamInterface<VectorRow>
        {

        };

        TEST_P(Medium_SparseValidationOfVector, AxpyiRefusesASparseVectorOutsideItsRules)
        {
            SparseVector x = sample_vector();
            GetParam().spoil(*this, x);

            expect_vector_refused(GetParam().name, x);
        }

        INSTANTIATE_TEST_SUITE_P(, Medium_SparseValidationOfVector,
                                 ::testing::ValuesIn(vector_rows()), row_name<VectorRow>);

        /** @brief Breaks one rule of a CSR matrix at a time. */
        class Medium_SparseValidationOfMatrix : public Medium_SparseValidation,
                                                public ::testing::WithParamInterface<MatrixRow>
        {

        };

        TEST_P(Medium_SparseValidationOfMatrix, SpmvRefusesASparseMatrixOutsideItsRules)
        {
            SparseMatrix A = sample();
            GetParam().spoil(*this, A);

            expect_matrix_refused(GetParam().name, A);
        }

        INSTANTIATE_TEST_SUITE_P(, Medium_SparseValidationOfMatrix,
                                 ::testing::ValuesIn(matrix_rows()), row_name<MatrixRow>);

        /** @brief Breaks one rule of the dense operands of spmm at a time. */
        class Medium_SparseValidationOfSpmm : public Medium_SparseValidation,
                                              public ::testing::WithParamInterface<SpmmRow>
        {

        };

        TEST_P(Medium_SparseValidationOfSpmm, SpmmRefusesDenseOperandsOutsideItsRules)
        {
            expect_refused(GetParam().name,
                           [&](SparseOps& ops) { GetParam().call(*this, ops); });
        }

        INSTANTIATE_TEST_SUITE_P(, Medium_SparseValidationOfSpmm, ::testing::ValuesIn(spmm_rows()),
                                 row_name<SpmmRow>);

        TEST_F(Medium_SparseValidation, SpmvRefusesOperandsOutsideItsRules)
        {
            const SparseMatrix A = sample();
            const Tensor four = dense(4);
            const Tensor five = dense(5);
            Tensor y4 = dense(4);
            Tensor y5 = dense(5);
            Tensor y_double = engine->create_tensor({4}, Core::DataType::Float64);

            expect_refused("x of the wrong length under NO_TRANS",
                           [&](SparseOps& ops) { ops.spmv(A, four, y4); });
            expect_refused("x of the NO_TRANS length under TRANS",
                           [&](SparseOps& ops) { ops.spmv(A, five, y5, Core::Transpose::Trans); });
            expect_refused("y of another dtype",
                           [&](SparseOps& ops) { ops.spmv(A, five, y_double); });
        }

        TEST_F(Medium_SparseValidation, SpmvRefusesAComplexAlphaOnFloat32)
        {
            const SparseMatrix A = sample();
            const Tensor five = dense(5);
            Tensor y4 = dense(4);
            const BLASScalar complex_alpha = std::complex<double>(1.0, 2.0);

            expect_refused("a complex alpha on FLOAT32", [&](SparseOps& ops)
            {
                ops.spmv(A, five, y4, Core::Transpose::NoTrans, complex_alpha);
            });
        }

        TEST_F(Medium_SparseValidation, SpmvRefusesATransFlagOutsideItsRules)
        {
            const SparseMatrix A = sample();
            const Tensor five = dense(5);
            Tensor y4 = dense(4);

            expect_refused("a trans of 0", [&](SparseOps& ops)
            {
                ops.spmv(A, five, y4, static_cast<Core::Transpose>(0));
            });
        }

        TEST_F(Medium_SparseValidation, SpsymvRefusesOperandsAndFlagsOutsideItsRules)
        {
            const SparseMatrix wide = sample();
            const SparseMatrix A = square();
            const Tensor four = dense(4);
            const Tensor five = dense(5);
            Tensor y = dense(4);
            const Core::Uplo lower = Core::Uplo::Lower;
            const Core::Uplo no_triangle = static_cast<Core::Uplo>(0);

            expect_refused("spsymv of a 4 x 5 matrix",
                           [&](SparseOps& ops) { ops.spsymv(wide, five, y, lower); });
            expect_refused("spsymv with an uplo of 0",
                           [&](SparseOps& ops) { ops.spsymv(A, four, y, no_triangle); });
            expect_refused("spsymv with x of the wrong length",
                           [&](SparseOps& ops) { ops.spsymv(A, five, y, lower); });
        }

        TEST_F(Medium_SparseValidation, SphemvRefusesOperandsAndFlagsOutsideItsRules)
        {
            const SparseMatrix wide = sample();
            const SparseMatrix A = square();
            const Tensor four = dense(4);
            const Tensor five = dense(5);
            Tensor y = dense(4);
            const Core::Uplo lower = Core::Uplo::Lower;
            const Core::Uplo no_triangle = static_cast<Core::Uplo>(0);

            expect_refused("sphemv of a 4 x 5 matrix",
                           [&](SparseOps& ops) { ops.sphemv(wide, five, y, lower); });
            expect_refused("sphemv with an uplo of 0",
                           [&](SparseOps& ops) { ops.sphemv(A, four, y, no_triangle); });
            expect_refused("sphemv with x of the wrong length",
                           [&](SparseOps& ops) { ops.sphemv(A, five, y, lower); });
        }

        TEST_F(Medium_SparseValidation, SptrsvRefusesOperandsOutsideItsRules)
        {
            const SparseMatrix wide = sample();
            const SparseMatrix A = square();
            Tensor b = dense(4);
            Tensor long_b = dense(5);
            Tensor strided = strided_view(8);
            const Core::Uplo lower = Core::Uplo::Lower;

            expect_refused("a 4 x 5 matrix", [&](SparseOps& ops) { ops.sptrsv(wide, b, lower); });
            expect_refused("a strided b", [&](SparseOps& ops) { ops.sptrsv(A, strided, lower); });
            expect_refused("b of the wrong length",
                           [&](SparseOps& ops) { ops.sptrsv(A, long_b, lower); });
        }

        TEST_F(Medium_SparseValidation, SptrsvRefusesFlagsOutsideItsRules)
        {
            const SparseMatrix A = square();
            Tensor b = dense(4);
            const Core::Uplo lower = Core::Uplo::Lower;

            expect_refused("an uplo of 0", [&](SparseOps& ops)
            {
                ops.sptrsv(A, b, static_cast<Core::Uplo>(0));
            });
            expect_refused("a diag of 0", [&](SparseOps& ops)
            {
                ops.sptrsv(A, b, lower, Core::Transpose::NoTrans, static_cast<Core::Diag>(0));
            });
        }

        TEST_F(Medium_SparseValidation, ForceReferenceRecordsAMatrixInSharedUsm)
        {
            GraphProbe probe(ExecutionMode::ForceReference);
            SparseOps ops(probe.recorder());

            const SparseMatrix A = sample();
            const Tensor x = dense(5);
            Tensor y = dense(4);

            ops.spmv(A, x, y);

            EXPECT_EQ(probe.take_recorded_count(), 1u) << "The matrix in shared USM must record.";
        }

        /**
         * @brief Gives a test FLOAT32 values in device-only USM, and skips the test on a device
         *        that offers none.
         */
        class Medium_SparseValidationDeviceOnly : public Medium_SparseValidation
        {
            protected:
                /** @brief Allocates seven device-only values, the entries of the sample matrix. */
                void SetUp() override
                {
                    Medium_SparseValidation::SetUp();
                    try
                    {
                        device_only_values_ = engine->create_tensor(
                            {static_cast<int64_t>(sample_matrix<float>().values.size())},
                            Core::DataType::Float32, SushiRuntime::Residency::Device);
                    }
                    catch (const std::exception&)
                    {
                        GTEST_SKIP() << "This device offers no device-only USM.";
                    }

                    const sycl::context context = TestEnvironment::queue().get_context();
                    if (sycl::get_pointer_type(device_only_values_.data(), context) !=
                        sycl::usm::alloc::device)
                        GTEST_SKIP() << "This device offers no device-only USM.";
                }

                /// @brief The values of the sample matrix, allocated in device-only USM.
                Tensor device_only_values_;
        };

        TEST_F(Medium_SparseValidationDeviceOnly, ForceReferenceRefusesAMatrixWithDeviceOnlyValues)
        {
            GraphProbe probe(ExecutionMode::ForceReference);
            SparseOps ops(probe.recorder());

            SparseMatrix A = sample();
            A.values = device_only_values_;
            const Tensor x = dense(5);
            Tensor y = dense(4);

            expect_refusal_naming([&] { ops.spmv(A, x, y); }, "ForceReference");
            EXPECT_EQ(0u, probe.take_recorded_count());
        }
    } // namespace Test
} // namespace SushiBLAS
