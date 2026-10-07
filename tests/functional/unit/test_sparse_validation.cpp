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
 * @file test_sparse_validation.cpp
 * @brief Tests the record-time operand checks of the sparse operations.
 * @author Mustafa Garip
 */

#include <array>
#include <string>
#include <vector>
#include <complex>
#include <cstdint>
#include <exception>
#include <stdexcept>
#include <gtest/gtest.h>
#include <sycl/sycl.hpp>
#include <SushiBLAS/SushiBLAS.h>
#include "../../common/test_utils.hpp"
#include "../../common/graph_probe.hpp"
#include "../../common/sparse_samples.hpp"

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
        class Unit_SparseValidation : public SushiBLASTest
        {
            protected:
                /** @brief Returns the entries 10, 20, 30 at the positions 6, 1, 4 of eight. */
                [[nodiscard]] SparseVector sample_vector()
                {
                    const SparseBuilder builder(engine->get_tensor_factory());
                    return builder.vector_from_entries(K_VECTOR_SIZE,
                                                       std::vector<int32_t>{6, 1, 4},
                                                       std::vector<float>{10.0f, 20.0f, 30.0f});
                }

                /** @brief Returns the CSR matrix of FLOAT32 host triplets, zero-based. */
                [[nodiscard]] SparseMatrix build(const HostTriplets<float>& host)
                {
                    const Core::IndexBase base = Core::IndexBase::Zero;
                    const SparseBuilder builder(engine->get_tensor_factory());
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

                /** @brief Returns the stride-2 view of the even elements of a tensor. */
                [[nodiscard]] static Tensor every_second(const Tensor& wide)
                {
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
                    SparseOps ops(probe.get_recorder());
                    EXPECT_THROW(record(ops), std::invalid_argument);
                    EXPECT_TRUE(probe.take_nodes().empty());
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
        };

        TEST_F(Unit_SparseValidation, ASparseVectorOutsideItsRulesIsRefused)
        {
            {
                SparseVector x = sample_vector();
                x.index_base = static_cast<Core::IndexBase>(0);
                expect_vector_refused("an index base of 0", x);
            }
            {
                SparseVector x = sample_vector();
                x.size = -1;
                expect_vector_refused("a negative size", x);
            }
            {
                SparseVector x = sample_vector();
                x.indices = create_initialized_tensor<float>({3}, {6.0f, 1.0f, 4.0f});
                expect_vector_refused("FLOAT32 indices", x);
            }
            {
                SparseVector x = sample_vector();
                x.size = K_PAST_INT32;
                expect_vector_refused("INT32 indices too narrow for the size", x);
            }
            {
                SparseVector x = sample_vector();
                x.indices = create_initialized_tensor<int32_t>({2}, {6, 1});
                expect_vector_refused("indices and values of different lengths", x);
            }
            {
                SparseVector x = sample_vector();
                x.indices = create_initialized_tensor<int32_t>({3, 1}, {6, 1, 4});
                expect_vector_refused("indices of rank 2", x);
            }
            {
                SparseVector x = sample_vector();
                const Tensor wide = dense(6);
                x.values = every_second(wide);
                expect_vector_refused("a strided values view", x);
            }
            {
                SparseVector x = sample_vector();
                x.values = Tensor(SushiRuntime::sushi_ptr<Storage>(), {3});
                expect_vector_refused("three values without storage", x);
            }
            {
                SparseVector x = sample_vector();
                x.size = 2;
                expect_vector_refused("more entries than the size", x);
            }
        }

        TEST_F(Unit_SparseValidation, ASparseMatrixOutsideItsRulesIsRefused)
        {
            {
                SparseMatrix A = sample();
                A.row_offsets = create_initialized_tensor<int32_t>({4}, {0, 2, 2, 5});
                expect_matrix_refused("as many row offsets as rows", A);
            }
            {
                SparseMatrix A = sample();
                A.column_indices =
                    create_initialized_tensor<int64_t>({7}, {0, 3, 1, 2, 4, 0, 4});
                expect_matrix_refused("column indices of another dtype than the offsets", A);
            }
            {
                SparseMatrix A = sample();
                A.cols = 0;
                expect_matrix_refused("entries in a matrix without columns", A);
            }
            {
                SparseMatrix A = sample();
                A.cols = K_PAST_INT32;
                expect_matrix_refused("INT32 indices too narrow for the columns", A);
            }
            expect_matrix_refused("a default-constructed matrix", SparseMatrix());
        }

        TEST_F(Unit_SparseValidation, SpmvRefusesOperandsScalarsAndFlagsOutsideItsRules)
        {
            const SparseMatrix A = sample();
            const Tensor four = dense(4);
            const Tensor five = dense(5);
            Tensor y4 = dense(4);
            Tensor y5 = dense(5);
            Tensor y_double = engine->create_tensor({4}, Core::DataType::Float64);
            const BlasScalar complex_alpha = std::complex<double>(1.0, 2.0);

            expect_refused("x of the wrong length under NO_TRANS",
                           [&](SparseOps& ops) { ops.spmv(A, four, y4); });
            expect_refused("x of the NO_TRANS length under TRANS",
                           [&](SparseOps& ops) { ops.spmv(A, five, y5, Core::Transpose::Trans); });
            expect_refused("y of another dtype",
                           [&](SparseOps& ops) { ops.spmv(A, five, y_double); });
            expect_refused("a complex alpha on FLOAT32", [&](SparseOps& ops)
            {
                ops.spmv(A, five, y4, Core::Transpose::NoTrans, complex_alpha);
            });
            expect_refused("a trans of 0", [&](SparseOps& ops)
            {
                ops.spmv(A, five, y4, static_cast<Core::Transpose>(0));
            });
        }

        TEST_F(Unit_SparseValidation, SpsymvAndSphemvRefuseOperandsAndFlagsOutsideTheirRules)
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
            expect_refused("sphemv of a 4 x 5 matrix",
                           [&](SparseOps& ops) { ops.sphemv(wide, five, y, lower); });
            expect_refused("spsymv with an uplo of 0",
                           [&](SparseOps& ops) { ops.spsymv(A, four, y, no_triangle); });
            expect_refused("sphemv with an uplo of 0",
                           [&](SparseOps& ops) { ops.sphemv(A, four, y, no_triangle); });
            expect_refused("spsymv with x of the wrong length",
                           [&](SparseOps& ops) { ops.spsymv(A, five, y, lower); });
            expect_refused("sphemv with x of the wrong length",
                           [&](SparseOps& ops) { ops.sphemv(A, five, y, lower); });
        }

        TEST_F(Unit_SparseValidation, SpmmRefusesDenseOperandsOutsideItsRules)
        {
            const SparseMatrix A = sample();
            const Tensor B = engine->create_tensor({5, 3});
            Tensor C = engine->create_tensor({4, 3});

            const Tensor cube = engine->create_tensor({5, 3, 1});
            const Tensor column_major(B.storage, {5, 3}, Core::DataType::Float32, 0,
                                      Core::Layout::ColumnMajor);
            const Tensor short_B = engine->create_tensor({4, 3});
            Tensor narrow_C = engine->create_tensor({4, 2});

            expect_refused("B of rank 3", [&](SparseOps& ops) { ops.spmm(A, cube, C); });
            expect_refused("B and C of different layouts",
                           [&](SparseOps& ops) { ops.spmm(A, column_major, C); });
            expect_refused("B with the wrong row count",
                           [&](SparseOps& ops) { ops.spmm(A, short_B, C); });
            expect_refused("C with the wrong column count",
                           [&](SparseOps& ops) { ops.spmm(A, B, narrow_C); });
        }

        TEST_F(Unit_SparseValidation, SptrsvRefusesOperandsAndFlagsOutsideItsRules)
        {
            const SparseMatrix wide = sample();
            const SparseMatrix A = square();
            Tensor b = dense(4);
            Tensor long_b = dense(5);
            const Tensor eight = dense(8);
            Tensor strided = every_second(eight);
            const Core::Uplo lower = Core::Uplo::Lower;

            expect_refused("a 4 x 5 matrix", [&](SparseOps& ops) { ops.sptrsv(wide, b, lower); });
            expect_refused("a strided b", [&](SparseOps& ops) { ops.sptrsv(A, strided, lower); });
            expect_refused("b of the wrong length",
                           [&](SparseOps& ops) { ops.sptrsv(A, long_b, lower); });
            expect_refused("an uplo of 0", [&](SparseOps& ops)
            {
                ops.sptrsv(A, b, static_cast<Core::Uplo>(0));
            });
            expect_refused("a diag of 0", [&](SparseOps& ops)
            {
                ops.sptrsv(A, b, lower, Core::Transpose::NoTrans, static_cast<Core::Diag>(0));
            });
        }

        TEST_F(Unit_SparseValidation, ForceReferenceRefusesAMatrixWithDeviceOnlyValues)
        {
            GraphProbe probe(ExecutionMode::ForceReference);
            SparseOps ops(probe.get_recorder());

            SparseMatrix A = sample();
            const Tensor x = dense(5);
            Tensor y = dense(4);

            EXPECT_NO_THROW(ops.spmv(A, x, y));
            EXPECT_EQ(probe.take_nodes().size(), 1u) << "The matrix in shared USM must record.";

            try
            {
                A.values = engine->create_tensor({A.nnz()}, Core::DataType::Float32,
                                                 SushiRuntime::API::Residency::Device);
            }
            catch (const std::exception&)
            {
                GTEST_SKIP() << "This device offers no device-only USM.";
            }

            const sycl::context context = TestEnvironment::get_context().get_queue().get_context();
            if (sycl::get_pointer_type(A.values.data(), context) != sycl::usm::alloc::device)
                GTEST_SKIP() << "This device offers no device-only USM.";

            std::string refusal;
            try
            {
                ops.spmv(A, x, y);
            }
            catch (const std::invalid_argument& error)
            {
                refusal = error.what();
            }
            EXPECT_NE(refusal.find("FORCE_REFERENCE"), std::string::npos)
                << "spmv threw \"" << refusal << "\" for device-only values";
            EXPECT_TRUE(probe.take_nodes().empty());
        }
    } // namespace Test
} // namespace SushiBLAS
