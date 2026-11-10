/**************************************************************************/
/* triangular_support.hpp                                           */
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
 * @file triangular_support.hpp
 * @brief Declares a host triangular system, its draws, the fixture and the row of the suites.
 * @author Mustafa Garip
 */

#pragma once

#include <array>
#include <cmath>
#include <future>
#include <limits>
#include <random>
#include <string>
#include <vector>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <utility>
#include <initializer_list>
#include <gtest/gtest.h>
#include <sycl/sycl.hpp>
#include <SushiBLAS/blas_kernels/sycl/level2/trmv.hpp>
#include <SushiBLAS/blas_kernels/sycl/level2/tbsv.hpp>
#include <SushiBLAS/blas_kernels/sycl/level2/tpsv.hpp>
#include <SushiBLAS/blas_kernels/sycl/level3/trsm.hpp>
#include <SushiBLAS/blas_kernels/sycl/level3/triangular_plan.hpp>
#include <SushiBLAS/SushiBLAS.hpp>
#include <SushiBLAS/core/traits.hpp>
#include <SushiBLAS/arithmetic/accumulator.hpp>
#include <SushiBLAS/arithmetic/complex_arithmetic.hpp>
#include <SushiBLAS/reference/reference_accumulator.hpp>
#include <SushiBLAS/blas_seam_kernels/band_access.hpp>
#include <SushiBLAS/blas_seam_kernels/packed_access.hpp>
#include <SushiBLAS/engine_fixture/sushiblas_test.hpp>
#include <SushiBLAS/tensor_expectations/scalar_dtype.hpp>
#include <SushiBLAS/test_runner/test_environment.hpp>
#include <SushiBLAS/blas_test_oracle/blas_truth.hpp>
#include "blas_rows.hpp"

namespace SushiBLAS
{
    namespace Test
    {
        namespace TriangularSupport
        {
            /// @brief Holds the seed every case of this file draws its operands from.
            inline constexpr std::uint32_t K_SEED = 20261007u;

            /// @brief Holds the order of the small systems: ten blocks of 7 with a short last one.
            inline constexpr int64_t K_ORDER = 67;

            /// @brief Holds the right-hand-side vectors of the small systems.
            inline constexpr int64_t K_VECTORS = 5;

            /// @brief Holds the block widths compared with one block of the whole order.
            inline constexpr int64_t K_BLOCK_WIDTHS[] = {1, 2, 7, 64, 1000};

            /// @brief Holds the block widths the exact systems are solved at.
            inline constexpr int64_t K_EXACT_WIDTHS[] = {1, 7, K_ORDER};

            /// @brief Holds the block width of the batched and the gated solve: 19 launches.
            inline constexpr int64_t K_NARROW_WIDTH = 7;

            /// @brief Holds the matrices of one batched solve.
            inline constexpr int64_t K_BATCH = 3;

            /// @brief Holds the order at which a device of several compute units blocks a solve.
            inline constexpr int64_t K_LARGE_ORDER = 1100;

            /// @brief Holds the right-hand-side vectors of the large exact TRSM.
            inline constexpr int64_t K_LARGE_VECTORS = 3;

            /// @brief Holds the order of the systems whose residual is bounded.
            inline constexpr int64_t K_RESIDUAL_ORDER = 300;

            /// @brief Holds the right-hand-side vectors of the systems whose residual is bounded.
            inline constexpr int64_t K_RESIDUAL_VECTORS = 4;

            /// @brief Holds the extent of the empty systems' other axis.
            inline constexpr int64_t K_EMPTY_EXTENT = 3;

            /// @brief Holds the orders of the exact TRMV cases.
            inline constexpr int64_t K_PRODUCT_ORDERS[] = {1, 2, 70};

            /// @brief Holds the increments of x in the exact TRMV cases.
            inline constexpr int64_t K_INCREMENTS[] = {1, 3};

            /// @brief Holds the order of the TRMV measured against the truth.
            inline constexpr int64_t K_PRODUCT_BOUND_ORDER = 1500;

            /// @brief Holds the order of the TRMV whose snapshot fits a pooled block.
            inline constexpr int64_t K_POOLED_ORDER = 64;

            /// @brief Holds the work-group width the TRMV kernel is called with directly.
            inline constexpr std::size_t K_GROUP_WIDTH = 8;

            /// @brief Holds the two row shapes of the TRMV kernel: per work-item and per group.
            inline constexpr std::size_t K_GROUP_WIDTHS[] = {0, K_GROUP_WIDTH};

            /// @brief Holds the scale of the right-hand side in the rounded cases.
            inline constexpr double K_ROUNDED_ALPHA = 1.5;

            /// @brief Holds the scale of the right-hand side in the exact cases.
            inline constexpr double K_EXACT_ALPHA = 2.0;

            /**
             * @brief Holds the value written between the elements of a strided x and into B
             *        where a solve must write nothing.
             */
            inline constexpr double K_SENTINEL = 77.0;

            /// @brief Holds the fractions a draw can take on each side of zero: 2^20.
            inline constexpr double K_FRACTION_STEPS = 1048576.0;

            /// @brief Holds the roundings a row product makes whatever its length.
            inline constexpr std::size_t K_FIXED_ROUNDINGS = 6;

            /// @brief Holds the roundings a substitution step makes whatever its length.
            inline constexpr std::size_t K_STEP_ROUNDINGS = 6;

            /// @brief Holds the roundings counted for a compensated sum of any length.
            inline constexpr std::size_t K_COMPENSATED_ROUNDINGS = 2;

            /// @brief Holds the order of the HALF system of test_blas_half_accumulation.cpp.
            inline constexpr int64_t K_HALF_ORDER = 64;

            /// @brief Holds the block widths the HALF system is solved at.
            inline constexpr int64_t K_HALF_WIDTHS[] = {8, K_HALF_ORDER};

            /// @brief Holds the off-diagonals of the small band systems: none, a few, every one.
            inline constexpr int64_t K_BAND_WIDTHS[] = {0, 3, K_ORDER - 1};

            /// @brief Holds the off-diagonals at which a device of several units blocks a band.
            inline constexpr int64_t K_LARGE_BAND = 400;

            /// @brief Holds the two modes a public call is run in.
            inline constexpr ExecutionMode K_MODES[] = {ExecutionMode::ForceReference,
                                                 ExecutionMode::ForceSYCL};

            /** @brief Holds a queued event that stays incomplete until the gate is opened. */
            class DependencyGate final
            {
                public:
                    /** @brief Submits a host task to queue that blocks until open() is called. */
                    explicit DependencyGate(sycl::queue& queue)
                        : event_(queue.submit(
                              [opened = opened_.get_future().share()](sycl::handler& h)
                              {
                                  h.host_task([opened]() { opened.wait(); });
                              }))
                    {

                    }

                    DependencyGate(const DependencyGate&) = delete;
                    DependencyGate& operator=(const DependencyGate&) = delete;
                    DependencyGate(DependencyGate&&) = delete;
                    DependencyGate& operator=(DependencyGate&&) = delete;

                    /** @brief Opens the gate, so that a test leaving early blocks no host task. */
                    ~DependencyGate()
                    {
                        open();
                    }

                    /** @brief Returns the event a kernel is handed as its dependency. */
                    [[nodiscard]] const sycl::event& event() const noexcept { return event_; }

                    /** @brief Lets the host task finish; a second call does nothing. */
                    void open()
                    {
                        if (is_open_)
                            return;

                        is_open_ = true;
                        opened_.set_value();
                    }

                private:
                    /// @brief Fulfilled by open(); declared first, because event_ is built from it.
                    std::promise<void> opened_;

                    /// @brief The event of the blocked host task.
                    sycl::event event_;

                    /// @brief Whether open() has fulfilled the promise.
                    bool is_open_ = false;
            };

            /** @brief Returns whether every command behind an event has finished. */
            [[nodiscard]] inline bool is_complete(const sycl::event& event)
            {
                return event.get_info<sycl::info::event::command_execution_status>() ==
                       sycl::info::event_command_status::complete;
            }

            /** @brief Holds the flags of one triangular call. */
            struct TriangularFlags
            {
                Core::Layout layout = Core::Layout::RowMajor;     /**< The memory order. */
                Core::Side side = Core::Side::Left;                /**< The side of op(A). */
                Core::Uplo uplo = Core::Uplo::Lower;               /**< The triangle read. */
                Core::Transpose trans = Core::Transpose::NoTrans; /**< The op applied to A. */
                Core::Diag diag = Core::Diag::NonUnit;            /**< The diagonal read. */
            };

            /** @brief Returns the flags of a call as one line of a failure message. */
            [[nodiscard]] inline std::string describe(const TriangularFlags& flags)
            {
                return std::string(flags.layout == Core::Layout::RowMajor ? "ROW_MAJOR"
                                                                           : "COLUMN_MAJOR") +
                       " " + Core::to_string(flags.side) + " " + Core::to_string(flags.uplo) +
                       " " + Core::to_string(flags.trans) + " " + Core::to_string(flags.diag);
            }

            /**
             * @brief Returns every flag combination over the given sides: both layouts, both
             *        triangles, both diagonals, and each op the element type tells apart.
             * @param is_complex Whether CONJ_TRANS is listed, which only a complex type tells apart.
             */
            [[nodiscard]] inline std::vector<TriangularFlags> flags_over(
                bool is_complex, std::initializer_list<Core::Side> sides)
            {
                std::vector<Core::Transpose> transposes = {Core::Transpose::NoTrans,
                                                           Core::Transpose::Trans};
                if (is_complex)
                    transposes.push_back(Core::Transpose::ConjTrans);

                std::vector<TriangularFlags> all;
                for (const Core::Layout layout :
                     {Core::Layout::RowMajor, Core::Layout::ColumnMajor})
                    for (const Core::Side side : sides)
                        for (const Core::Uplo uplo : {Core::Uplo::Upper, Core::Uplo::Lower})
                            for (const Core::Transpose trans : transposes)
                                for (const Core::Diag diag :
                                     {Core::Diag::NonUnit, Core::Diag::Unit})
                                    all.push_back(
                                        TriangularFlags{layout, side, uplo, trans, diag});
                return all;
            }

            /** @brief Returns a real value as an element of T with a zero imaginary part. */
            template<typename T>
            [[nodiscard]] T element_of(double value) noexcept
            {
                return static_cast<T>(static_cast<Kernels::RealType<T>>(value));
            }

            /** @brief Returns the element no kernel may read: NaN in every component. */
            template<typename T>
            [[nodiscard]] T poison() noexcept
            {
                using R = Kernels::RealType<T>;
                const R nan = std::numeric_limits<R>::quiet_NaN();
                if constexpr (Core::is_complex_v<T>)
                    return T(nan, nan);
                else
                    return nan;
            }

            /** @brief Draws integers and binary fractions from one generator with a fixed seed. */
            class Draws final
            {
                public:
                    /** @brief Seeds the generator. */
                    explicit Draws(std::uint32_t seed)
                        : generator_(seed)
                    {

                    }

                    /** @brief Returns an integer from low to high, both included. */
                    [[nodiscard]] int integer(int low, int high)
                    {
                        const std::uint32_t span = static_cast<std::uint32_t>(high - low + 1);
                        return low + static_cast<int>(next() % span);
                    }

                    /** @brief Returns a multiple of 2^-20 in [-1, 1), which every type holds. */
                    [[nodiscard]] double fraction()
                    {
                        return static_cast<double>(next() >> 11) / K_FRACTION_STEPS - 1.0;
                    }

                    /** @brief Returns an integer of T, a Gaussian integer for a complex T. */
                    template<typename T>
                    [[nodiscard]] T integer_element(int low, int high)
                    {
                        using R = Kernels::RealType<T>;
                        if constexpr (Core::is_complex_v<T>)
                        {
                            const R real = static_cast<R>(integer(low, high));
                            const R imaginary = static_cast<R>(integer(low, high));
                            return T(real, imaginary);
                        }
                        else
                            return static_cast<R>(integer(low, high));
                    }

                    /** @brief Returns a fraction of T; a complex T takes the real part first. */
                    template<typename T>
                    [[nodiscard]] T fraction_element()
                    {
                        using R = Kernels::RealType<T>;
                        if constexpr (Core::is_complex_v<T>)
                        {
                            const R real = static_cast<R>(fraction());
                            const R imaginary = static_cast<R>(fraction());
                            return T(real, imaginary);
                        }
                        else
                            return static_cast<R>(fraction());
                    }

                private:
                    /** @brief Returns the next 32 bits of the generator. */
                    [[nodiscard]] std::uint32_t next()
                    {
                        return static_cast<std::uint32_t>(generator_());
                    }

                    /// @brief The generator every draw advances.
                    std::mt19937 generator_;
            };

            /** @brief Returns count integers of T from low to high. */
            template<typename T>
            [[nodiscard]] std::vector<T> integer_values(std::size_t count, int low, int high,
                                                        Draws& draws)
            {
                std::vector<T> values(count);
                for (T& value : values)
                    value = draws.integer_element<T>(low, high);
                return values;
            }

            /** @brief Returns count fractions of T in [-1, 1). */
            template<typename T>
            [[nodiscard]] std::vector<T> fraction_values(std::size_t count, Draws& draws)
            {
                std::vector<T> values(count);
                for (T& value : values)
                    value = draws.fraction_element<T>();
                return values;
            }

            /** @brief Returns 1, 2, -2 or 4, or 2i or -2i if complex: an exact divisor. */
            template<typename T>
            [[nodiscard]] T exact_pivot(Draws& draws)
            {
                if constexpr (Core::is_complex_v<T>)
                {
                    const T pivots[] = {T(1), T(2), T(-2), T(4), T(0, 2), T(0, -2)};
                    return pivots[draws.integer(0, 5)];
                }
                else
                {
                    const T pivots[] = {T(1), T(2), T(-2), T(4)};
                    return pivots[draws.integer(0, 3)];
                }
            }

            /**
             * @brief Returns a full square matrix of integers from -2 to 2 around a diagonal of
             *        exact pivots, row by row.
             */
            template<typename T>
            [[nodiscard]] std::vector<T> exact_matrix(int64_t order, Draws& draws)
            {
                std::vector<T> full(static_cast<std::size_t>(order * order));
                for (int64_t r = 0; r < order; ++r)
                    for (int64_t c = 0; c < order; ++c)
                        full[static_cast<std::size_t>(r * order + c)] =
                            (r == c) ? exact_pivot<T>(draws) : draws.integer_element<T>(-2, 2);
                return full;
            }

            /**
             * @brief Returns a full square matrix of fractions divided by the order around a
             *        diagonal of a fraction plus 2, row by row.
             */
            template<typename T>
            [[nodiscard]] std::vector<T> dominant_matrix(int64_t order, Draws& draws)
            {
                std::vector<T> full(static_cast<std::size_t>(order * order));
                for (int64_t r = 0; r < order; ++r)
                {
                    for (int64_t c = 0; c < order; ++c)
                    {
                        const T value = draws.fraction_element<T>();
                        full[static_cast<std::size_t>(r * order + c)] =
                            (r == c) ? value + element_of<T>(2.0)
                                     : value / element_of<T>(static_cast<double>(order));
                    }
                }
                return full;
            }

            /** @brief Returns a full matrix with zero wherever a column is over k from its row. */
            template<typename T>
            [[nodiscard]] std::vector<T> banded(std::vector<T> full, int64_t order, int64_t k)
            {
                for (int64_t r = 0; r < order; ++r)
                    for (int64_t c = 0; c < order; ++c)
                        if (r - c > k || c - r > k)
                            full[static_cast<std::size_t>(r * order + c)] = element_of<T>(0.0);
                return full;
            }

            /** @brief Returns whether a triangular call reads element (r, c) of A. */
            [[nodiscard]] inline bool is_read(const TriangularFlags& flags, int64_t r, int64_t c) noexcept
            {
                if (r == c)
                    return flags.diag != Core::Diag::Unit;
                return flags.uplo == Core::Uplo::Upper ? r < c : r > c;
            }

            /**
             * @brief Returns the packed storage of the triangle a call reads, poisoned elsewhere.
             * @param full order * order elements, row by row.
             */
            template<typename T>
            [[nodiscard]] std::vector<T> packed_storage(const TriangularFlags& flags, int64_t order,
                                                        const std::vector<T>& full)
            {
                const Kernels::PackedAccess stored = Kernels::packed_access(
                    flags.layout, flags.uplo, Core::Transpose::NoTrans, order);

                std::vector<T> packed(static_cast<std::size_t>(order * (order + 1) / 2),
                                      poison<T>());
                for (int64_t r = 0; r < order; ++r)
                    for (int64_t c = 0; c < order; ++c)
                        if (is_read(flags, r, c))
                            packed[static_cast<std::size_t>(stored.index(r, c))] =
                                full[static_cast<std::size_t>(r * order + c)];
                return packed;
            }

            /**
             * @brief Returns the band storage of the triangle a call reads, k + 1 elements per
             *        line, poisoned elsewhere.
             * @param full order * order elements, row by row.
             */
            template<typename T>
            [[nodiscard]] std::vector<T> band_storage(const TriangularFlags& flags, int64_t order,
                                                      int64_t k, const std::vector<T>& full)
            {
                const Kernels::BandAccess stored = Kernels::triangle_band_access(
                    flags.layout, flags.uplo, Core::Transpose::NoTrans, k, k + 1);

                std::vector<T> band(static_cast<std::size_t>(order * (k + 1)), poison<T>());
                for (int64_t r = 0; r < order; ++r)
                    for (int64_t c = 0; c < order; ++c)
                        if (is_read(flags, r, c) && stored.holds(r, c))
                            band[static_cast<std::size_t>(stored.index(r, c))] =
                                full[static_cast<std::size_t>(r * order + c)];
                return band;
            }

            /** @brief Returns every element of values times factor. */
            template<typename T>
            [[nodiscard]] std::vector<T> scaled(const std::vector<T>& values, T factor)
            {
                std::vector<T> out(values.size());
                for (std::size_t i = 0; i < values.size(); ++i)
                    out[i] = factor * values[i];
                return out;
            }

            /** @brief Returns values spread increment apart, with gap in every element between. */
            template<typename T>
            [[nodiscard]] std::vector<T> spaced(const std::vector<T>& values, int64_t increment,
                                                T gap)
            {
                if (values.empty())
                    return {};

                const std::size_t step = static_cast<std::size_t>(increment);
                std::vector<T> out((values.size() - 1) * step + 1, gap);
                for (std::size_t i = 0; i < values.size(); ++i)
                    out[i * step] = values[i];
                return out;
            }

            /** @brief Returns the elements of values from first up to, and not including, last. */
            template<typename T>
            [[nodiscard]] std::vector<T> slice_of(const std::vector<T>& values, std::size_t first,
                                                  std::size_t last)
            {
                return std::vector<T>(values.begin() + static_cast<std::ptrdiff_t>(first),
                                      values.begin() + static_cast<std::ptrdiff_t>(last));
            }

            /** @brief Returns the offset of element (r, c) of a dense rows x cols matrix. */
            [[nodiscard]] constexpr std::size_t offset_of(Core::Layout layout, int64_t rows,
                                                          int64_t cols, int64_t r,
                                                          int64_t c) noexcept
            {
                return static_cast<std::size_t>(layout == Core::Layout::RowMajor ? r * cols + c
                                                                                  : c * rows + r);
            }

            /** @brief Returns extent as a leading dimension: 1 when the axis is empty. */
            [[nodiscard]] constexpr int64_t leading(int64_t extent) noexcept
            {
                return extent > 0 ? extent : 1;
            }

            /**
             * @brief Holds a triangular call on the host as steps and vectors: op(A) * X = B on
             *        the left, X * op(A) = B on the right.
             * @tparam T The element type.
             */
            template<typename T>
            class HostSolve final
            {
                public:
                    /**
                     * @brief Binds the flags, the extents and a full square matrix.
                     * @param full order * order elements, row by row; the call reads one triangle.
                     */
                    HostSolve(const TriangularFlags& flags, int64_t order, int64_t vectors,
                              std::vector<T> full)
                        : flags_(flags), order_(order), vectors_(vectors), full_(std::move(full))
                    {

                    }

                    /** @brief Returns the flags of the call. */
                    [[nodiscard]] const TriangularFlags& flags() const noexcept { return flags_; }

                    /** @brief Returns the steps of one substitution: the order of A. */
                    [[nodiscard]] int64_t order() const noexcept { return order_; }

                    /** @brief Returns the independent vectors of B. */
                    [[nodiscard]] int64_t vectors() const noexcept { return vectors_; }

                    /** @brief Returns the rows of B. */
                    [[nodiscard]] int64_t rows() const noexcept
                    {
                        return is_left() ? order_ : vectors_;
                    }

                    /** @brief Returns the columns of B. */
                    [[nodiscard]] int64_t cols() const noexcept
                    {
                        return is_left() ? vectors_ : order_;
                    }

                    /** @brief Returns the elements of B. */
                    [[nodiscard]] std::size_t elements() const noexcept
                    {
                        return static_cast<std::size_t>(order_ * vectors_);
                    }

                    /** @brief Returns the leading dimension of the dense A. */
                    [[nodiscard]] int64_t lda() const noexcept { return leading(order_); }

                    /** @brief Returns the leading dimension of the dense B. */
                    [[nodiscard]] int64_t ldb() const noexcept
                    {
                        return leading(flags_.layout == Core::Layout::RowMajor ? cols() : rows());
                    }

                    /** @brief Returns whether unknown t has a factor in step s, itself included. */
                    [[nodiscard]] bool is_stored(int64_t s, int64_t t) const noexcept
                    {
                        const Position p = source(s, t);
                        return p.row == p.col || is_off_diagonal_stored(p);
                    }

                    /** @brief Returns the factor of unknown t in step s; 0 off the triangle. */
                    [[nodiscard]] T coefficient(int64_t s, int64_t t) const
                    {
                        const Position p = source(s, t);
                        if (p.row == p.col && flags_.diag == Core::Diag::Unit)
                            return element_of<T>(1.0);
                        if (p.row != p.col && !is_off_diagonal_stored(p))
                            return element_of<T>(0.0);
                        return Kernels::conjugate_if(Core::is_conjugated(flags_.trans),
                                                     full_[at(p.row, p.col)]);
                    }

                    /** @brief Returns the offset of unknown (s, v) in the dense B. */
                    [[nodiscard]] std::size_t unknown(int64_t s, int64_t v) const noexcept
                    {
                        return is_left() ? offset_of(flags_.layout, rows(), cols(), s, v)
                                         : offset_of(flags_.layout, rows(), cols(), v, s);
                    }

                    /** @brief Returns the dense A, poisoned wherever the call must not read. */
                    [[nodiscard]] std::vector<T> stored_matrix() const
                    {
                        std::vector<T> stored(full_.size(), poison<T>());
                        for (int64_t r = 0; r < order_; ++r)
                        {
                            for (int64_t c = 0; c < order_; ++c)
                            {
                                const bool read = (r == c) ? flags_.diag != Core::Diag::Unit
                                                           : is_off_diagonal_stored(Position{r, c});
                                if (read)
                                    stored[offset_of(flags_.layout, order_, order_, r, c)] =
                                        full_[at(r, c)];
                            }
                        }
                        return stored;
                    }

                    /**
                     * @brief Returns op(A) * X on the left, X * op(A) on the right, laid out as B.
                     * @param x The elements of X, laid out as B.
                     */
                    [[nodiscard]] std::vector<T> product(const std::vector<T>& x) const
                    {
                        std::vector<T> out(x.size(), element_of<T>(0.0));
                        for (int64_t v = 0; v < vectors_; ++v)
                        {
                            for (int64_t s = 0; s < order_; ++s)
                            {
                                T sum = element_of<T>(0.0);
                                for (int64_t t = 0; t < order_; ++t)
                                    if (is_stored(s, t))
                                        sum += coefficient(s, t) * x[unknown(t, v)];
                                out[unknown(s, v)] = sum;
                            }
                        }
                        return out;
                    }

                private:
                    /** @brief Holds the row and the column of one element of A. */
                    struct Position
                    {
                        int64_t row; /**< The row of the element. */
                        int64_t col; /**< The column of the element. */
                    };

                    /** @brief Returns whether op(A) stands on the left of X. */
                    [[nodiscard]] bool is_left() const noexcept
                    {
                        return flags_.side == Core::Side::Left;
                    }

                    /** @brief Returns the offset of element (r, c) of the full matrix. */
                    [[nodiscard]] std::size_t at(int64_t r, int64_t c) const noexcept
                    {
                        return static_cast<std::size_t>(r * order_ + c);
                    }

                    /** @brief Returns the element of A with the factor of unknown t in step s. */
                    [[nodiscard]] Position source(int64_t s, int64_t t) const noexcept
                    {
                        const Position of_op = is_left() ? Position{s, t} : Position{t, s};
                        return Core::is_transposed(flags_.trans)
                                   ? Position{of_op.col, of_op.row}
                                   : of_op;
                    }

                    /** @brief Returns whether an off-diagonal element lies in the triangle read. */
                    [[nodiscard]] bool is_off_diagonal_stored(const Position& p) const noexcept
                    {
                        return flags_.uplo == Core::Uplo::Upper ? p.row < p.col : p.row > p.col;
                    }

                    /// @brief The flags of the call.
                    TriangularFlags flags_;

                    /// @brief The order of A.
                    int64_t order_;

                    /// @brief The independent vectors of B.
                    int64_t vectors_;

                    /// @brief The full square matrix, row by row.
                    std::vector<T> full_;
            };

            /** @brief Returns the queue every kernel of this file is launched on. */
            [[nodiscard]] inline sycl::queue& test_queue()
            {
                return TestEnvironment::queue();
            }

            /**
             * @brief Returns a rank-1 tensor of the engine that holds values.
             * @return A tensor without storage when values is empty.
             */
            template<typename E>
            [[nodiscard]] Tensor device_buffer(Engine& engine, const std::vector<E>& values)
            {
                Tensor buffer = engine.create_tensor({static_cast<int64_t>(values.size())},
                                                     sushiblas_dtype<E>());
                if (!values.empty())
                    test_queue().memcpy(buffer.data(), values.data(), values.size() * sizeof(E))
                        .wait();
                return buffer;
            }

            /** @brief Returns the elements of a dense tensor of E. */
            template<typename E>
            [[nodiscard]] std::vector<E> read_buffer(const Tensor& buffer)
            {
                std::vector<E> host(static_cast<std::size_t>(buffer.num_elements));
                if (!host.empty())
                    test_queue().memcpy(host.data(), buffer.data(), host.size() * sizeof(E))
                        .wait();
                return host;
            }

            /** @brief Returns the first element of a tensor as E, or null without storage. */
            template<typename E>
            [[nodiscard]] E* pointer_of(Tensor& buffer)
            {
                return static_cast<E*>(buffer.data_or_null());
            }

            /** @brief Returns a dense view of a buffer's elements as dims in a layout. */
            [[nodiscard]] inline Tensor matrix_view(const Tensor& buffer,
                                             std::initializer_list<int64_t> dims,
                                             Core::Layout layout)
            {
                return Tensor(buffer.storage, dims, buffer.dtype, buffer.storage_offset, layout);
            }

            /** @brief Returns a rank-1 view of every increment-th element of a buffer. */
            [[nodiscard]] inline Tensor strided_view(const Tensor& buffer, int64_t length,
                                              int64_t increment)
            {
                const int64_t dims[] = {length};
                std::array<int64_t, Core::K_MAX_TENSOR_RANK> strides{};
                strides[0] = increment;
                return Tensor(buffer.storage, SushiRuntime::span<const int64_t>(dims, 1), strides,
                              buffer.dtype, buffer.storage_offset, buffer.layout);
            }

            /** @brief Expects two results to hold the same bytes. */
            template<typename E>
            void expect_same_bits(const std::vector<E>& actual, const std::vector<E>& expected)
            {
                ASSERT_EQ(actual.size(), expected.size());
                if (actual.empty())
                    return;
                EXPECT_EQ(0, std::memcmp(actual.data(), expected.data(),
                                         actual.size() * sizeof(E)));
            }

            /** @brief Expects two results to hold equal values; stops at the first that differs. */
            template<typename E>
            void expect_equal_values(const std::vector<E>& actual, const std::vector<E>& expected)
            {
                ASSERT_EQ(actual.size(), expected.size());
                for (std::size_t i = 0; i < actual.size(); ++i)
                    ASSERT_EQ(actual[i], expected[i]) << "index " << i;
            }

            /**
             * @brief Runs the triangular kernels and the public calls on host-described systems.
             */
            class TriangularFixture : public SushiBLASTest
            {
                protected:
                    /**
                     * @brief Submits the SYCL TRSM kernel on device operands at one block width.
                     * @param shape Gives the flags and the extents of one matrix of the batch.
                     * @param str_a Elements from one A to the next; 0 shares one A.
                     */
                    template<typename T>
                    [[nodiscard]] sycl::event submit_solve(const HostSolve<T>& shape, Tensor& A,
                                                           int64_t str_a, Tensor& B,
                                                           int64_t batch_size, T alpha,
                                                           int64_t width,
                                                           const std::vector<sycl::event>& deps)
                    {
                        const TriangularFlags& flags = shape.flags();
                        return Kernels::SYCL::trsm<T>(
                            test_queue(), Kernels::SYCL::TriangularSolvePlan{width}, flags.layout,
                            flags.side, flags.uplo, flags.trans, flags.diag, shape.rows(),
                            shape.cols(), alpha, pointer_of<const T>(A), shape.lda(), str_a,
                            pointer_of<T>(B), shape.ldb(), shape.rows() * shape.cols(), batch_size,
                            nullptr, deps);
                    }

                    /**
                     * @brief Returns the device B after the SYCL kernel solved a batch.
                     * @param a One dense A, or batch_size of them when str_a is not 0.
                     * @param b The whole device B, which may hold more elements than the batch.
                     */
                    template<typename T>
                    [[nodiscard]] std::vector<T> run_solve_kernel(const HostSolve<T>& shape,
                                                                  const std::vector<T>& a,
                                                                  int64_t str_a,
                                                                  const std::vector<T>& b,
                                                                  int64_t batch_size, T alpha,
                                                                  int64_t width)
                    {
                        Tensor A = device_buffer<T>(*engine, a);
                        Tensor B = device_buffer<T>(*engine, b);
                        submit_solve(shape, A, str_a, B, batch_size, alpha, width, {})
                            .wait_and_throw();
                        return read_buffer<T>(B);
                    }

                    /** @brief Returns X of one system, solved by the SYCL kernel at one width. */
                    template<typename T>
                    [[nodiscard]] std::vector<T> solve_with_kernel(const HostSolve<T>& system,
                                                                   const std::vector<T>& b, T alpha,
                                                                   int64_t width)
                    {
                        return run_solve_kernel(system, system.stored_matrix(), 0, b, 1, alpha,
                                                width);
                    }

                    /** @brief Returns X of one system, solved by Level3::trsm in a mode. */
                    template<typename T>
                    [[nodiscard]] std::vector<T> solve_through_view(const HostSolve<T>& system,
                                                                    const std::vector<T>& b,
                                                                    double alpha,
                                                                    ExecutionMode mode)
                    {
                        const TriangularFlags& flags = system.flags();
                        set_mode(mode);

                        Tensor a_buffer = device_buffer<T>(*engine, system.stored_matrix());
                        Tensor b_buffer = device_buffer<T>(*engine, b);
                        Tensor A = matrix_view(a_buffer, {system.order(), system.order()},
                                               flags.layout);
                        Tensor B = matrix_view(b_buffer, {system.rows(), system.cols()},
                                               flags.layout);

                        SushiBLAS::BLASOps(recorder())
                            .trsm(A, B, flags.side, flags.uplo, flags.trans,
                                                         flags.diag, alpha);
                        recording->run().value();
                        return read_buffer<T>(b_buffer);
                    }

                    /** @brief Returns x of a one-vector system, solved by Level2::trsv in mode. */
                    template<typename T>
                    [[nodiscard]] std::vector<T> solve_vector_through_view(
                        const HostSolve<T>& system, const std::vector<T>& b, ExecutionMode mode)
                    {
                        const TriangularFlags& flags = system.flags();
                        set_mode(mode);

                        Tensor a_buffer = device_buffer<T>(*engine, system.stored_matrix());
                        Tensor x = device_buffer<T>(*engine, b);
                        Tensor A = matrix_view(a_buffer, {system.order(), system.order()},
                                               flags.layout);

                        SushiBLAS::BLASOps(recorder())
                            .trsv(A, x, flags.uplo, flags.trans, flags.diag);
                        recording->run().value();
                        return read_buffer<T>(x);
                    }

                    /**
                     * @brief Returns the whole buffer of x after Level2::trmv ran in a mode.
                     * @param spaced_x The elements of x, increment apart.
                     */
                    template<typename T>
                    [[nodiscard]] std::vector<T> multiply_through_view(
                        const HostSolve<T>& system, const std::vector<T>& spaced_x,
                        int64_t increment, ExecutionMode mode)
                    {
                        const TriangularFlags& flags = system.flags();
                        set_mode(mode);

                        Tensor a_buffer = device_buffer<T>(*engine, system.stored_matrix());
                        Tensor x_buffer = device_buffer<T>(*engine, spaced_x);
                        Tensor A = matrix_view(a_buffer, {system.order(), system.order()},
                                               flags.layout);
                        Tensor x = strided_view(x_buffer, system.order(), increment);

                        SushiBLAS::BLASOps(recorder())
                            .trmv(A, x, flags.uplo, flags.trans, flags.diag);
                        recording->run().value();
                        return read_buffer<T>(x_buffer);
                    }

                    /**
                     * @brief Returns the whole buffer of x after the SYCL TRMV kernel ran in one
                     *        row shape.
                     * @param group_width 0 for one work-item per row, else the work-group width.
                     */
                    template<typename T>
                    [[nodiscard]] std::vector<T> multiply_with_kernel(
                        const HostSolve<T>& system, const std::vector<T>& spaced_x,
                        int64_t increment, std::size_t group_width)
                    {
                        const TriangularFlags& flags = system.flags();
                        const std::size_t order = static_cast<std::size_t>(system.order());

                        Tensor A = device_buffer<T>(*engine, system.stored_matrix());
                        Tensor x = device_buffer<T>(*engine, spaced_x);
                        Tensor work =
                            device_buffer<T>(*engine, std::vector<T>(order, element_of<T>(0.0)));

                        Kernels::SYCL::trmv<T>(test_queue(), group_width, flags.layout, flags.uplo,
                                               flags.trans, flags.diag, system.order(),
                                               pointer_of<const T>(A), system.lda(),
                                               pointer_of<T>(x), increment, pointer_of<T>(work),
                                               {})
                            .wait_and_throw();
                        return read_buffer<T>(x);
                    }

                    /**
                     * @brief Submits the SYCL TPSV kernel on device operands at one block width.
                     * @param AP The packed storage of system's triangle.
                     */
                    template<typename T>
                    [[nodiscard]] sycl::event submit_packed_solve(
                        const HostSolve<T>& system, Tensor& AP, Tensor& x, int64_t increment,
                        int64_t width, const std::vector<sycl::event>& deps)
                    {
                        const TriangularFlags& flags = system.flags();
                        return Kernels::SYCL::tpsv<T>(
                            test_queue(), Kernels::SYCL::TriangularSolvePlan{width}, flags.layout,
                            flags.uplo, flags.trans, flags.diag, system.order(),
                            pointer_of<const T>(AP), pointer_of<T>(x), increment, nullptr, deps);
                    }

                    /**
                     * @brief Returns the whole buffer of x after the SYCL TPSV kernel solved it.
                     * @param full The matrix system was built from.
                     * @param spaced_b The right-hand side, increment apart.
                     */
                    template<typename T>
                    [[nodiscard]] std::vector<T> solve_packed_with_kernel(
                        const HostSolve<T>& system, const std::vector<T>& full,
                        const std::vector<T>& spaced_b, int64_t increment, int64_t width)
                    {
                        Tensor AP = device_buffer<T>(
                            *engine, packed_storage(system.flags(), system.order(), full));
                        Tensor x = device_buffer<T>(*engine, spaced_b);
                        submit_packed_solve(system, AP, x, increment, width, {}).wait_and_throw();
                        return read_buffer<T>(x);
                    }

                    /**
                     * @brief Returns the whole buffer of x after the SYCL TBSV kernel solved it.
                     * @param full The matrix system was built from, zero beyond k off-diagonals.
                     * @param spaced_b The right-hand side, increment apart.
                     */
                    template<typename T>
                    [[nodiscard]] std::vector<T> solve_band_with_kernel(
                        const HostSolve<T>& system, int64_t k, const std::vector<T>& full,
                        const std::vector<T>& spaced_b, int64_t increment, int64_t width)
                    {
                        const TriangularFlags& flags = system.flags();
                        Tensor A = device_buffer<T>(
                            *engine, band_storage(flags, system.order(), k, full));
                        Tensor x = device_buffer<T>(*engine, spaced_b);

                        Kernels::SYCL::tbsv<T>(
                            test_queue(), Kernels::SYCL::TriangularSolvePlan{width}, flags.layout,
                            flags.uplo, flags.trans, flags.diag, system.order(), k,
                            pointer_of<const T>(A), k + 1, pointer_of<T>(x), increment, nullptr,
                            {})
                            .wait_and_throw();
                        return read_buffer<T>(x);
                    }

                    /** @brief Returns x of a packed system, solved by Level2::tpsv in ForceSYCL. */
                    template<typename T>
                    [[nodiscard]] std::vector<T> solve_packed_through_view(
                        const HostSolve<T>& system, const std::vector<T>& full,
                        const std::vector<T>& b)
                    {
                        const TriangularFlags& flags = system.flags();
                        set_mode(ExecutionMode::ForceSYCL);

                        Tensor buffer = device_buffer<T>(
                            *engine, packed_storage(flags, system.order(), full));
                        Tensor x = device_buffer<T>(*engine, b);
                        Tensor AP = matrix_view(
                            buffer, {static_cast<int64_t>(buffer.num_elements)}, flags.layout);

                        SushiBLAS::BLASOps(recorder())
                            .tpsv(AP, x, flags.uplo, flags.trans, flags.diag);
                        recording->run().value();
                        return read_buffer<T>(x);
                    }

                    /** @brief Returns x of a band system, solved by Level2::tbsv in ForceSYCL. */
                    template<typename T>
                    [[nodiscard]] std::vector<T> solve_band_through_view(
                        const HostSolve<T>& system, int64_t k, const std::vector<T>& full,
                        const std::vector<T>& b)
                    {
                        const TriangularFlags& flags = system.flags();
                        const bool by_rows = (flags.layout == Core::Layout::RowMajor);
                        set_mode(ExecutionMode::ForceSYCL);

                        Tensor buffer = device_buffer<T>(
                            *engine, band_storage(flags, system.order(), k, full));
                        Tensor x = device_buffer<T>(*engine, b);
                        Tensor A = by_rows
                            ? matrix_view(buffer, {system.order(), k + 1}, flags.layout)
                            : matrix_view(buffer, {k + 1, system.order()}, flags.layout);

                        SushiBLAS::BLASOps(recorder()).tbsv(A, x, flags.uplo, k, flags.trans,
                                                         flags.diag);
                        recording->run().value();
                        return read_buffer<T>(x);
                    }

                    /** @brief Returns the unit roundoff of the type the kernel of mode sums in. */
                    template<typename T>
                    [[nodiscard]] static double summing_unit(ExecutionMode mode) noexcept
                    {
                        return mode == ExecutionMode::ForceReference
                                   ? unit_roundoff<Kernels::ReferenceAccumulator<T>>()
                                   : unit_roundoff<Kernels::Accumulator<T>>();
                    }

                    /**
                     * @brief Returns the roundings the kernel of mode makes on a sum of len terms.
                     * @param fixed The roundings counted whatever the length.
                     */
                    template<typename T>
                    [[nodiscard]] static std::size_t roundings(ExecutionMode mode, int64_t len,
                                                               std::size_t fixed) noexcept
                    {
                        const bool compensated = mode == ExecutionMode::ForceReference &&
                                                 Kernels::ReferenceSum<T>::K_IS_COMPENSATED;
                        return (compensated ? K_COMPENSATED_ROUNDINGS
                                            : static_cast<std::size_t>(len)) + fixed;
                    }

                    /**
                     * @brief Expects every step of a solved real system to leave a residual within
                     *        the rounding bound of the step's accumulation.
                     * @param x The solution a kernel of mode wrote over b.
                     */
                    template<typename T>
                    void expect_residual_within_bound(const HostSolve<T>& system,
                                                      const std::vector<T>& b, double alpha,
                                                      const std::vector<T>& x, ExecutionMode mode)
                    {
                        ASSERT_EQ(x.size(), b.size());
                        const double unit = summing_unit<T>(mode);

                        for (int64_t v = 0; v < system.vectors(); ++v)
                        {
                            for (int64_t s = 0; s < system.order(); ++s)
                            {
                                TruthSum sum;
                                sum.add_product(alpha,
                                                static_cast<double>(b[system.unknown(s, v)]));

                                int64_t len = 0;
                                for (int64_t t = 0; t < system.order(); ++t)
                                {
                                    if (!system.is_stored(s, t))
                                        continue;
                                    sum.add_product(-static_cast<double>(system.coefficient(s, t)),
                                                    static_cast<double>(x[system.unknown(t, v)]));
                                    if (t != s)
                                        ++len;
                                }

                                const TruthValue residual = sum.value();
                                const double bound =
                                    rounding_bound(roundings<T>(mode, len, K_STEP_ROUNDINGS), unit,
                                                   residual.magnitude) +
                                    2.0 * unit_roundoff<T>() * residual.magnitude;
                                ASSERT_LE(std::fabs(residual.value), bound)
                                    << "step " << s << ", vector " << v;
                            }
                        }
                    }

                    /**
                     * @brief Expects every row of a real triangular product within its rounding
                     *        bound of the truth.
                     * @param actual The product a kernel of mode wrote over x.
                     */
                    template<typename T>
                    void expect_product_within_bound(const HostSolve<T>& system,
                                                     const std::vector<T>& x,
                                                     const std::vector<T>& actual,
                                                     ExecutionMode mode)
                    {
                        ASSERT_EQ(actual.size(), x.size());
                        const double unit = summing_unit<T>(mode);

                        for (int64_t s = 0; s < system.order(); ++s)
                        {
                            TruthSum sum;
                            int64_t len = 0;
                            for (int64_t t = 0; t < system.order(); ++t)
                            {
                                if (!system.is_stored(s, t))
                                    continue;
                                const T factor = system.coefficient(s, t);
                                sum.add_product(static_cast<double>(factor),
                                                static_cast<double>(x[system.unknown(t, 0)]));
                                ++len;
                            }

                            const TruthValue truth = sum.value();
                            const double bound =
                                rounding_bound(roundings<T>(mode, len, K_FIXED_ROUNDINGS), unit,
                                               truth.magnitude) +
                                unit_roundoff<T>() * std::fabs(truth.value);
                            ASSERT_NEAR(static_cast<double>(actual[system.unknown(s, 0)]),
                                        truth.value, bound)
                                << "row " << s;
                        }
                    }
            };

            /** @brief Returns the flags of a row-major, left-side, non-unit call. */
            [[nodiscard]] inline TriangularFlags plain_flags(Core::Uplo uplo,
                                                      Core::Transpose trans) noexcept
            {
                TriangularFlags flags;
                flags.uplo = uplo;
                flags.trans = trans;
                return flags;
            }
            /** @brief Returns the values of a constant array as a vector, for a row builder. */
            template<std::size_t N>
            [[nodiscard]] std::vector<int64_t> values_of(const int64_t (&values)[N])
            {
                return std::vector<int64_t>(values, values + N);
            }

            /**
             * @brief Holds the matrices of one batched solve: the host description of each, the
             *        stored A of the batch and the right-hand sides.
             * @tparam T The element type.
             */
            template<typename T>
            struct SolveBatch
            {
                std::vector<HostSolve<T>> systems; /**< The systems, one per matrix. */
                std::vector<T> a;                  /**< The stored A of the batch. */
                std::vector<T> b;                  /**< The right-hand sides of the batch. */
            };

            /**
             * @brief Returns a batch of K_BATCH systems of order K_ORDER.
             * @param shares_a Whether every matrix uses the A of the first.
             */
            template<typename T>
            [[nodiscard]] SolveBatch<T> make_batch(const TriangularFlags& flags, bool shares_a,
                                                   Draws& draws)
            {
                SolveBatch<T> batch;
                std::vector<T> full;
                for (int64_t k = 0; k < K_BATCH; ++k)
                {
                    if (k == 0 || !shares_a)
                    {
                        full = dominant_matrix<T>(K_ORDER, draws);
                        const std::vector<T> stored =
                            HostSolve<T>(flags, K_ORDER, K_VECTORS, full).stored_matrix();
                        batch.a.insert(batch.a.end(), stored.begin(), stored.end());
                    }
                    batch.systems.emplace_back(flags, K_ORDER, K_VECTORS, full);

                    const std::vector<T> rhs =
                        fraction_values<T>(batch.systems.back().elements(), draws);
                    batch.b.insert(batch.b.end(), rhs.begin(), rhs.end());
                }
                return batch;
            }

            /** @brief Holds the element types of the triangular suites: every one but HALF. */
            inline const std::vector<Core::DataType> K_TRIANGULAR_TYPES = {
                Core::DataType::Float32, Core::DataType::Float64, Core::DataType::Complex32,
                Core::DataType::Complex64};

            /** @brief Holds the element types a double-precision truth exists for. */
            inline const std::vector<Core::DataType> K_TRIANGULAR_REAL_TYPES = {
                Core::DataType::Float32, Core::DataType::Float64};

            using Test::axis_name;

            /** @brief Returns the tail of a row named by a set of flags. */
            [[nodiscard]] inline std::string axis_name(const TriangularFlags& flags)
            {
                return "_" + word(flags.layout) + "_" + word(flags.side) + "_" + word(flags.uplo) +
                       "_" + word(flags.trans) + "_" + word(flags.diag);
            }

            /** @brief Holds the flags of a call and the block width it is solved at. */
            struct FlagsWidth
            {
                TriangularFlags flags; /**< The flags of the call. */
                int64_t width;         /**< The block width of the solve. */
            };

            /** @brief Returns the tail of a row named by flags and a block width. */
            [[nodiscard]] inline std::string axis_name(const FlagsWidth& axis)
            {
                return axis_name(axis.flags) + "_w" + std::to_string(axis.width);
            }

            /**
             * @brief Holds one row of a triangular suite: the element type and the one axis the
             *        cases of the suite vary besides it.
             * @tparam Axis NoAxis, or the type of the value that completes the row.
             */
            template<typename Axis>
            struct TriangularRow
            {
                Core::DataType type; /**< The element type of every operand. */
                Axis axis;           /**< The value that completes the row. */
            };

            /** @brief Returns every row over the given element types and axis values. */
            template<typename Axis>
            [[nodiscard]] std::vector<TriangularRow<Axis>> rows_over(
                const std::vector<Core::DataType>& types, const std::vector<Axis>& axes)
            {
                std::vector<TriangularRow<Axis>> rows;
                for (const Core::DataType type : types)
                    for (const Axis& axis : axes)
                        rows.push_back({type, axis});
                return rows;
            }

            /** @brief Returns the rows of a suite that has no axis of its own. */
            [[nodiscard]] inline std::vector<TriangularRow<NoAxis>> plain_rows_over(
                const std::vector<Core::DataType>& types)
            {
                return rows_over(types, std::vector<NoAxis>{NoAxis{}});
            }

            /** @brief Returns whether a data type is one of the two complex types. */
            [[nodiscard]] inline bool is_complex_type(Core::DataType type) noexcept
            {
                return type == Core::DataType::Complex32 || type == Core::DataType::Complex64;
            }

            /** @brief Returns one row per element type and flag combination over the sides. */
            [[nodiscard]] inline std::vector<TriangularRow<TriangularFlags>> flag_rows(
                const std::vector<Core::DataType>& types, std::initializer_list<Core::Side> sides)
            {
                std::vector<TriangularRow<TriangularFlags>> rows;
                for (const Core::DataType type : types)
                    for (const TriangularFlags& flags : flags_over(is_complex_type(type), sides))
                        rows.push_back({type, flags});
                return rows;
            }

            /** @brief Returns one row per element type, flag combination and block width. */
            [[nodiscard]] inline std::vector<TriangularRow<FlagsWidth>> width_rows(
                const std::vector<Core::DataType>& types, std::initializer_list<Core::Side> sides,
                const std::vector<int64_t>& widths)
            {
                std::vector<TriangularRow<FlagsWidth>> rows;
                for (const TriangularRow<TriangularFlags>& row : flag_rows(types, sides))
                    for (const int64_t width : widths)
                        rows.push_back({row.type, FlagsWidth{row.axis, width}});
                return rows;
            }

            /** @brief Returns the name of a row for GoogleTest: element type, then the axis. */
            template<typename Axis>
            [[nodiscard]] std::string triangular_row_name(
                const ::testing::TestParamInfo<TriangularRow<Axis>>& info)
            {
                return Core::to_string(info.param.type) + axis_name(info.param.axis);
            }

            /**
             * @brief Runs a case once per row: skips a device that lacks the element type of the
             *        row, then gives the case the fixture of the triangular kernels.
             * @tparam Axis NoAxis, or the type of the value that completes each row.
             */
            template<typename Axis>
            class TriangularCase : public TriangularFixture,
                                   public ::testing::WithParamInterface<TriangularRow<Axis>>
            {
                protected:
                    /** @brief Builds the engine, or skips the row. */
                    void SetUp() override
                    {
                        TriangularFixture::SetUp();
                        if (!device_runs(this->GetParam().type))
                            GTEST_SKIP() << "the device does not compute in this element type";
                    }
            };
        } // namespace TriangularSupport
    } // namespace Test
} // namespace SushiBLAS
