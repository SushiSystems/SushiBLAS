/**************************************************************************/
/* test_convert.cpp                                                       */
/**************************************************************************/
/*                          This file is part of:                         */
/*                                SushiBLAS                               */
/*                https://github.com/SushiSystems/SushiBLAS               */
/*                         https://sushisystems.io                        */
/**************************************************************************/
/* Copyright (c) 2026-present Mustafa Garip & Sushi Systems               */
/*                                                                        */
/* Licensed under the Apache License, Version 2.0 (the "License");        */
/* you may not use this file except in compliance with the License.       */
/* You may obtain a copy of the License at                                */
/*                                                                        */
/*     http://www.apache.org/licenses/LICENSE-2.0                         */
/*                                                                        */
/* Unless required by applicable law or agreed to in writing, software    */
/* distributed under the License is distributed on an "AS IS" BASIS,      */
/* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or        */
/* implied. See the License for the specific language governing           */
/* permissions and limitations under the License.                         */
/**************************************************************************/

// `elementwise().convert` - the dtype-converting copy.
//
// The values here are chosen so the assertions cannot pass without the
// conversion actually happening. A round trip of small integers through fp16 is
// a true statement about almost any implementation, including a `memcpy` that
// got the widths right by luck, so it appears below only as an explicitly
// labelled sanity case. Everything else asserts on a value that *changes*: an
// fp32 number fp16 cannot represent must come back as the correctly *rounded*
// fp16 neighbour (not merely "close"), one past fp16's range must come back as
// an infinity, and one below it must come back as zero.
//
// The expectations are exact, and they are worked out by hand from IEEE-754
// binary16 rather than taken from another conversion in this tree - a
// differential oracle against `static_cast` on the host would agree with a
// device kernel that made the same mistake.
//
// The ordering cases assert on `in_degree`, never on an output value, for the
// reason Regression_Sentinel.ReductionsOrderAgainstViewsOfOneAllocation gives at
// length: with no dependency edge the tasks may still come out in the right
// order by luck, so a correct number proves nothing about ordering.

#include <cmath>
#include <limits>
#include <string>
#include <vector>
#include <cstddef>
#include <cstring>
#include <stdexcept>
#include <gtest/gtest.h>
#include <SushiBLAS/SushiBLAS.h>
#include "../../common/test_utils.hpp"

namespace SushiBLAS
{
    namespace Test
    {
        namespace
        {
            /** @brief The three dtypes `convert` serves, in the order the switch lists them. */
            const std::vector<Core::DataType> kRealDTypes =
                {Core::DataType::HALF, Core::DataType::FLOAT32, Core::DataType::FLOAT64};

            /**
             * @brief The nearest IEEE-754 binary16 value to 0.1, as an exact double.
             *
             * 0.1 = 1.100110011001100110... x 2^-4. Keeping ten mantissa bits gives
             * 1.1001100110 (the first dropped bit is 0, so it rounds down), which is
             * 1.599609375 x 2^-4. Not equal to 0.1f, which is the point.
             */
            constexpr double kTenthAsHalf = 0.0999755859375;

            /** @brief The largest finite binary16 value. */
            constexpr double kHalfMax = 65504.0;

            /**
             * @brief 2 x 2^-24: the binary16 subnormal nearest to 1e-7.
             *
             * 1e-7 / 2^-24 is 1.677..., which rounds to 2, so the conversion lands on a
             * subnormal rather than flushing to zero.
             */
            constexpr double kTinyAsHalf = 1.1920928955078125e-7;
        }

        class Unit_ConvertTest : public SushiBLASTest
        {
            protected:
                /**
                 * @brief Reads a whole real tensor back to the host, widened to double.
                 *
                 * Widening HALF or FLOAT32 to double is exact, so an `EXPECT_EQ` against a
                 * double literal below is a statement about the stored bits and not about
                 * a tolerance - which is what these cases need, since "close to 0.1" is
                 * true of both the right answer and the wrong one.
                 */
                std::vector<double> read_exact(const Tensor& t)
                {
                    const size_t n = static_cast<size_t>(t.num_elements);
                    std::vector<double> out(n);
                    auto& q = TestEnvironment::get_context().get_queue();

                    if (t.dtype == Core::DataType::HALF)
                    {
                        std::vector<sycl::half> host(n);
                        q.memcpy(host.data(), t.data(), n * sizeof(sycl::half)).wait();
                        for (size_t i = 0; i < n; ++i) out[i] = static_cast<double>(host[i]);
                    }
                    else if (t.dtype == Core::DataType::FLOAT32)
                    {
                        std::vector<float> host(n);
                        q.memcpy(host.data(), t.data(), n * sizeof(float)).wait();
                        for (size_t i = 0; i < n; ++i) out[i] = static_cast<double>(host[i]);
                    }
                    else
                    {
                        q.memcpy(out.data(), t.data(), n * sizeof(double)).wait();
                    }
                    return out;
                }

                /** @brief Writes host doubles into a tensor of any of the three real dtypes. */
                void write_exact(Tensor& t, const std::vector<double>& values)
                {
                    const size_t n = values.size();
                    ASSERT_EQ(n, static_cast<size_t>(t.num_elements));
                    auto& q = TestEnvironment::get_context().get_queue();

                    if (t.dtype == Core::DataType::HALF)
                    {
                        std::vector<sycl::half> host(n);
                        for (size_t i = 0; i < n; ++i) host[i] = static_cast<sycl::half>(static_cast<float>(values[i]));
                        q.memcpy(t.data(), host.data(), n * sizeof(sycl::half)).wait();
                    }
                    else if (t.dtype == Core::DataType::FLOAT32)
                    {
                        std::vector<float> host(n);
                        for (size_t i = 0; i < n; ++i) host[i] = static_cast<float>(values[i]);
                        q.memcpy(t.data(), host.data(), n * sizeof(float)).wait();
                    }
                    else
                    {
                        q.memcpy(t.data(), values.data(), n * sizeof(double)).wait();
                    }
                }

                /** @brief The message of the `std::runtime_error` @p fn throws, or "" if it does not. */
                template<typename Fn>
                std::string message_of(Fn&& fn)
                {
                    try
                    {
                        fn();
                    }
                    catch (const std::runtime_error& e)
                    {
                        return std::string(e.what());
                    }
                    return std::string();
                }
        };

        // =====================================================================
        // The cases the conversion has to change a value to pass
        // =====================================================================

        // fp32 -> fp16 on a value fp16 cannot hold. The assertion is on the exact
        // correctly-rounded neighbour, and there is a second assertion that the result
        // is NOT the input - so an implementation that widened, reinterpreted, or
        // otherwise carried the fp32 value through unchanged fails here.
        TEST_F(Unit_ConvertTest, Fp32ToFp16RoundsToTheNearestRepresentableValue)
        {
            auto src = engine->create_tensor({1}, Core::DataType::FLOAT32);
            auto dst = engine->create_tensor({1}, Core::DataType::HALF);
            write_exact(src, {static_cast<double>(0.1f)});

            engine->elementwise().convert(src, dst);
            engine->execute().wait();

            const auto got = read_exact(dst);
            EXPECT_EQ(kTenthAsHalf, got[0])
                << "0.1f must arrive as the binary16 value nearest it, 0.0999755859375.";
            EXPECT_NE(static_cast<double>(0.1f), got[0])
                << "0.1 is not representable in binary16, so a result equal to the fp32 "
                   "input means no conversion happened at all.";
        }

        // Round-to-nearest-*even*, not round-away and not truncate. 1 + 2^-11 sits
        // exactly halfway between the binary16 neighbours 1.0 (mantissa even) and
        // 1 + 2^-10 (mantissa odd), so only ties-to-even gives 1.0. The second value is
        // just past halfway and must go the other way, which is what stops "always round
        // down" from passing the first one.
        TEST_F(Unit_ConvertTest, Fp32ToFp16BreaksTiesToEven)
        {
            auto src = engine->create_tensor({2}, Core::DataType::FLOAT32);
            auto dst = engine->create_tensor({2}, Core::DataType::HALF);
            write_exact(src, {1.0 + 0.00048828125,        // 1 + 2^-11, exactly halfway
                              1.0 + 0.000732421875});     // 1 + 3*2^-12, past halfway

            engine->elementwise().convert(src, dst);
            engine->execute().wait();

            const auto got = read_exact(dst);
            EXPECT_EQ(1.0, got[0]) << "A tie must resolve to the neighbour with the even mantissa.";
            EXPECT_EQ(1.0 + 0.0009765625, got[1]) << "Past halfway must round up to 1 + 2^-10.";
        }

        // The behaviour a loss-scaling backoff is built on: a gradient scaled past
        // fp16's range does not wrap, saturate or clamp - it becomes an infinity, which
        // is a thing `count_nonfinite` can then find.
        TEST_F(Unit_ConvertTest, Fp32PastTheHalfRangeBecomesInfinity)
        {
            auto src = engine->create_tensor({4}, Core::DataType::FLOAT32);
            auto dst = engine->create_tensor({4}, Core::DataType::HALF);
            write_exact(src, {kHalfMax, 70000.0, -70000.0, 1.0e30});

            engine->elementwise().convert(src, dst);
            engine->execute().wait();

            const auto got = read_exact(dst);
            EXPECT_EQ(kHalfMax, got[0]) << "The largest finite binary16 value must stay finite.";
            EXPECT_TRUE(std::isinf(got[1]) && got[1] > 0.0) << "70000 is past binary16's range.";
            EXPECT_TRUE(std::isinf(got[2]) && got[2] < 0.0) << "The sign must survive the overflow.";
            EXPECT_TRUE(std::isinf(got[3]) && got[3] > 0.0);
        }

        // The other end of the range. 1e-8 is below half of the smallest subnormal and
        // must flush to zero; 1e-7 is not, and must land on a subnormal rather than
        // being flushed with it.
        TEST_F(Unit_ConvertTest, Fp32BelowTheHalfRangeUnderflowsToZeroButSubnormalsSurvive)
        {
            auto src = engine->create_tensor({2}, Core::DataType::FLOAT32);
            auto dst = engine->create_tensor({2}, Core::DataType::HALF);
            write_exact(src, {1.0e-8, 1.0e-7});

            engine->elementwise().convert(src, dst);
            engine->execute().wait();

            const auto got = read_exact(dst);
            EXPECT_EQ(0.0, got[0]) << "1e-8 is below half the smallest binary16 subnormal.";
            EXPECT_EQ(kTinyAsHalf, got[1])
                << "1e-7 rounds to 2 * 2^-24; flushing it to zero would mean subnormals are "
                   "being discarded rather than represented.";
        }

        // Non-finite inputs pass through unchanged in both directions. This is the pairing
        // that makes `reductions().count_nonfinite` meaningful on a converted tensor: if a
        // cast quietly turned a NaN into a number, screening the fp16 copy would say the
        // fp32 original was clean.
        TEST_F(Unit_ConvertTest, NonFiniteValuesSurviveTheConversion)
        {
            auto src = engine->create_tensor({3}, Core::DataType::FLOAT32);
            auto down = engine->create_tensor({3}, Core::DataType::HALF);
            auto up = engine->create_tensor({3}, Core::DataType::FLOAT64);
            write_exact(src, {std::numeric_limits<double>::infinity(),
                              -std::numeric_limits<double>::infinity(),
                              std::numeric_limits<double>::quiet_NaN()});

            engine->elementwise().convert(src, down);
            engine->elementwise().convert(src, up);
            engine->execute().wait();

            for (const Tensor* t : {&down, &up})
            {
                const auto got = read_exact(*t);
                EXPECT_TRUE(std::isinf(got[0]) && got[0] > 0.0);
                EXPECT_TRUE(std::isinf(got[1]) && got[1] < 0.0);
                EXPECT_TRUE(std::isnan(got[2]));
            }
        }

        // Narrowing then widening gives the *narrow* value back, not the original. The
        // complement of the round-trip sanity case below: it says what the round trip
        // costs when the value is one fp16 cannot hold.
        TEST_F(Unit_ConvertTest, ARoundTripThroughHalfKeepsOnlyWhatHalfCouldHold)
        {
            auto src = engine->create_tensor({1}, Core::DataType::FLOAT32);
            auto narrow = engine->create_tensor({1}, Core::DataType::HALF);
            auto back = engine->create_tensor({1}, Core::DataType::FLOAT32);
            write_exact(src, {static_cast<double>(0.1f)});

            engine->elementwise().convert(src, narrow);
            engine->elementwise().convert(narrow, back);
            engine->execute().wait();

            const auto got = read_exact(back);
            EXPECT_EQ(kTenthAsHalf, got[0])
                << "Widening back must reproduce the fp16 value exactly, and must not "
                   "recover precision the narrowing threw away.";
        }

        // =====================================================================
        // The sanity case, labelled as one
        // =====================================================================

        // Exactly-representable values survive every conversion, which is necessary and
        // nowhere near sufficient - it would pass for an implementation that had the
        // element widths right and the rounding wrong. It is here to cover all nine arms
        // of the dtype cross product in one place, and that coverage is its whole job.
        TEST_F(Unit_ConvertTest, EveryRealDTypePairConvertsExactlyRepresentableValues)
        {
            const std::vector<double> values = {0.0, 1.0, -2.0, 3.5, -4.25, 1024.0};

            for (Core::DataType from : kRealDTypes)
            {
                for (Core::DataType to : kRealDTypes)
                {
                    auto src = engine->create_tensor({6}, from);
                    auto dst = engine->create_tensor({6}, to);
                    write_exact(src, values);

                    engine->elementwise().convert(src, dst);
                    engine->execute().wait();

                    const auto got = read_exact(dst);
                    for (size_t i = 0; i < values.size(); ++i)
                    {
                        EXPECT_EQ(values[i], got[i])
                            << "at index " << i << " converting " << Core::to_string(from)
                            << " -> " << Core::to_string(to);
                    }
                }
            }
        }

        // Converting a dtype to itself is legal and copies, which is the decision
        // documented on ElementwiseOps::convert: a mixed-precision policy holds its
        // compute dtype as runtime data, and when it equals the master dtype the op has
        // to degenerate into a copy rather than throw.
        TEST_F(Unit_ConvertTest, ConvertingADTypeToItselfIsAPlainCopy)
        {
            auto src = engine->create_tensor({3}, Core::DataType::FLOAT32);
            auto dst = engine->create_tensor({3}, Core::DataType::FLOAT32);
            // 2^40 rather than a decimal like 1e30: the expectation is an exact equality,
            // and a decimal that fp32 cannot hold would be rounded by `write_exact` before
            // `convert` ever saw it - making this a test of the test.
            write_exact(src, {1.5, -2.25, 1099511627776.0});

            EXPECT_NO_THROW(engine->elementwise().convert(src, dst));
            engine->execute().wait();

            const auto got = read_exact(dst);
            EXPECT_EQ(1.5, got[0]);
            EXPECT_EQ(-2.25, got[1]);
            EXPECT_EQ(1099511627776.0, got[2]);
        }

        // A multi-dimensional operand is converted whole, not just its first row.
        TEST_F(Unit_ConvertTest, ARankTwoOperandIsConvertedInFull)
        {
            auto src = engine->create_tensor({3, 4}, Core::DataType::FLOAT64);
            auto dst = engine->create_tensor({3, 4}, Core::DataType::HALF);
            std::vector<double> values(12);
            for (size_t i = 0; i < values.size(); ++i) values[i] = static_cast<double>(i) - 5.0;
            write_exact(src, values);

            engine->elementwise().convert(src, dst);
            engine->execute().wait();

            const auto got = read_exact(dst);
            for (size_t i = 0; i < values.size(); ++i) EXPECT_EQ(values[i], got[i]) << "at index " << i;
        }

        // =====================================================================
        // What it refuses, and whether the refusal says enough to act on
        // =====================================================================

        TEST_F(Unit_ConvertTest, ADTypePairOutsideTheRealDomainThrowsNamingBoth)
        {
            auto real32 = engine->create_tensor({2}, Core::DataType::FLOAT32);
            auto cplx = engine->create_tensor({2}, Core::DataType::COMPLEX32);
            auto integral = engine->create_tensor({2}, Core::DataType::INT64);

            const std::string from_complex = message_of([&]{ engine->elementwise().convert(cplx, real32); });
            EXPECT_NE(std::string::npos, from_complex.find("COMPLEX32")) << from_complex;
            EXPECT_NE(std::string::npos, from_complex.find("FLOAT32")) << from_complex;

            const std::string to_integer = message_of([&]{ engine->elementwise().convert(real32, integral); });
            EXPECT_NE(std::string::npos, to_integer.find("FLOAT32")) << to_integer;
            EXPECT_NE(std::string::npos, to_integer.find("INT64")) << to_integer;

            // And nothing was recorded: a rejected conversion must not leave a task behind.
            EXPECT_TRUE(engine->get_graph().take_nodes().empty());
        }

        TEST_F(Unit_ConvertTest, DisagreeingShapesThrowNamingWhatDisagreed)
        {
            auto src = engine->create_tensor({3, 4}, Core::DataType::FLOAT32);
            auto wrong_dim = engine->create_tensor({3, 5}, Core::DataType::HALF);
            auto wrong_rank = engine->create_tensor({12}, Core::DataType::HALF);

            const std::string dim = message_of([&]{ engine->elementwise().convert(src, wrong_dim); });
            EXPECT_NE(std::string::npos, dim.find("dimension")) << dim;

            const std::string rank = message_of([&]{ engine->elementwise().convert(src, wrong_rank); });
            EXPECT_NE(std::string::npos, rank.find("rank")) << rank;
        }

        // A slice along an inner axis keeps its parent's strides, so its elements are not
        // where a flat walk would look for them. Refused by name rather than read wrongly.
        TEST_F(Unit_ConvertTest, ANonContiguousOperandIsRefused)
        {
            auto parent = engine->create_tensor({4, 4}, Core::DataType::FLOAT32);
            auto inner = parent.slice(1, 1, 3);
            auto dst = engine->create_tensor({4, 2}, Core::DataType::HALF);

            ASSERT_FALSE(inner.is_contiguous());
            const std::string msg = message_of([&]{ engine->elementwise().convert(inner, dst); });
            EXPECT_NE(std::string::npos, msg.find("contiguous")) << msg;
        }

        // Overlapping operands are a race between work-items, not an in-place update:
        // each item reads one element and writes another of a different width.
        TEST_F(Unit_ConvertTest, OverlappingOperandsAreRefused)
        {
            constexpr int64_t kN = 8;
            auto storage = SushiRuntime::make_sushi<Storage>(
                TestEnvironment::get_context().get_allocator(),
                static_cast<size_t>(2 * kN) * sizeof(float),
                SushiRuntime::Memory::Strategies::Shared::instance());

            // Two FLOAT32 views starting four elements apart over an eight-element span:
            // they share the middle four.
            Tensor a(storage, {kN}, Core::DataType::FLOAT32, 0);
            Tensor b(storage, {kN}, Core::DataType::FLOAT32, kN / 2);

            const std::string msg = message_of([&]{ engine->elementwise().convert(a, b); });
            EXPECT_NE(std::string::npos, msg.find("overlap")) << msg;
        }

        // =====================================================================
        // What it declares to the dependency tracker
        // =====================================================================

        // **Both operands are views at a non-zero offset, and that is the whole point.**
        // At offset zero a bare `data()` pointer and a real `region()` name the same
        // resource, so an op that declared the wrong one would still get its edge and this
        // test would pass without proving anything. Partway into an arena they diverge:
        // `data()` is `base + offset` and converts to the whole-buffer region *of that
        // address*, which the tracker files under a resource the producer never touched.
        // Verified by mutation - rewriting `src.region()` as the bare pointer makes the
        // assertion below report 0.
        TEST_F(Unit_ConvertTest, AConvertIsOrderedBehindTheOpThatWroteItsSource)
        {
            constexpr int64_t kN = 4;
            auto arena = SushiRuntime::make_sushi<Storage>(
                TestEnvironment::get_context().get_allocator(),
                static_cast<size_t>(2 * kN) * sizeof(float),
                SushiRuntime::Memory::Strategies::Shared::instance());

            Tensor seed(arena, {kN}, Core::DataType::FLOAT32, 0);
            Tensor src(arena, {kN}, Core::DataType::FLOAT32, kN);
            auto dst = engine->create_tensor({kN}, Core::DataType::HALF);
            write_exact(seed, {1.0, 2.0, 3.0, 4.0});

            engine->elementwise().mul(seed, seed, src);
            engine->elementwise().convert(src, dst);

            auto nodes = engine->get_graph().take_nodes();
            ASSERT_EQ(2u, nodes.size());
            EXPECT_EQ(1, nodes[1]->in_degree.load(std::memory_order_relaxed))
                << "The convert reading `src` must depend on the op that wrote it. An "
                   "in-degree of 0 means the read was declared from a bare pointer rather "
                   "than from Tensor::region(), so the tracker filed it under a resource "
                   "nothing else touches and emitted no edge at all.";
        }

        TEST_F(Unit_ConvertTest, AReaderOfTheConvertedTensorIsOrderedBehindTheConvert)
        {
            constexpr int64_t kN = 4;
            auto arena = SushiRuntime::make_sushi<Storage>(
                TestEnvironment::get_context().get_allocator(),
                static_cast<size_t>(2 * kN) * sizeof(float),
                SushiRuntime::Memory::Strategies::Shared::instance());

            auto src = engine->create_tensor({kN}, Core::DataType::FLOAT32);
            Tensor dst(arena, {kN}, Core::DataType::FLOAT32, kN);
            auto total = engine->create_tensor({1}, Core::DataType::FLOAT32);
            write_exact(src, {1.0, 2.0, 3.0, 4.0});

            engine->elementwise().convert(src, dst);
            engine->reductions().sum(dst, total);

            auto nodes = engine->get_graph().take_nodes();
            ASSERT_EQ(2u, nodes.size());
            EXPECT_EQ(1, nodes[1]->in_degree.load(std::memory_order_relaxed))
                << "The write set must name `dst` by its own byte interval, so the "
                   "reduction reading it is ordered behind the conversion. `dst` is a view "
                   "partway into an arena, which is where a bare-pointer write set stops "
                   "agreeing with the reader's region.";
        }

        // The control, and the reason the two assertions above are worth writing: two
        // converts over *disjoint* views of one arena must NOT serialise. It is the half
        // of the hazard the two above cannot see - they catch an *under*-declared operand
        // (no edge where one is needed), this catches an *over*-declared one (an edge
        // where none is). Verified by mutation as well: declaring
        // `src.storage->allocation_base()` instead of `src.region()` makes the assertion
        // below report 1, while leaving both ordering tests above green.
        TEST_F(Unit_ConvertTest, ConvertsOverDisjointViewsOfOneArenaDoNotSerialise)
        {
            constexpr int64_t kN = 8;

            auto f32_arena = SushiRuntime::make_sushi<Storage>(
                TestEnvironment::get_context().get_allocator(),
                static_cast<size_t>(2 * kN) * sizeof(float),
                SushiRuntime::Memory::Strategies::Shared::instance());
            auto f16_arena = SushiRuntime::make_sushi<Storage>(
                TestEnvironment::get_context().get_allocator(),
                static_cast<size_t>(2 * kN) * sizeof(sycl::half),
                SushiRuntime::Memory::Strategies::Shared::instance());

            Tensor src_a(f32_arena, {kN}, Core::DataType::FLOAT32, 0);
            Tensor src_b(f32_arena, {kN}, Core::DataType::FLOAT32, kN);
            Tensor dst_a(f16_arena, {kN}, Core::DataType::HALF, 0);
            Tensor dst_b(f16_arena, {kN}, Core::DataType::HALF, kN);

            engine->elementwise().convert(src_a, dst_a);
            engine->elementwise().convert(src_b, dst_b);

            auto nodes = engine->get_graph().take_nodes();
            ASSERT_EQ(2u, nodes.size());
            EXPECT_EQ(0, nodes[0]->in_degree.load(std::memory_order_relaxed));
            EXPECT_EQ(0, nodes[1]->in_degree.load(std::memory_order_relaxed))
                << "Two conversions over non-overlapping halves of one arena share no "
                   "bytes, so neither may depend on the other. An in-degree of 1 means the "
                   "operands were described by their allocation rather than by their own "
                   "byte interval, which serialises everything a memory planner packs.";
        }

        // The offset case, behaviourally: a view partway into an arena must convert its
        // own elements and nothing else.
        TEST_F(Unit_ConvertTest, AViewAtANonZeroOffsetConvertsItsOwnElements)
        {
            constexpr int64_t kN = 4;

            auto f32_arena = SushiRuntime::make_sushi<Storage>(
                TestEnvironment::get_context().get_allocator(),
                static_cast<size_t>(2 * kN) * sizeof(float),
                SushiRuntime::Memory::Strategies::Shared::instance());
            auto f16_arena = SushiRuntime::make_sushi<Storage>(
                TestEnvironment::get_context().get_allocator(),
                static_cast<size_t>(2 * kN) * sizeof(sycl::half),
                SushiRuntime::Memory::Strategies::Shared::instance());

            Tensor head(f32_arena, {kN}, Core::DataType::FLOAT32, 0);
            Tensor tail(f32_arena, {kN}, Core::DataType::FLOAT32, kN);
            Tensor out_head(f16_arena, {kN}, Core::DataType::HALF, 0);
            Tensor out_tail(f16_arena, {kN}, Core::DataType::HALF, kN);

            write_exact(head, {1.0, 2.0, 3.0, 4.0});
            write_exact(tail, {5.0, 6.0, 7.0, 8.0});
            write_exact(out_head, {-1.0, -1.0, -1.0, -1.0});

            engine->elementwise().convert(tail, out_tail);
            engine->execute().wait();

            const auto got_tail = read_exact(out_tail);
            EXPECT_EQ(5.0, got_tail[0]);
            EXPECT_EQ(8.0, got_tail[3]);

            const auto got_head = read_exact(out_head);
            for (size_t i = 0; i < 4; ++i)
            {
                EXPECT_EQ(-1.0, got_head[i])
                    << "Converting the tail view wrote over the head at index " << i
                    << ", which is what dropping storage_offset looks like.";
            }
        }

    } // namespace Test
} // namespace SushiBLAS
