// The public op-dispatch seam, exercised the way a downstream consumer would.
//
// This translation unit reaches SushiBLAS::Dispatch through installed public headers and
// hands it functors this file wrote - which is the whole point of promoting those five
// headers out of src/. It is the in-tree half of the proof; the out-of-tree half is
// tests/package/main.cpp, which does the same against a find_package()'d install.

#include <vector>
#include <stdexcept>
#include <gtest/gtest.h>
#include <SushiBLAS/SushiBLAS.h>
#include <SushiBLAS/engine/dispatch.hpp>
#include <SushiBLAS/engine/logic/dispatch.hpp>
#include <SushiBLAS/engine/math/safe_math.hpp>
#include <SushiBLAS/engine/math/elementwise/dispatch.hpp>
#include <SushiBLAS/engine/math/random/dispatch.hpp>
#include <SushiBLAS/engine/math/reductions/dispatch.hpp>
#include "../../common/test_utils.hpp"

using namespace SushiBLAS::Test;
using namespace SushiRuntime::Graph::Literals;
using SushiBLAS::Core::DataType;

class Unit_DispatchSeamTest : public SushiBLASTest {};

TEST_F(Unit_DispatchSeamTest, FusesAConsumerFunctorIntoOneBinaryKernel)
{
    auto a = engine->create_tensor({4});
    auto b = engine->create_tensor({4});
    auto c = engine->create_tensor({4});
    fill_tensor(a, {1.0f, 2.0f, 3.0f, 4.0f});
    fill_tensor(b, {2.0f, 4.0f, 6.0f, 8.0f});

    // mul, add and div composed into one functor: three ops the public API would have
    // recorded as three tasks and three kernel launches, collapsed into one.
    sb::Dispatch::execute_binary(*engine, a, b, c, "test.fused_binary", "test.fused_binary"_op,
        [](auto x, auto y) { return (x * y + decltype(x)(1)) / decltype(x)(2); });
    engine->execute().wait();

    verify_tensor(c, {1.5f, 4.5f, 9.5f, 16.5f});
}

TEST_F(Unit_DispatchSeamTest, FusesAConsumerFunctorIntoOneUnaryKernel)
{
    auto t = engine->create_tensor({4});
    fill_tensor(t, {0.0f, 1.0f, 2.0f, 3.0f});

    // A consumer's own activation, written against safe_math rather than raw SYCL
    // builtins: softplus-like, but nothing in SushiBLAS's op catalogue.
    sb::Dispatch::execute_unary_inplace(*engine, t, "test.fused_unary", "test.fused_unary"_op,
        [](auto x) { return sb::Dispatch::safe_log(decltype(x)(1) + sb::Dispatch::safe_exp(x)); });
    engine->execute().wait();

    verify_tensor(t, {0.693147f, 1.313262f, 2.126928f, 3.048587f}, 1e-4);
}

TEST_F(Unit_DispatchSeamTest, ConsumerFunctorCanOptIntoComplex)
{
    using cf = std::complex<float>;

    auto a = create_initialized_tensor<cf>({2}, {cf(1.0f, 1.0f), cf(2.0f, -1.0f)});
    auto b = create_initialized_tensor<cf>({2}, {cf(0.0f, 1.0f), cf(1.0f, 1.0f)});
    auto c = engine->create_tensor({2}, DataType::COMPLEX32);

    // (1+i)(i) + (1+i) = (-1+i) + (1+i) = 2i
    // (2-i)(1+i) + (2-i) = (3+i) + (2-i) = 5
    sb::Dispatch::execute_binary<sb::Dispatch::Domain::REAL_AND_COMPLEX>(
        *engine, a, b, c, "test.fused_complex", "test.fused_complex"_op,
        [](auto x, auto y) { return x * y + x; });
    engine->execute().wait();

    auto expected = create_initialized_tensor<cf>({2}, {cf(0.0f, 2.0f), cf(5.0f, 0.0f)});
    SB_ASSERT_TENSOR_NEAR(c, expected, 1e-4, cf);
}

TEST_F(Unit_DispatchSeamTest, ConsumerCanDriveTheLogicAndReductionSeamsToo)
{
    auto a = engine->create_tensor({4});
    auto b = engine->create_tensor({4});
    auto mask = engine->create_tensor({4});
    fill_tensor(a, {1.0f, 5.0f, 2.0f, 9.0f});
    fill_tensor(b, {3.0f, 3.0f, 3.0f, 3.0f});

    // "within 2 of b" - a predicate SushiBLAS does not ship.
    sb::Dispatch::execute_logic_binary(*engine, a, b, mask, "test.near", "test.near"_op,
        [](auto x, auto y)
        {
            auto d = x - y;
            return (d < decltype(x)(0) ? -d : d) <= decltype(x)(2) ? decltype(x)(1) : decltype(x)(0);
        });
    engine->execute().wait();
    verify_tensor(mask, {1.0f, 1.0f, 1.0f, 0.0f});

    auto total = engine->create_tensor({1});
    sb::Dispatch::execute_reduction(*engine, mask, total, "test.count", "test.count"_op,
        []<typename T>(sycl::queue& q, const T* pT, T* pR, int64_t size, const std::vector<sycl::event>& deps)
        {
            return q.submit([&](sycl::handler& h)
            {
                h.depends_on(deps);
                h.parallel_for(sycl::range<1>(size),
                    sycl::reduction(pR, T(0), sycl::plus<T>(), sycl::property::reduction::initialize_to_identity{}),
                    [=](sycl::id<1> idx, auto& reducer) { reducer.combine(pT[idx[0]]); });
            });
        });
    engine->execute().wait();
    verify_tensor(total, {3.0f});
}

TEST_F(Unit_DispatchSeamTest, RealOnlyDomainRejectsComplexAtRecordTime)
{
    auto a = engine->create_tensor({2}, DataType::COMPLEX32);
    auto b = engine->create_tensor({2}, DataType::COMPLEX32);
    auto c = engine->create_tensor({2}, DataType::COMPLEX32);

    // The default domain is REAL, so a functor that was only ever meant for real scalars
    // is rejected before it can be handed a complex operand - rather than compiled for
    // one, or (as before) silently skipped.
    EXPECT_THROW(
        sb::Dispatch::execute_binary(*engine, a, b, c, "test.real_only", "test.real_only"_op,
                                     [](auto x, auto y) { return x + y; }),
        std::runtime_error);
}

// --- Task metadata parameter slots -------------------------------------------------

TEST_F(Unit_DispatchSeamTest, RngMetadataSlotsAreInRangeAndRoundTrip)
{
    // The bug: the random dispatch wrote the seed to slot 10 and the offset to slot 11 of
    // an eight-slot array. TaskMetadata::set_param ignores an out-of-range index without
    // a word, so every random task recorded a zero seed and a zero offset. With the fixed
    // slots these read back exactly, including a seed whose value would not survive the
    // round trip through a double.
    constexpr uint64_t kSeed = 0xDEADBEEFCAFEF00Dull;
    constexpr uint64_t kOffset = 42;

    static_assert(sb::Dispatch::K_RNG_SEED_PARAM < static_cast<int>(SushiRuntime::Graph::TaskMetadata::K_MAX_PARAMS));
    static_assert(sb::Dispatch::K_RNG_OFFSET_PARAM < static_cast<int>(SushiRuntime::Graph::TaskMetadata::K_MAX_PARAMS));

    SushiRuntime::Graph::TaskMetadata meta;
    sb::Dispatch::set_rng_params(meta, kSeed, kOffset);

    EXPECT_EQ(meta.get_param<uint64_t>(sb::Dispatch::K_RNG_SEED_PARAM), kSeed);
    EXPECT_EQ(meta.get_param<uint64_t>(sb::Dispatch::K_RNG_OFFSET_PARAM), kOffset);
}

TEST_F(Unit_DispatchSeamTest, DistributionParamsDoNotCollideWithTheRngSlots)
{
    SushiRuntime::Graph::TaskMetadata meta;
    const std::vector<double> params = {1.0, 2.0, 3.0, 4.0};

    sb::Dispatch::set_task_params(meta, params, "test.params",
                                  SushiRuntime::Graph::TaskMetadata::K_MAX_PARAMS - sb::Dispatch::K_MAX_RNG_PARAMS);
    sb::Dispatch::set_rng_params(meta, 7, 8);

    for (size_t i = 0; i < params.size(); ++i)
    {
        EXPECT_DOUBLE_EQ(meta.get_param<double>(static_cast<int>(i)), params[i]);
    }
    EXPECT_EQ(meta.get_param<uint64_t>(sb::Dispatch::K_RNG_SEED_PARAM), 7u);
    EXPECT_EQ(meta.get_param<uint64_t>(sb::Dispatch::K_RNG_OFFSET_PARAM), 8u);
}

TEST_F(Unit_DispatchSeamTest, TooManyTaskParamsThrowsInsteadOfDroppingThem)
{
    auto t = engine->create_tensor({4});
    const std::vector<float> too_many(SushiRuntime::Graph::TaskMetadata::K_MAX_PARAMS + 1, 1.0f);

    EXPECT_THROW(
        sb::Dispatch::execute_unary_inplace(*engine, t, "test.overflow", "test.overflow"_op,
                                            [](auto x) { return x; }, too_many),
        std::runtime_error);
}
