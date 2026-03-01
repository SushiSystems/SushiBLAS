#include <cmath>
#include <limits>
#include <vector>
#include <random>
#include <numeric>
#include <gtest/gtest.h>
#include <SushiBLAS/SushiBLAS.h>
#include "test_common.hpp"
#include "../../kernels/reference_reductions.hpp"

using namespace SushiBLAS;

template <typename T>
class ReductionOpsTyped : public SushiBLASTest 
{
protected:
    std::vector<T> generate_random_data(size_t size) 
    {
        std::vector<T> data(size);
        std::mt19937 gen(42);
        if constexpr (std::is_integral_v<T>) 
        {
            std::uniform_int_distribution<T> dist(-10, 10);
            for (auto& v : data) v = dist(gen);
        } else 
        {
            std::uniform_real_distribution<T> dist(static_cast<T>(-10.0), static_cast<T>(10.0));
            for (auto& v : data) v = dist(gen);
        }
        return data;
    }
};

using TestTypes = ::testing::Types<float>;
TYPED_TEST_SUITE(ReductionOpsTyped, TestTypes);

TYPED_TEST(ReductionOpsTyped, Sum) 
{
    using T = TypeParam;
    auto data = this->generate_random_data(100);
    T expected = ReferenceKernels::sum(data);

    auto t = this->template create_initialized_tensor<T>({100}, data);
    auto res = this->engine->create_tensor({1}, get_sushiblas_dtype<T>());

    this->engine->reductions().sum(t, res);
    this->engine->execute().wait();

    EXPECT_TENSOR_NEAR(res, {expected}, static_cast<float>(1e-4));
}

TYPED_TEST(ReductionOpsTyped, Mean) 
{
    using T = TypeParam;
    auto data = this->generate_random_data(100);
    T expected = ReferenceKernels::mean(data);

    auto t = this->template create_initialized_tensor<T>({100}, data);
    auto res = this->engine->create_tensor({1}, get_sushiblas_dtype<T>());

    this->engine->reductions().mean(t, res);
    this->engine->execute().wait();

    EXPECT_TENSOR_NEAR(res, {expected}, static_cast<float>(1e-4));
}

TYPED_TEST(ReductionOpsTyped, Max) 
{
    using T = TypeParam;
    auto data = this->generate_random_data(100);
    data[25] = static_cast<T>(999.0);
    T expected = ReferenceKernels::max(data);

    auto t = this->template create_initialized_tensor<T>({100}, data);
    auto res = this->engine->create_tensor({1}, get_sushiblas_dtype<T>());

    this->engine->reductions().max(t, res);
    this->engine->execute().wait();

    EXPECT_TENSOR_NEAR(res, {expected}, static_cast<float>(1e-4));
}

TYPED_TEST(ReductionOpsTyped, Min) 
{
    using T = TypeParam;
    auto data = this->generate_random_data(100);
    data[10] = static_cast<T>(-999.0);
    T expected = ReferenceKernels::min(data);

    auto t = this->template create_initialized_tensor<T>({100}, data);
    auto res = this->engine->create_tensor({1}, get_sushiblas_dtype<T>());

    this->engine->reductions().min(t, res);
    this->engine->execute().wait();

    EXPECT_TENSOR_NEAR(res, {expected}, static_cast<float>(1e-4));
}

TYPED_TEST(ReductionOpsTyped, ArgMax) 
{
    using T = TypeParam;
    auto data = this->generate_random_data(100);
    data[42] = static_cast<T>(999.0);
    T expected = ReferenceKernels::argmax(data);

    auto t = this->template create_initialized_tensor<T>({100}, data);
    auto res = this->engine->create_tensor({1}, get_sushiblas_dtype<T>());

    this->engine->reductions().argmax(t, res);
    this->engine->execute().wait();

    EXPECT_TENSOR_NEAR(res, {expected}, static_cast<float>(1e-4));
}

TYPED_TEST(ReductionOpsTyped, ArgMin) 
{
    using T = TypeParam;
    auto data = this->generate_random_data(100);
    data[7] = static_cast<T>(-999.0);
    T expected = ReferenceKernels::argmin(data);

    auto t = this->template create_initialized_tensor<T>({100}, data);
    auto res = this->engine->create_tensor({1}, get_sushiblas_dtype<T>());

    this->engine->reductions().argmin(t, res);
    this->engine->execute().wait();

    EXPECT_TENSOR_NEAR(res, {expected}, static_cast<float>(1e-4));
}

TYPED_TEST(ReductionOpsTyped, Var) 
{
    using T = TypeParam;
    auto data = this->generate_random_data(5);
    T expected = ReferenceKernels::var(data);

    auto t = this->template create_initialized_tensor<T>({5}, data);
    auto res = this->engine->create_tensor({1}, get_sushiblas_dtype<T>());

    this->engine->reductions().var(t, res);
    this->engine->execute().wait();

    EXPECT_TENSOR_NEAR(res, {expected}, static_cast<float>(1e-4));
}

TYPED_TEST(ReductionOpsTyped, Std) 
{
    using T = TypeParam;
    auto data = this->generate_random_data(5);
    T expected = ReferenceKernels::std_dev(data);

    auto t = this->template create_initialized_tensor<T>({5}, data);
    auto res = this->engine->create_tensor({1}, get_sushiblas_dtype<T>());

    this->engine->reductions().std(t, res);
    this->engine->execute().wait();

    EXPECT_TENSOR_NEAR(res, {expected}, static_cast<float>(1e-4));
}

TYPED_TEST(ReductionOpsTyped, Prod) 
{
    using T = TypeParam;
    std::vector<T> data = {static_cast<T>(1), static_cast<T>(2), static_cast<T>(3), static_cast<T>(4), static_cast<T>(5)};
    T expected = ReferenceKernels::prod(data);

    auto t = this->template create_initialized_tensor<T>({5}, data);
    auto res = this->engine->create_tensor({1}, get_sushiblas_dtype<T>());

    this->engine->reductions().prod(t, res);
    this->engine->execute().wait();

    EXPECT_TENSOR_NEAR(res, {expected}, static_cast<float>(1e-4));
}

TYPED_TEST(ReductionOpsTyped, LogSumExp) 
{
    using T = TypeParam;
    std::vector<T> data = {static_cast<T>(1), static_cast<T>(2), static_cast<T>(3)};
    T expected = ReferenceKernels::logsumexp(data);

    auto t = this->template create_initialized_tensor<T>({3}, data);
    auto res = this->engine->create_tensor({1}, get_sushiblas_dtype<T>());

    this->engine->reductions().logsumexp(t, res);
    this->engine->execute().wait();

    EXPECT_TENSOR_NEAR(res, {expected}, static_cast<float>(1e-4));
}

TYPED_TEST(ReductionOpsTyped, Asum) 
{
    using T = TypeParam;
    auto data = this->generate_random_data(5);
    T expected = ReferenceKernels::asum(data);

    auto t = this->template create_initialized_tensor<T>({5}, data);
    auto res = this->engine->create_tensor({1}, get_sushiblas_dtype<T>());

    this->engine->reductions().asum(t, res);
    this->engine->execute().wait();

    EXPECT_TENSOR_NEAR(res, {expected}, static_cast<float>(1e-4));
}

TYPED_TEST(ReductionOpsTyped, Nrm2) 
{
    using T = TypeParam;
    auto data = this->generate_random_data(4);
    T expected = ReferenceKernels::nrm2(data);

    auto t = this->template create_initialized_tensor<T>({4}, data);
    auto res = this->engine->create_tensor({1}, get_sushiblas_dtype<T>());

    this->engine->reductions().nrm2(t, res);
    this->engine->execute().wait();

    EXPECT_TENSOR_NEAR(res, {expected}, static_cast<float>(1e-4));
}
