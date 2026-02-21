#pragma once

#include <memory>
#include <gtest/gtest.h>
#include <SushiBLAS/SushiBLAS.h>

/**
 * @brief Universal test fixture for SushiBLAS tests.
 * 
 * Provides a RuntimeContext and a managed Engine instance.
 */
class SushiBLASTest : public ::testing::Test 
{
    protected:
        inline static sr::Execution::RuntimeContext ctx;
        std::unique_ptr<sb::Engine> engine;

        void SetUp() override 
        {
            engine = std::make_unique<sb::Engine>(ctx, sb::Core::Layout::ROW_MAJOR);
        }

        void TearDown() override 
        {
            engine.reset();
        }

        // Helper to re-initialize engine with a different layout if needed
        void reinit_engine(sb::Core::Layout layout) 
        {
            engine = std::make_unique<sb::Engine>(ctx, layout);
        }

        /** @brief Fills a tensor with data from a vector. */
        template<typename T = float>
        void fill_tensor(sb::Tensor& t, const std::vector<T>& data) 
        {
            SB_THROW_IF(static_cast<int64_t>(data.size()) != t.num_elements, 
                        "Data size mismatch. Vector: {}, Tensor: {}", data.size(), t.num_elements);
            
            T* ptr = t.data_as<T>();
            std::copy(data.begin(), data.end(), ptr);
        }

        /** @brief Verifies tensor data against expected values. */
        template<typename T = float>
        void verify_tensor(const sb::Tensor& t, const std::vector<T>& expected) 
        {
            SB_THROW_IF(static_cast<int64_t>(expected.size()) != t.num_elements, 
                        "Data size mismatch. Expected: {}, Actual: {}", expected.size(), t.num_elements);
            
            const T* ptr = t.data_as<T>();
            for(size_t i = 0; i < expected.size(); ++i) 
            {
                EXPECT_FLOAT_EQ(ptr[i], expected[i]) << "Mismatch at index " << i;
            }
        }
};
