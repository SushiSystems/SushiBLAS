/****************************************************************/
/* test_spatial_im2col.cpp                                      */
/* SushiBLAS - https://github.com/SushiSystems/SushiBLAS        */
/* Copyright (c) 2026 Sushi Systems                             */
/* Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.    */
/* Commercial use requires a licence from Sushi Systems.        */
/****************************************************************/

#include <string>
#include <vector>
#include <cstddef>
#include <cstdint>
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
            /**
             * @brief The bit pattern of a float.
             *
             * `std::bit_cast` is C++20 and this tree is C++17; `std::memcpy`
             * between two trivially-copyable objects of equal size is the same
             * operation with the same guarantees. The comparison is on bits
             * because `==` on floats calls two NaNs different and +0 and -0
             * equal, and neither is what "the same answer" means here.
             *
             * @param value The float to reinterpret.
             * @return Its IEEE-754 bit pattern.
             */
            uint32_t float_bits(float value) noexcept
            {
                uint32_t bits = 0;
                std::memcpy(&bits, &value, sizeof(bits));
                return bits;
            }
        } // namespace

        class Unit_SpatialIm2ColTest : public SushiBLASTest
        {
            protected:
                /** @brief An NHWC tensor holding 0, 1, 2, ... in memory order. */
                Tensor ramp(int64_t n, int64_t h, int64_t w, int64_t c)
                {
                    auto t = engine->create_tensor({n, h, w, c}, Core::DataType::FLOAT32);
                    float* p = t.data_as<float>();
                    for (int64_t i = 0; i < t.num_elements; ++i)
                        p[i] = static_cast<float>(i);
                    return t;
                }
        };

        TEST_F(Unit_SpatialIm2ColTest, AUnitWindowIsTheIdentityGather)
        {
            // 1x1, stride 1, no padding: the column matrix is the input reshaped,
            // element for element and in the same order. If this fails, the row
            // and element index arithmetic disagree with each other and nothing
            // further is worth reading.
            auto in = ramp(2, 3, 4, 5);
            auto cols = engine->create_tensor({2 * 3 * 4, 5}, Core::DataType::FLOAT32);

            Spatial::ConvGeometry g;   // 1x1, stride 1, dilation 1, no pad
            engine->spatial().im2col(in, cols, g);
            engine->execute();

            const float* out = cols.data_as<float>();
            for (int64_t i = 0; i < in.num_elements; ++i)
                EXPECT_FLOAT_EQ(out[i], static_cast<float>(i)) << "element " << i;
        }

        TEST_F(Unit_SpatialIm2ColTest, EachRowIsOneWindowWithChannelInnermost)
        {
            // 2x2 stride 1 over a 1x3x3x2 input: 4 windows, 8 entries each.
            // Written out by hand rather than derived, so the test is a
            // specification and not a second implementation.
            auto in = ramp(1, 3, 3, 2);
            auto cols = engine->create_tensor({4, 8}, Core::DataType::FLOAT32);

            Spatial::ConvGeometry g;
            g.kernel_h = 2;
            g.kernel_w = 2;

            engine->spatial().im2col(in, cols, g);
            engine->execute();

            // Input element (h, w, c) holds (h*3 + w)*2 + c.
            // Row 0 is the window at (0,0): taps (0,0) (0,1) (1,0) (1,1).
            const std::vector<float> row0 = {0, 1, 2, 3, 6, 7, 8, 9};
            // Row 1 is the window at (0,1): taps (0,1) (0,2) (1,1) (1,2).
            const std::vector<float> row1 = {2, 3, 4, 5, 8, 9, 10, 11};
            // Row 3 is the window at (1,1): taps (1,1) (1,2) (2,1) (2,2).
            const std::vector<float> row3 = {8, 9, 10, 11, 14, 15, 16, 17};

            const float* out = cols.data_as<float>();
            for (std::size_t i = 0; i < row0.size(); ++i)
            {
                EXPECT_FLOAT_EQ(out[0 * 8 + i], row0[i]) << "row 0 entry " << i;
                EXPECT_FLOAT_EQ(out[1 * 8 + i], row1[i]) << "row 1 entry " << i;
                EXPECT_FLOAT_EQ(out[3 * 8 + i], row3[i]) << "row 3 entry " << i;
            }
        }

        TEST_F(Unit_SpatialIm2ColTest, OutOfBoundsTapsReadZeroRatherThanNeighbouringMemory)
        {
            // Padding is a parameter, not a separate zero-padded copy - which is
            // what saves a full-tensor copy forward and a crop backward. The
            // property that makes that safe is exactly this one.
            auto in = ramp(1, 2, 2, 1);
            auto cols = engine->create_tensor({4, 9}, Core::DataType::FLOAT32);

            Spatial::ConvGeometry g;
            g.kernel_h = 3;
            g.kernel_w = 3;
            g.pad_top = 1;
            g.pad_left = 1;

            engine->spatial().im2col(in, cols, g);
            engine->execute();

            // Row 0 is the window centred on pixel (0,0); its first tap sits at
            // (-1,-1) and its last at (1,1). Only the four in-bounds taps carry
            // data: positions 4, 5, 7, 8 of the 3x3.
            const std::vector<float> row0 = {0, 0, 0, 0, 0, 1, 0, 2, 3};

            const float* out = cols.data_as<float>();
            for (std::size_t i = 0; i < row0.size(); ++i)
                EXPECT_FLOAT_EQ(out[i], row0[i]) << "entry " << i;
        }

        TEST_F(Unit_SpatialIm2ColTest, DilationSpacesTheTapsWithoutMovingTheWindows)
        {
            auto in = ramp(1, 5, 5, 1);
            auto cols = engine->create_tensor({1, 9}, Core::DataType::FLOAT32);

            Spatial::ConvGeometry g;
            g.kernel_h = 3;
            g.kernel_w = 3;
            g.dilation_h = 2;
            g.dilation_w = 2;

            engine->spatial().im2col(in, cols, g);
            engine->execute();

            // The single window's taps are rows {0,2,4} x columns {0,2,4}.
            const std::vector<float> expected = {0, 2, 4, 10, 12, 14, 20, 22, 24};

            const float* out = cols.data_as<float>();
            for (std::size_t i = 0; i < expected.size(); ++i)
                EXPECT_FLOAT_EQ(out[i], expected[i]) << "entry " << i;
        }

        TEST_F(Unit_SpatialIm2ColTest, AsymmetricPaddingNeedsTheExplicitExtentOverload)
        {
            // 2x2 stride 1 over a 3x3 image padded on the bottom and right only:
            // 3x3 windows out. The two-operand overload cannot express this - it
            // has no pad_right to read - and the point of this test is that it
            // refuses rather than silently producing a differently-shaped answer.
            auto in = ramp(1, 3, 3, 1);
            auto cols = engine->create_tensor({9, 4}, Core::DataType::FLOAT32);

            Spatial::ConvGeometry g;
            g.kernel_h = 2;
            g.kernel_w = 2;

            // Symmetric-padding derivation gives W_o = 2, which does not divide 9.
            EXPECT_THROW(engine->spatial().im2col(in, cols, g), std::exception);

            engine->spatial().im2col(in, cols, g, 3, 3);
            engine->execute();

            const float* out = cols.data_as<float>();

            // Window (0,0): taps (0,0) (0,1) (1,0) (1,1).
            const std::vector<float> row0 = {0, 1, 3, 4};
            // Window (2,2): only its first tap is in bounds; the rest fall past
            // the bottom and right edges and read the implicit zero.
            const std::vector<float> row8 = {8, 0, 0, 0};

            for (std::size_t i = 0; i < row0.size(); ++i)
            {
                EXPECT_FLOAT_EQ(out[0 * 4 + i], row0[i]) << "row 0 entry " << i;
                EXPECT_FLOAT_EQ(out[8 * 4 + i], row8[i]) << "row 8 entry " << i;
            }
        }

        TEST_F(Unit_SpatialIm2ColTest, TheSyclKernelAgreesWithTheReferenceKernel)
        {
            // The comparison is done here, explicitly, rather than by running
            // under ExecutionMode::VERIFY. VERIFY submits the device kernel and
            // then runs the reference one over the *same* output buffer, so the
            // reference answer is what survives and a disagreement leaves no
            // trace - see Internal::dispatch_with_mode. Whatever the enum's
            // comment says, it is not an assertion, so this test makes its own:
            // fill two buffers under the two forced modes and compare them.
            //
            // Bit-exact, not near: im2col only ever copies an element or writes
            // a zero, so the two kernels must produce the same bits or one of
            // them is addressing memory differently from the other.
            auto in = ramp(2, 7, 5, 3);

            Spatial::ConvGeometry g;
            g.kernel_h = 3;
            g.kernel_w = 2;
            g.stride_h = 2;
            g.stride_w = 1;
            g.dilation_h = 2;
            g.pad_top = 1;
            g.pad_left = 1;

            const int64_t out_h = Spatial::output_extent(7, 1, 1, 2, 3, 2);
            const int64_t out_w = Spatial::output_extent(5, 1, 1, 1, 2, 1);
            const int64_t rows = 2 * out_h * out_w;
            const int64_t width = 3 * 2 * 3;

            auto from_reference = engine->create_tensor({rows, width}, Core::DataType::FLOAT32);
            auto from_device = engine->create_tensor({rows, width}, Core::DataType::FLOAT32);

            engine->set_mode(ExecutionMode::FORCE_REFERENCE);
            engine->spatial().im2col(in, from_reference, g, out_h, out_w);
            engine->execute();

            engine->set_mode(ExecutionMode::FORCE_SYCL);
            engine->spatial().im2col(in, from_device, g, out_h, out_w);
            engine->execute();

            const float* expected = from_reference.data_as<float>();
            const float* got = from_device.data_as<float>();
            for (int64_t i = 0; i < rows * width; ++i)
                EXPECT_EQ(float_bits(got[i]), float_bits(expected[i]))
                    << "entry " << i << ": reference " << expected[i] << ", device " << got[i];
        }

        TEST_F(Unit_SpatialIm2ColTest, AColumnMatrixOfTheWrongShapeIsRefusedByName)
        {
            auto in = ramp(1, 4, 4, 2);
            auto cols = engine->create_tensor({9, 7}, Core::DataType::FLOAT32);   // 7 != 2*2*2

            Spatial::ConvGeometry g;
            g.kernel_h = 2;
            g.kernel_w = 2;

            try
            {
                engine->spatial().im2col(in, cols, g, 3, 3);
                FAIL() << "a column matrix whose width is not kernel_h*kernel_w*C must be refused";
            }
            catch (const std::exception& e)
            {
                const std::string message = e.what();
                EXPECT_NE(message.find("spatial.im2col"), std::string::npos) << message;
                EXPECT_NE(message.find("7"), std::string::npos) << message;
                EXPECT_NE(message.find("8"), std::string::npos) << message;
            }
        }

        TEST_F(Unit_SpatialIm2ColTest, ARankThatIsNotFourIsRefusedAtRecordTime)
        {
            auto in = engine->create_tensor({4, 4, 2}, Core::DataType::FLOAT32);
            auto cols = engine->create_tensor({16, 8}, Core::DataType::FLOAT32);

            EXPECT_THROW(engine->spatial().im2col(in, cols, Spatial::ConvGeometry{}),
                         std::exception);
        }
    }
}
