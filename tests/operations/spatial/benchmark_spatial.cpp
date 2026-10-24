/**************************************************************************/
/* benchmark_spatial.cpp                                                  */
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
 * @file benchmark_spatial.cpp
 * @brief Times the spatial operations on NHWC images and prints GB/s per image side.
 * @author Mustafa Garip
 */

#include <array>
#include <cstdint>
#include <cstdio>
#include <vector>
#include <SushiBLAS/SushiBLAS.hpp>

#include <SushiBLAS/benchmark_harness/benchmark_harness.hpp>

namespace
{
    namespace Bench = SushiBLAS::Bench;
    using SushiBLAS::Core::DataType;
    using SushiBLAS::Spatial::ConvGeometry;

    /// @brief Image sides, H = W, timed when no size is given.
    const std::vector<int64_t> K_DEFAULT_SIZES = {32, 64, 128};

    /// @brief Data types timed for every side.
    const std::vector<DataType> K_DTYPES = {DataType::Float32, DataType::Float64};

    /// @brief Images in every batch.
    constexpr int64_t K_BATCH = 8;

    /// @brief Channels of every image.
    constexpr int64_t K_CHANNELS = 64;

    /// @brief Side of the convolution window, padded by one so the output keeps the input's side.
    constexpr int64_t K_WINDOW = 3;

    /// @brief Side and stride of the pooling window, and the factor of the upsampling.
    constexpr int64_t K_FACTOR = 2;

    /// @brief Bytes of one INT64 argmax index.
    constexpr double K_INDEX_BYTES = 8.0;

    /// @brief Value written into every element of every input.
    constexpr double K_VALUE = 0.5;

    /// @brief Returns a 3x3 window with a stride of one and one zero on every side.
    [[nodiscard]] ConvGeometry same_window() noexcept
    {
        ConvGeometry g;
        g.kernel_h = K_WINDOW;
        g.kernel_w = K_WINDOW;
        g.pad_top = 1;
        g.pad_left = 1;
        return g;
    }

    /// @brief Returns a 2x2 window with a stride of two and no padding.
    [[nodiscard]] ConvGeometry pooling_window() noexcept
    {
        ConvGeometry g;
        g.kernel_h = K_FACTOR;
        g.kernel_w = K_FACTOR;
        g.stride_h = K_FACTOR;
        g.stride_w = K_FACTOR;
        return g;
    }

    /// @brief Returns an [N, H, W, C] tensor filled with K_VALUE.
    [[nodiscard]] SushiBLAS::Tensor image(SushiBLAS::Engine& engine, DataType dtype, int64_t h,
                                          int64_t w, int64_t c)
    {
        SushiBLAS::Tensor t = engine.create_tensor({K_BATCH, h, w, c}, dtype);
        Bench::fill_tensor(t, K_VALUE);
        return t;
    }

    /// @brief Returns a view of base that stores its [N, H, W, C] elements in NCHW order.
    [[nodiscard]] SushiBLAS::Tensor nchw_view(const SushiBLAS::Tensor& base, int64_t h, int64_t w,
                                              int64_t c)
    {
        std::array<int64_t, SushiBLAS::Core::K_MAX_TENSOR_RANK> strides{};
        strides[0] = c * h * w;
        strides[1] = w;
        strides[2] = 1;
        strides[3] = h * w;

        const int64_t dims[4] = {K_BATCH, h, w, c};
        return SushiBLAS::Tensor(base.storage, SushiRuntime::span<const int64_t>(dims, dims + 4),
                                 strides, base.dtype, base.storage_offset, base.layout);
    }

    /// @brief Times the window gathers, the window products and the poolings at one side.
    void benchmark_windows(SushiBLAS::Engine& engine, const Bench::ResultTable& table,
                           DataType dtype, int64_t side)
    {
        SushiBLAS::RecordingGraph graph(engine);
        SushiBLAS::TaskRecorder& recorder = graph.recorder();
        SushiBLAS::SpatialOps spatial(recorder);
        const int64_t c = K_CHANNELS;
        const int64_t half = side / K_FACTOR;
        const double element = static_cast<double>(Bench::dtype_bytes(dtype));
        const double e = static_cast<double>(K_BATCH * side * side * c) * element;
        const double pooled = e / static_cast<double>(K_FACTOR * K_FACTOR);
        const double indices = pooled / element * K_INDEX_BYTES;
        const double taps = static_cast<double>(K_WINDOW * K_WINDOW);
        const double weight_bytes = taps * static_cast<double>(c) * element;
        const double gb = Bench::K_BYTES_PER_GIGABYTE;

        const ConvGeometry same = same_window();
        const ConvGeometry pool = pooling_window();
        SushiBLAS::Tensor in = image(engine, dtype, side, side, c);
        SushiBLAS::Tensor dy = image(engine, dtype, side, side, c);
        SushiBLAS::Tensor out = image(engine, dtype, side, side, c);
        SushiBLAS::Tensor cols =
            engine.create_tensor({K_BATCH * side * side, K_WINDOW * K_WINDOW * c}, dtype);
        SushiBLAS::Tensor weights = engine.create_tensor({K_WINDOW, K_WINDOW, c}, dtype);
        SushiBLAS::Tensor dweights = engine.create_tensor({K_WINDOW, K_WINDOW, c}, dtype);
        SushiBLAS::Tensor small = image(engine, dtype, half, half, c);
        SushiBLAS::Tensor argmax = engine.create_tensor({K_BATCH, half, half, c}, DataType::Int64);
        Bench::fill_tensor(cols, K_VALUE);
        Bench::fill_tensor(weights, K_VALUE);

        // A 3x3 window reads nine elements for each element it writes into cols.
        table.print_row("im2col", dtype, side,
                        Bench::measure(graph, [&]() { spatial.im2col(in, cols, same); }),
                        e + taps * e, gb);
        table.print_row("col2im", dtype, side,
                        Bench::measure(graph, [&]()
                                       { spatial.col2im(cols, out, same, side, side); }),
                        taps * e + e, gb);
        table.print_row("wdot", dtype, side,
                        Bench::measure(graph, [&]()
                                       { spatial.window_dot(in, weights, out, same); }),
                        2.0 * e + weight_bytes, gb);
        table.print_row("wdot dx", dtype, side,
                        Bench::measure(graph, [&]()
                                       { spatial.window_dot_backward(dy, weights, out, same); }),
                        2.0 * e + weight_bytes, gb);
        table.print_row("wdot dw", dtype, side,
                        Bench::measure(graph, [&]()
                                       {
                                           spatial.window_dot_weights_backward(dy, in, dweights,
                                                                               same);
                                       }),
                        2.0 * e + weight_bytes, gb);
        table.print_row("pool max", dtype, side,
                        Bench::measure(graph, [&]()
                                       { spatial.pool_max(in, small, argmax, pool); }),
                        e + pooled + indices, gb);
        table.print_row("pmax dx", dtype, side,
                        Bench::measure(graph, [&]()
                                       { spatial.pool_max_backward(small, argmax, out, pool); }),
                        pooled + indices + e, gb);
        table.print_row("pool avg", dtype, side,
                        Bench::measure(graph, [&]() { spatial.pool_avg(in, small, pool); }),
                        e + pooled, gb);
        table.print_row("pavg dx", dtype, side,
                        Bench::measure(graph, [&]()
                                       { spatial.pool_avg_backward(small, out, pool); }),
                        pooled + e, gb);
    }

    /// @brief Times the upsampling, the channel copies, the affine, the broadcast and the layout.
    void benchmark_copies(SushiBLAS::Engine& engine, const Bench::ResultTable& table,
                          DataType dtype, int64_t side)
    {
        SushiBLAS::RecordingGraph graph(engine);
        SushiBLAS::TaskRecorder& recorder = graph.recorder();
        SushiBLAS::SpatialOps spatial(recorder);
        const int64_t c = K_CHANNELS;
        const int64_t big = side * K_FACTOR;
        const double element = static_cast<double>(Bench::dtype_bytes(dtype));
        const double e = static_cast<double>(K_BATCH * side * side * c) * element;
        const double upsampled = e * static_cast<double>(K_FACTOR * K_FACTOR);
        const double channel_bytes = static_cast<double>(c) * element;
        const double plane_bytes = static_cast<double>(K_BATCH) * channel_bytes;
        const double gb = Bench::K_BYTES_PER_GIGABYTE;

        SushiBLAS::Tensor in = image(engine, dtype, side, side, c);
        SushiBLAS::Tensor out = image(engine, dtype, side, side, c);
        SushiBLAS::Tensor wide = image(engine, dtype, side, side, 2 * c);
        SushiBLAS::Tensor large = image(engine, dtype, big, big, c);
        SushiBLAS::Tensor gamma = engine.create_tensor({c}, dtype);
        SushiBLAS::Tensor beta = engine.create_tensor({c}, dtype);
        SushiBLAS::Tensor plane = engine.create_tensor({K_BATCH, c}, dtype);
        SushiBLAS::Tensor nchw_storage = image(engine, dtype, side, side, c);
        SushiBLAS::Tensor nchw = nchw_view(nchw_storage, side, side, c);
        Bench::fill_tensor(gamma, K_VALUE);
        Bench::fill_tensor(beta, K_VALUE);
        Bench::fill_tensor(plane, K_VALUE);

        table.print_row("upsample", dtype, side,
                        Bench::measure(graph, [&]()
                                       {
                                           spatial.upsample_nearest(in, large, K_FACTOR,
                                                                    K_FACTOR);
                                       }),
                        e + upsampled, gb);
        table.print_row("ups dx", dtype, side,
                        Bench::measure(graph, [&]()
                                       {
                                           spatial.upsample_nearest_backward(large, out, K_FACTOR,
                                                                             K_FACTOR);
                                       }),
                        upsampled + e, gb);
        table.print_row("concat", dtype, side,
                        Bench::measure(graph, [&]() { spatial.concat_channels(in, wide, c); }),
                        2.0 * e, gb);
        table.print_row("slice", dtype, side,
                        Bench::measure(graph, [&]() { spatial.slice_channels(wide, c, out); }),
                        2.0 * e, gb);
        table.print_row("affine", dtype, side,
                        Bench::measure(graph, [&]()
                                       { spatial.channel_affine(in, gamma, beta, out); }),
                        2.0 * e + 2.0 * channel_bytes, gb);
        table.print_row("broadcast", dtype, side,
                        Bench::measure(graph, [&]()
                                       { spatial.broadcast_spatial(plane, out, K_VALUE); }),
                        plane_bytes + e, gb);
        table.print_row("to nchw", dtype, side,
                        Bench::measure(graph, [&]() { spatial.layout_convert(in, nchw); }),
                        2.0 * e, gb);
        table.print_row("to nhwc", dtype, side,
                        Bench::measure(graph, [&]() { spatial.layout_convert(nchw, out); }),
                        2.0 * e, gb);
    }

    /// @brief Times every data type at every requested side.
    void run(int argc, char** argv)
    {
        const std::vector<int64_t> sizes = Bench::parse_sizes(argc, argv, K_DEFAULT_SIZES);

        SushiRuntime::Runtime runtime = SushiRuntime::Runtime::create().value();
        SushiBLAS::Engine engine(runtime);
        Bench::print_environment(engine);
        std::printf("batch:         %lld images of %lld channels\n\n",
                    static_cast<long long>(K_BATCH), static_cast<long long>(K_CHANNELS));

        const Bench::ResultTable table("GB/s");
        for (const DataType dtype : K_DTYPES)
        {
            for (const int64_t side : sizes)
            {
                benchmark_windows(engine, table, dtype, side);
                benchmark_copies(engine, table, dtype, side);
            }
        }
    }
} // namespace

/// @brief Runs the spatial benchmark; sizes are optional image sides.
int main(int argc, char** argv)
{
    return Bench::run_guarded(run, argc, argv);
}
