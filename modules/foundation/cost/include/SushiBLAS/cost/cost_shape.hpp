/**************************************************************************/
/* cost_shape.hpp                                                         */
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
 * @file cost_shape.hpp
 * @brief Declares the shape facts an operation states for its cost, one struct per family.
 * @author Mustafa Garip
 * @see modules/foundation/cost/README.md
 */

#pragma once

#include <cstdint>
#include <variant>
#include <SushiBLAS/core/common.hpp>

namespace SushiBLAS
{
    /** @brief Holds the extents of a matrix product `C = op(A) * op(B)`, batched. */
    struct GemmShape
    {
        std::int64_t m = 0;                              /**< Rows of C. */
        std::int64_t n = 0;                              /**< Columns of C. */
        std::int64_t k = 0;                              /**< The summed extent. */
        std::int64_t batch = 1;                          /**< Independent products. */
        Core::DataType dtype = Core::DataType::Float32;  /**< Element type of the operands. */
        std::uint32_t epilogue_flops_per_element = 0;    /**< Work the fused epilogue does on each element of C. */
    };

    /** @brief Holds the extents of a convolution; a depthwise one has groups equal to both channel counts. */
    struct ConvolutionShape
    {
        std::int64_t batch = 0;                          /**< Images. */
        std::int64_t out_h = 0;                          /**< Output rows. */
        std::int64_t out_w = 0;                          /**< Output columns. */
        std::int64_t out_channels = 0;                   /**< Output channels. */
        std::int64_t in_channels = 0;                    /**< Input channels. */
        std::int64_t groups = 1;                         /**< Channel groups; divides both channel counts. */
        std::int64_t kernel_h = 1;                       /**< Window rows. */
        std::int64_t kernel_w = 1;                       /**< Window columns. */
        Core::DataType dtype = Core::DataType::Float32;  /**< Element type of the operands. */
        std::uint32_t epilogue_flops_per_output = 0;     /**< Work the fused bias or activation does on each output. */
    };

    /** @brief Holds the element count of a task that applies one formula to each element. */
    struct ElementwiseShape
    {
        std::int64_t count = 0;                          /**< Elements the task covers. */
        Core::DataType dtype = Core::DataType::Float32;  /**< Element type of the operands. */
    };

    /** @brief Holds the extents of a task that folds inputs into outputs. */
    struct ReductionShape
    {
        std::int64_t inputs = 0;                         /**< Elements read. */
        std::int64_t outputs = 1;                        /**< Results written. */
        Core::DataType dtype = Core::DataType::Float32;  /**< Element type of the operands. */
    };

    /** @brief Holds the extents of a task that folds a window of taps into each output. */
    struct WindowShape
    {
        std::int64_t outputs = 0;                        /**< Output elements. */
        std::int64_t taps = 0;                           /**< Window elements folded into each. */
        Core::DataType dtype = Core::DataType::Float32;  /**< Element type of the operands. */
    };

    /** @brief Holds the extents of a batched Fourier transform over rows and columns. */
    struct TransformShape
    {
        std::int64_t batch = 0;                          /**< Independent transforms. */
        std::int64_t rows = 1;                           /**< Rows of one transform; 1 for a 1-D one. */
        std::int64_t columns = 0;                        /**< Columns of one transform. */
        bool real = false;                               /**< True for a real-to-complex transform. */
        Core::DataType dtype = Core::DataType::Float32;  /**< The scalar the transform computes in. */
    };

    /** @brief Holds the element type of a task that only moves data. */
    struct CopyShape
    {
        Core::DataType dtype = Core::DataType::Float32;  /**< Element type of the operands. */
    };

    /**
     * @brief Holds the shape facts one call site states; the empty alternative states none.
     *
     * A family is one struct above and one rule in cost_table.cpp. cost_of visits the variant
     * once, so a new family adds a struct and its rule.
     * @see modules/foundation/cost/README.md
     */
    using CostShape = std::variant<std::monostate, GemmShape, ConvolutionShape, ElementwiseShape,
                                   ReductionShape, WindowShape, TransformShape, CopyShape>;
} // namespace SushiBLAS
