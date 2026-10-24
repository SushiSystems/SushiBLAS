/**************************************************************************/
/* SushiBLAS.hpp                                                          */
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
 * @file SushiBLAS.hpp
 * @brief Includes every public header of SushiBLAS.
 * @author Mustafa Garip
 */

#pragma once

#include <SushiBLAS/arithmetic/accumulator.hpp>
#include <SushiBLAS/arithmetic/complex_arithmetic.hpp>
#include <SushiBLAS/arithmetic/device_math.hpp>
#include <SushiBLAS/arithmetic/safe_math.hpp>
#include <SushiBLAS/core/blas_types.hpp>
#include <SushiBLAS/core/common.hpp>
#include <SushiBLAS/core/errors.hpp>
#include <SushiBLAS/core/export.hpp>
#include <SushiBLAS/core/logger.hpp>
#include <SushiBLAS/core/sparse_types.hpp>
#include <SushiBLAS/core/traits.hpp>
#include <SushiBLAS/core/version.hpp>
#include <SushiBLAS/geometry/interpolation.hpp>
#include <SushiBLAS/geometry/matrix4.hpp>
#include <SushiBLAS/geometry/quaternion.hpp>
#include <SushiBLAS/geometry/spatial_geometry.hpp>
#include <SushiBLAS/geometry/transform.hpp>
#include <SushiBLAS/geometry/vector3.hpp>
#include <SushiBLAS/graph_primitives/combiners.hpp>
#include <SushiBLAS/graph_primitives/counting_sort.hpp>
#include <SushiBLAS/graph_primitives/fixed_order_fold.hpp>
#include <SushiBLAS/graph_primitives/fixed_order_reduce.hpp>
#include <SushiBLAS/graph_primitives/fixed_order_scan.hpp>
#include <SushiBLAS/graph_primitives/invalid_graph.hpp>
#include <SushiBLAS/graph_primitives/prefix_scan.hpp>
#include <SushiBLAS/graph_primitives/radix_plan.hpp>
#include <SushiBLAS/graph_primitives/reduce.hpp>
#include <SushiBLAS/graph_primitives/segmented_reduce.hpp>
#include <SushiBLAS/launch/device_policy.hpp>
#include <SushiBLAS/launch/launch_1d.hpp>
#include <SushiBLAS/orthonormal/orthonormal_sweep.hpp>
#include <SushiBLAS/reduction/reduce_1d.hpp>
#include <SushiBLAS/reduction/reduce_axis.hpp>
#include <SushiBLAS/reference/reference_accumulator.hpp>
#include <SushiBLAS/reference/serial_1d.hpp>
#include <SushiBLAS/reference/serial_reduce.hpp>
#include <SushiBLAS/sparse_assembly/sparse_assembly.hpp>
#include <SushiBLAS/substitution/running_solution.hpp>
#include <SushiBLAS/sparse_storage/sparse_builder.hpp>
#include <SushiBLAS/sparse_storage/sparse_matrix.hpp>
#include <SushiBLAS/sparse_storage/sparse_vector.hpp>
#include <SushiBLAS/tensor/host_data_type.hpp>
#include <SushiBLAS/tensor/storage.hpp>
#include <SushiBLAS/tensor/tensor.hpp>
#include <SushiBLAS/tensor/tensor_factory.hpp>
#include <SushiBLAS/blas_seam_kernels/band_access.hpp>
#include <SushiBLAS/blas_seam_kernels/conjugation.hpp>
#include <SushiBLAS/blas_seam_kernels/givens.hpp>
#include <SushiBLAS/blas_seam_kernels/level3/channel_bias_epilogue.hpp>
#include <SushiBLAS/blas_seam_kernels/level3/gemm_epilogue.hpp>
#include <SushiBLAS/blas_seam_kernels/level3/packed_product_plan.hpp>
#include <SushiBLAS/blas_seam_kernels/modified_givens.hpp>
#include <SushiBLAS/blas_seam_kernels/operand_access.hpp>
#include <SushiBLAS/blas_seam_kernels/packed_access.hpp>
#include <SushiBLAS/blas_seam_kernels/reference/level3/gemm.hpp>
#include <SushiBLAS/blas_seam_kernels/row_cursor.hpp>
#include <SushiBLAS/blas_seam_kernels/sycl/level3/gemm.hpp>
#include <SushiBLAS/blas_seam_kernels/sycl/level3/gemm_accessed_packed.hpp>
#include <SushiBLAS/blas_seam_kernels/sycl/level3/gemm_packed.hpp>
#include <SushiBLAS/blas_seam_kernels/sycl/level3/packed_micro_kernel.hpp>
#include <SushiBLAS/blas_seam_kernels/sycl/level3/packed_panel.hpp>
#include <SushiBLAS/blas_seam_kernels/sycl/level3/packed_product.hpp>
#include <SushiBLAS/blas_seam_kernels/symmetric_access.hpp>
#include <SushiBLAS/blas_seam_kernels/symmetric_row.hpp>
#include <SushiBLAS/engine/dispatch.hpp>
#include <SushiBLAS/engine/engine.hpp>
#include <SushiBLAS/engine/execution_mode.hpp>
#include <SushiBLAS/engine/kernel_pair_dispatcher.hpp>
#include <SushiBLAS/engine/operand_check.hpp>
#include <SushiBLAS/engine/operand_set.hpp>
#include <SushiBLAS/engine/operation_options.hpp>
#include <SushiBLAS/engine/rng_stream.hpp>
#include <SushiBLAS/engine/scratch.hpp>
#include <SushiBLAS/engine/scratch_pool.hpp>
#include <SushiBLAS/engine/scratch_shelf.hpp>
#include <SushiBLAS/engine/task.hpp>
#include <SushiBLAS/engine/task_description.hpp>
#include <SushiBLAS/engine/task_handle.hpp>
#include <SushiBLAS/engine/task_recorder.hpp>
#include <SushiBLAS/functor_map/map_task.hpp>
#include <SushiBLAS/functor_map/read_binding.hpp>
#include <SushiBLAS/functor_map/scalar.hpp>
#include <SushiBLAS/functor_map/scalar_binding.hpp>
#include <SushiBLAS/functor_map/write_binding.hpp>
#include <SushiBLAS/functor_map/index_task.hpp>
#include <SushiBLAS/operands/argument_check.hpp>
#include <SushiBLAS/operands/blas_metadata.hpp>
#include <SushiBLAS/operands/matrix_operand.hpp>
#include <SushiBLAS/operands/scalar.hpp>
#include <SushiBLAS/operands/scalar_result.hpp>
#include <SushiBLAS/operands/scratch_lease.hpp>
#include <SushiBLAS/operands/vector_operand.hpp>
#include <SushiBLAS/blas/band_operand.hpp>
#include <SushiBLAS/blas/blas.hpp>
#include <SushiBLAS/blas/blas_task.hpp>
#include <SushiBLAS/blas/dispatch.hpp>
#include <SushiBLAS/blas/level1.hpp>
#include <SushiBLAS/blas/level2.hpp>
#include <SushiBLAS/blas/level3.hpp>
#include <SushiBLAS/blas/local_tile.hpp>
#include <SushiBLAS/blas/packed_operand.hpp>
#include <SushiBLAS/blas/packed_product_scratch.hpp>
#include <SushiBLAS/blas/scalar_operand.hpp>
#include <SushiBLAS/blas/sparse.hpp>
#include <SushiBLAS/elementwise/dispatch.hpp>
#include <SushiBLAS/elementwise/elementwise.hpp>
#include <SushiBLAS/elementwise/wrapping_math.hpp>
#include <SushiBLAS/experimental/lapack/linalg.hpp>
#include <SushiBLAS/experimental/signal/transforms.hpp>
#include <SushiBLAS/io/io.hpp>
#include <SushiBLAS/io/io_status.hpp>
#include <SushiBLAS/logic/dispatch.hpp>
#include <SushiBLAS/logic/logic.hpp>
#include <SushiBLAS/nonlinear/dispatch.hpp>
#include <SushiBLAS/nonlinear/nonlinear.hpp>
#include <SushiBLAS/random/dispatch.hpp>
#include <SushiBLAS/random/random.hpp>
#include <SushiBLAS/reductions/arg_reduction.hpp>
#include <SushiBLAS/reductions/centred_reduction.hpp>
#include <SushiBLAS/reductions/dispatch.hpp>
#include <SushiBLAS/reductions/reductions.hpp>
#include <SushiBLAS/sparse/accumulator_size.hpp>
#include <SushiBLAS/sparse/index_check.hpp>
#include <SushiBLAS/sparse/sparse.hpp>
#include <SushiBLAS/sparse/sparse_matrix_operand.hpp>
#include <SushiBLAS/sparse/sparse_vector_operand.hpp>
#include <SushiBLAS/spatial/spatial.hpp>
#include <SushiBLAS/spatial/window.hpp>
#include <SushiBLAS/signal/signal.hpp>
#include <SushiBLAS/blend/blend.hpp>
#include <SushiBLAS/index/index_ops.hpp>
