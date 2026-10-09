/**************************************************************************/
/* SushiBLAS.h                                                            */
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
 * @file SushiBLAS.h
 * @brief Includes the core, storage, tensor, engine and version headers and every operation
 * family of SushiBLAS.
 * @author Mustafa Garip
 */

#pragma once

#include <SushiBLAS/core/version.hpp>
#include <SushiBLAS/core/common.hpp>
#include <SushiBLAS/core/logger.hpp>
#include <SushiBLAS/tensor/storage.hpp>
#include <SushiBLAS/tensor/tensor.hpp>
#include <SushiBLAS/tensor/tensor_factory.hpp>
#include <SushiBLAS/sparse_storage/sparse_vector.hpp>
#include <SushiBLAS/sparse_storage/sparse_matrix.hpp>
#include <SushiBLAS/sparse_storage/sparse_builder.hpp>
#include <SushiBLAS/engine/engine.hpp>

#include <SushiBLAS/io/io.hpp>
#include <SushiBLAS/blas/blas.hpp>
#include <SushiBLAS/sparse/sparse.hpp>
#include <SushiBLAS/spatial/spatial.hpp>
#include <SushiBLAS/logic/logic.hpp>
#include <SushiBLAS/random/random.hpp>
#include <SushiBLAS/nonlinear/nonlinear.hpp>
#include <SushiBLAS/reductions/reductions.hpp>
#include <SushiBLAS/elementwise/elementwise.hpp>
