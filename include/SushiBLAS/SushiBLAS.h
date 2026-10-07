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

#include <SushiBLAS/version.hpp>
#include <SushiBLAS/core/common.hpp>
#include <SushiBLAS/core/logger.hpp>
#include <SushiBLAS/storage.hpp>
#include <SushiBLAS/tensor.hpp>
#include <SushiBLAS/tensor_factory.hpp>
#include <SushiBLAS/sparse_vector.hpp>
#include <SushiBLAS/sparse_matrix.hpp>
#include <SushiBLAS/sparse_builder.hpp>
#include <SushiBLAS/engine.hpp>

#include <SushiBLAS/io.hpp>
#include <SushiBLAS/engine/blas.hpp>
#include <SushiBLAS/engine/sparse.hpp>
#include <SushiBLAS/engine/spatial.hpp>
#include <SushiBLAS/engine/logic/logic.hpp>
#include <SushiBLAS/engine/math/random.hpp>
#include <SushiBLAS/engine/math/nonlinear.hpp>
#include <SushiBLAS/engine/math/reductions.hpp>
#include <SushiBLAS/engine/math/elementwise.hpp>
