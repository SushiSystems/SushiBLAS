/**************************************************************************/
/* io.hpp                                                                 */
/**************************************************************************/
/*                          This file is part of:                         */
/*                                SushiBLAS                               */
/*                https://github.com/SushiSystems/SushiBLAS               */
/*                         https://sushisystems.io                        */
/**************************************************************************/
/* Copyright (c) 2026-present  Mustafa Garip & Sushi Systems              */
/*                                                                        */
/* Permission is hereby granted, free of charge, to any person obtaining  */
/* a copy of this software and associated documentation files (the        */
/* "Software"), to deal in the Software without restriction, including    */
/* without limitation the rights to use, copy, modify, merge, publish,    */
/* distribute, sublicense, and/or sell copies of the Software, and to     */
/* permit persons to whom the Software is furnished to do so, subject to  */
/* the following conditions:                                              */
/*                                                                        */
/* The above copyright notice and this permission notice shall be         */
/* included in all copies or substantial portions of the Software.        */
/*                                                                        */
/* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,        */
/* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF     */
/* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. */
/* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY   */
/* CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,   */
/* TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE      */
/* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.                 */
/**************************************************************************/

#pragma once

#include <string>
#include <SushiBLAS/tensor.hpp>

namespace SushiBLAS
{
    class Engine;

    /**
     * @class IO
     * @brief Input/Output operations for Tensors.
     * 
     * This class provides methods to save, load, and display tensors. It supports
     * both native .sushi formats and NumPy .npy exports for interoperability.
     */
    class IO 
    {
        public:
            /**
             * @brief Construct a new IO object.
             * @param e Reference to the parent Engine.
             */
            explicit IO(Engine& e) : engine_(e) {}

            /**
             * @brief Save tensor in the native .sushi format.
             * 
             * This includes a header with metadata (shape, dtype) followed by raw data.
             * @param t The tensor to save.
             * @param path The filesystem path to save the file to.
             */
            // TODO: Implement asynchronous support for save operations using TaskGraph (add_host_node).
            void save(const Tensor& t, const std::string& path);

            /**
             * @brief Load a .sushi file into a tensor.
             * 
             * This will verify that the file metadata matches the tensor.
             * @param t The target tensor to load data into.
             * @param path The filesystem path of the .sushi file.
             */
            void load(Tensor& t, const std::string& path);

            /**
             * @brief Save tensor in NumPy .npy format.
             * 
             * This allows the tensor to be loaded directly in Python using np.load().
             * @param t The tensor to export.
             * @param path The filesystem path to save the .npy file.
             */
            // TODO: Implement asynchronous support for save operations using TaskGraph (add_host_node).
            void save_npy(const Tensor& t, const std::string& path);

            /**
             * @brief Load a NumPy .npy file into a tensor.
             * 
             * Parses the NumPy header and verifies compatibility with the target tensor.
             * @param t The target tensor to load data into.
             * @param path The filesystem path of the .npy file.
             */
            void load_npy(Tensor& t, const std::string& path);

            /**
             * @brief Raw binary save (no header).
             * @param t The tensor to save.
             * @param path The filesystem path for the binary file.
             */
            // TODO: Implement asynchronous support for save operations using TaskGraph (add_host_node).
            void save_bin(const Tensor& t, const std::string& path);

            /**
             * @brief Raw binary load (no header).
             * @param t The target tensor to load data into.
             * @param path The filesystem path of the binary file.
             */
            void load_bin(Tensor& t, const std::string& path);

            /**
             * @brief Print tensor content to the console in a readable format.
             * @param t The tensor to print.
             * @param precision Number of decimal places for numeric display.
             * @param edge_items Number of items to show at the edges of each dimension.
             */
            void print(const Tensor& t, int precision = 4, int edge_items = 3);

            /**
             * @brief Convert tensor content to a formatted string.
             * 
             * This captures the same formatting used by the print() method.
             * @param t The tensor to convert.
             * @param precision Number of decimal places for numeric display.
             * @param edge_items Number of items to show at the edges of each dimension.
             * @return A formatted string representation of the tensor.
             */
            std::string to_string(const Tensor& t, int precision = 4, int edge_items = 3);

        private:
            /**
             * @brief Internal recursive printing engine for N-dimensional tensors.
             */
            void print_recursive(std::ostream& os, const float* data, const int64_t* shape, 
                                 const int64_t* strides, int rank, int current_dim, 
                                 int64_t offset, int precision, int edge_items);

            Engine& engine_;
    };

} // namespace SushiBLAS
