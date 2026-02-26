/**************************************************************************/
/* io.cpp                                                                 */
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

#include <vector>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <iostream>
#include <SushiBLAS/io.hpp>
#include <SushiBLAS/engine.hpp>

namespace SushiBLAS
{
    /** @brief Internal header for .sushi files */
    struct SushiHeader 
    {
        char magic[6] = "SUSHI";
        uint32_t version = 1;
        int32_t rank;
        int64_t shape[Core::MAX_TENSOR_RANK];
        int32_t dtype;
        int32_t layout;
    };

    void IO::save(const Tensor& t, const std::string& path)
    {
        SushiHeader header;
        header.rank = t.rank;
        header.dtype = static_cast<int32_t>(t.dtype);
        header.layout = static_cast<int32_t>(t.layout);

        for (int i = 0; i < t.rank; ++i)
            header.shape[i] = t.shape[i];

        engine_.execute().wait();
        
        std::ofstream ofs(path, std::ios::binary);
        SB_THROW_IF(!ofs.is_open(), "Failed to open file for writing: {}", path);

        ofs.write(reinterpret_cast<const char*>(&header), sizeof(SushiHeader));
        
        size_t bpe = (t.dtype == Core::DataType::FLOAT64 || t.dtype == Core::DataType::COMPLEX32) ? 8 : 
                     (t.dtype == Core::DataType::COMPLEX64) ? 16 : 4;
        size_t bytes = t.num_elements * bpe;
        
        if (t.storage->strategy == SushiRuntime::Memory::AllocStrategy::DEVICE)
        {
            std::vector<char> buf(bytes);
            engine_.get_context().get_queue().memcpy(buf.data(), t.data(), bytes).wait();
            ofs.write(buf.data(), bytes);
        }
        else
            ofs.write(static_cast<const char*>(t.data()), bytes);

        SB_LOG_INFO("Saved native SushiBLAS file: {}", path);
    }

    void IO::load(Tensor& t, const std::string& path)
    {
        std::ifstream ifs(path, std::ios::binary);
        SB_THROW_IF(!ifs.is_open(), "Failed to open file for reading: {}", path);

        SushiHeader header;
        ifs.read(reinterpret_cast<char*>(&header), sizeof(SushiHeader));

        SB_THROW_IF(std::string(header.magic, 5) != "SUSHI", "Invalid .sushi file magic.");
        SB_THROW_IF(header.rank != t.rank, "Rank mismatch in .sushi file.");

        for (int i = 0; i < t.rank; ++i)
            SB_THROW_IF(header.shape[i] != t.shape[i], "Dimension mismatch at index {}", i);

        size_t bpe = (t.dtype == Core::DataType::FLOAT64 || t.dtype == Core::DataType::COMPLEX32) ? 8 : 
                     (t.dtype == Core::DataType::COMPLEX64) ? 16 : 4;
        size_t bytes = t.num_elements * bpe;
        
        if (t.storage->strategy == SushiRuntime::Memory::AllocStrategy::DEVICE)
        {
            std::vector<char> buf(bytes);
            ifs.read(buf.data(), bytes);
            engine_.get_context().get_queue().memcpy(t.data(), buf.data(), bytes).wait();
        }
        else
            ifs.read(static_cast<char*>(t.data()), bytes);
    }

    void IO::save_bin(const Tensor& t, const std::string& path)
    {
        engine_.execute().wait();
        std::ofstream ofs(path, std::ios::binary);
        SB_THROW_IF(!ofs.is_open(), "Failed to open file: {}", path);

        size_t bpe = (t.dtype == Core::DataType::FLOAT64 || t.dtype == Core::DataType::COMPLEX32) ? 8 : 
                     (t.dtype == Core::DataType::COMPLEX64) ? 16 : 4;
        size_t bytes = t.num_elements * bpe;
        ofs.write(static_cast<const char*>(t.data()), bytes);
    }

    void IO::load_bin(Tensor& t, const std::string& path)
    {
        std::ifstream ifs(path, std::ios::binary);
        SB_THROW_IF(!ifs.is_open(), "Failed to open file: {}", path);

        size_t bpe = (t.dtype == Core::DataType::FLOAT64 || t.dtype == Core::DataType::COMPLEX32) ? 8 : 
                     (t.dtype == Core::DataType::COMPLEX64) ? 16 : 4;
        size_t bytes = t.num_elements * bpe;
        ifs.read(static_cast<char*>(t.data()), bytes);
    }

    void IO::save_npy(const Tensor& t, const std::string& path)
    {
        engine_.execute().wait();
        
        std::ofstream ofs(path, std::ios::binary);
        SB_THROW_IF(!ofs.is_open(), "Failed to open file: {}", path);

        // 1. Prepare NumPy Header
        const char magic[] = "\x93NUMPY";
        uint8_t major = 1;
        uint8_t minor = 0;
        
        std::string dict = "{'descr': '";

        if (t.dtype == Core::DataType::FLOAT32) dict += "<f4";
        else if (t.dtype == Core::DataType::FLOAT64) dict += "<f8";
        else if (t.dtype == Core::DataType::COMPLEX32) dict += "<c8";
        else if (t.dtype == Core::DataType::COMPLEX64) dict += "<c16";
        
        dict += "', 'fortran_order': ";
        dict += (t.layout == Core::Layout::COLUMN_MAJOR) ? "True" : "False";
        dict += ", 'shape': (";

        for (int i = 0; i < t.rank; ++i) 
        {
            dict += std::to_string(t.shape[i]);

            if (t.rank == 1 || i < t.rank - 1) dict += ",";
            if (i < t.rank - 1) dict += " ";
        }
        dict += "), }";

        while ((10 + dict.length() + 1) % 64 != 0) dict += " ";

        dict += "\n";

        uint16_t header_len = static_cast<uint16_t>(dict.length());

        ofs.write(magic, 6);
        ofs.write(reinterpret_cast<const char*>(&major), 1);
        ofs.write(reinterpret_cast<const char*>(&minor), 1);
        ofs.write(reinterpret_cast<const char*>(&header_len), 2);
        ofs.write(dict.c_str(), header_len);

        // 2. Write Data
        size_t bpe = (t.dtype == Core::DataType::FLOAT64 || t.dtype == Core::DataType::COMPLEX32) ? 8 : 
                     (t.dtype == Core::DataType::COMPLEX64) ? 16 : 4;
        
        size_t bytes = t.num_elements * bpe;

        if (t.storage->strategy == SushiRuntime::Memory::AllocStrategy::DEVICE)
        {
            std::vector<char> buf(bytes);
            engine_.get_context().get_queue().memcpy(buf.data(), t.data(), bytes).wait();
            ofs.write(buf.data(), bytes);
        }
        else
            ofs.write(static_cast<const char*>(t.data()), bytes);
        
        SB_LOG_INFO("Exported NumPy file: {}", path);
    }

    void IO::load_npy(Tensor& t, const std::string& path)
    {
        std::ifstream ifs(path, std::ios::binary);
        SB_THROW_IF(!ifs.is_open(), "Failed to open file for reading: {}", path);

        char magic[6];
        ifs.read(magic, 6);
        SB_THROW_IF(std::string(magic, 6) != "\x93NUMPY", "Invalid .npy file magic.");

        uint8_t major, minor;
        ifs.read(reinterpret_cast<char*>(&major), 1);
        ifs.read(reinterpret_cast<char*>(&minor), 1);

        uint16_t header_len;
        ifs.read(reinterpret_cast<char*>(&header_len), 2);

        std::string header(header_len, ' ');
        ifs.read(&header[0], header_len);

        // TODO: Implement comprehensive .npy header parsing for metadata verification (dtype, fortran_order).
        // TODO: Enhance shape verification logic to ensure exact matching.
        size_t shape_pos = header.find("'shape': (");
        if (shape_pos != std::string::npos)
            for (int i = 0; i < t.rank; ++i)
                if (header.find(std::to_string(t.shape[i]), shape_pos) == std::string::npos)
                    SB_THROW_IF(true, "Shape dimension {} mismatch in .npy header.", i);

        // TODO: Optimize data loading path for different allocation strategies and layout orders.
        size_t bpe = (t.dtype == Core::DataType::FLOAT64 || t.dtype == Core::DataType::COMPLEX32) ? 8 : 
                     (t.dtype == Core::DataType::COMPLEX64) ? 16 : 4;
        size_t bytes = t.num_elements * bpe;

        if (t.storage->strategy == SushiRuntime::Memory::AllocStrategy::DEVICE)
        {
            std::vector<char> buf(bytes);
            ifs.read(buf.data(), bytes);
            engine_.get_context().get_queue().memcpy(t.data(), buf.data(), bytes).wait();
        }
        else
            ifs.read(static_cast<char*>(t.data()), bytes);

        SB_LOG_INFO("Loaded NumPy file: {}", path);
    }

    std::string IO::to_string(const Tensor& t, int precision, int edge_items)
    {
        engine_.execute().wait();
        
        size_t bpe = (t.dtype == Core::DataType::FLOAT64 || t.dtype == Core::DataType::COMPLEX32) ? 8 : 
                     (t.dtype == Core::DataType::COMPLEX64) ? 16 : 4;
        
        std::vector<char> host_buf(t.num_elements * bpe);
        engine_.get_context().get_queue().memcpy(host_buf.data(), t.data(), t.num_elements * bpe).wait();

        std::vector<std::pair<float, float>> host_data(t.num_elements);
        
        for (int64_t i = 0; i < t.num_elements; ++i)
        {
            if (t.dtype == Core::DataType::FLOAT32) host_data[i] = {reinterpret_cast<float*>(host_buf.data())[i], 0.0f};
            else if (t.dtype == Core::DataType::FLOAT64) host_data[i] = {static_cast<float>(reinterpret_cast<double*>(host_buf.data())[i]), 0.0f};
            else if (t.dtype == Core::DataType::COMPLEX32) 
            {
                float* c_ptr = reinterpret_cast<float*>(host_buf.data()) + (i * 2);
                host_data[i] = {c_ptr[0], c_ptr[1]};
            }
            else if (t.dtype == Core::DataType::COMPLEX64)
            {
                double* c_ptr = reinterpret_cast<double*>(host_buf.data()) + (i * 2);
                host_data[i] = {static_cast<float>(c_ptr[0]), static_cast<float>(c_ptr[1])};
            }
        }

        std::stringstream ss;
        ss << "Tensor(shape=[";

        for (int i = 0; i < t.rank; ++i)
            ss << t.shape[i] << (i == t.rank - 1 ? "" : ", ");

        ss << "], dtype=";

        if (t.dtype == Core::DataType::FLOAT32) ss << "f32";
        else if (t.dtype == Core::DataType::FLOAT64) ss << "f64";
        else if (t.dtype == Core::DataType::COMPLEX32) ss << "c32";
        else if (t.dtype == Core::DataType::COMPLEX64) ss << "c64";

        ss << "):" << std::endl;
        ss << std::fixed << std::setprecision(precision);

        print_recursive(ss, reinterpret_cast<float*>(host_data.data()), t.shape.data(), 
                        t.strides.data(), t.rank, 0, 0, precision, edge_items);
        
        return ss.str();
    }

    void IO::print(const Tensor& t, int precision, int edge_items)
    {
        std::cout << to_string(t, precision, edge_items);
    }

    void IO::print_recursive(std::ostream& os, const float* data, const int64_t* shape, 
                             const int64_t* strides, int rank, int current_dim, 
                             int64_t offset, int precision, int edge_items)
    {
        auto get_val = [&](int64_t idx) -> std::string 
        {
            float re = data[idx * 2];
            float im = data[idx * 2 + 1];

            if (std::abs(im) < 1e-9) 
            {
                std::stringstream val_ss;
                val_ss << std::fixed << std::setprecision(precision) << re;
                return val_ss.str();
            }

            std::stringstream val_ss;
            val_ss << std::fixed << std::setprecision(precision) << "(" << re << (im >= 0 ? "+" : "") << im << "j)";
            return val_ss.str();
        };

        if (current_dim == rank - 1) 
        {
            os << "[";

            for (int64_t i = 0; i < shape[current_dim]; ++i)
            {
                if (shape[current_dim] > edge_items * 2 && i == edge_items)
                {
                    os << "... ";
                    i = shape[current_dim] - edge_items - 1;
                    continue;
                }
                os << get_val(offset + i * strides[current_dim]) << (i == shape[current_dim] - 1 ? "" : ", ");
            }

            os << "]";
            return;
        }

        os << "[";

        for (int64_t i = 0; i < shape[current_dim]; ++i)
        {
            if (shape[current_dim] > edge_items * 2 && i == edge_items)
            {
                os << "\n" << std::string(current_dim + 1, ' ') << "...\n" << std::string(current_dim + 1, ' ');
                i = shape[current_dim] - edge_items - 1;
                continue;
            }

            print_recursive(os, data, shape, strides, rank, current_dim + 1, 
                            offset + i * strides[current_dim], precision, edge_items);
            
            if (i < shape[current_dim] - 1)
                os << ",\n" << std::string(current_dim + 1, ' ');
        }

        os << "]";

        if (current_dim == 0)
            os << std::endl;
    }
} // namespace SushiBLAS