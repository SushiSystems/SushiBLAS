/****************************************************************/
/* io.cpp                                                       */
/* SushiBLAS - https://github.com/SushiSystems/SushiBLAS        */
/* Copyright (c) 2026 Sushi Systems                             */
/* Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.    */
/* Commercial use requires a licence from Sushi Systems.        */
/****************************************************************/

#include <vector>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <iostream>
#include <SushiBLAS/io.hpp>
#include <SushiBLAS/engine.hpp>

namespace SushiBLAS
{
    namespace
    {
        /** @brief Holds the only .sushi format version this build reads and writes. */
        constexpr uint32_t K_SUSHI_FORMAT_VERSION = 1;

        /** @brief Returns the number of bytes the tensor's elements occupy. */
        size_t payload_bytes(const Tensor& t)
        {
            return static_cast<size_t>(t.num_elements) * Core::bytes_per_element(t.dtype);
        }

        /** @brief Reports whether the host reaches the tensor's storage only through the queue. */
        bool is_device_allocated(const Tensor& t)
        {
            return t.storage->strategy == &SushiRuntime::Memory::Strategies::Device::instance();
        }

        /** @brief Reads exactly @p bytes into @p destination and throws when the file ends first. */
        void read_exact(std::ifstream& ifs, char* destination, size_t bytes,
                        const char* part, const std::string& path)
        {
            ifs.read(destination, static_cast<std::streamsize>(bytes));

            const size_t bytes_read = static_cast<size_t>(ifs.gcount());
            SB_THROW_IF(bytes_read != bytes,
                        "File {} ends inside its {}: {} bytes expected, {} read; nothing was loaded.",
                        path, part, bytes, bytes_read);
        }

        /** @brief Writes the tensor's bytes to the stream, copying device storage to the host first. */
        void write_payload(Engine& engine, std::ofstream& ofs, const Tensor& t)
        {
            const size_t bytes = payload_bytes(t);

            if (is_device_allocated(t))
            {
                std::vector<char> buf(bytes);
                engine.get_context().get_queue().memcpy(buf.data(), t.data(), bytes).wait();
                ofs.write(buf.data(), static_cast<std::streamsize>(bytes));
            }
            else
                ofs.write(static_cast<const char*>(t.data()), static_cast<std::streamsize>(bytes));
        }

        /** @brief Reads the tensor's bytes from the stream and stores them once all have arrived. */
        void read_payload(Engine& engine, std::ifstream& ifs, Tensor& t, const std::string& path)
        {
            const size_t bytes = payload_bytes(t);

            std::vector<char> buf(bytes);
            read_exact(ifs, buf.data(), bytes, "tensor data", path);

            if (is_device_allocated(t))
                engine.get_context().get_queue().memcpy(t.data(), buf.data(), bytes).wait();
            else
                std::memcpy(t.data(), buf.data(), bytes);
        }
    } // namespace

    /** @brief Internal header for .sushi files */
    struct SushiHeader
    {
        char magic[6] = "SUSHI";
        uint32_t version = K_SUSHI_FORMAT_VERSION;
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

        write_payload(engine_, ofs, t);

        SB_LOG_INFO("Saved native SushiBLAS file: {}", path);
    }

    void IO::load(Tensor& t, const std::string& path)
    {
        std::ifstream ifs(path, std::ios::binary);
        SB_THROW_IF(!ifs.is_open(), "Failed to open file for reading: {}", path);

        SushiHeader header;
        read_exact(ifs, reinterpret_cast<char*>(&header), sizeof(SushiHeader), "header", path);

        SB_THROW_IF(std::string(header.magic, 5) != "SUSHI", "Invalid .sushi file magic.");
        SB_THROW_IF(header.version != K_SUSHI_FORMAT_VERSION,
                    "File {} has .sushi format version {}; this build reads version {} only.",
                    path, header.version, K_SUSHI_FORMAT_VERSION);
        SB_THROW_IF(header.rank != t.rank, "Rank mismatch in .sushi file.");

        for (int i = 0; i < t.rank; ++i)
            SB_THROW_IF(header.shape[i] != t.shape[i], "Dimension mismatch at index {}", i);

        SB_THROW_IF(header.dtype != static_cast<int32_t>(t.dtype),
                    "File {} holds dtype id {}; the tensor it is loaded into is {}.",
                    path, header.dtype, Core::to_string(t.dtype));
        SB_THROW_IF(header.layout != static_cast<int32_t>(t.layout),
                    "File {} holds layout id {}; the tensor it is loaded into has layout id {}.",
                    path, header.layout, static_cast<int32_t>(t.layout));

        read_payload(engine_, ifs, t, path);
    }

    void IO::save_bin(const Tensor& t, const std::string& path)
    {
        engine_.execute().wait();
        std::ofstream ofs(path, std::ios::binary);
        SB_THROW_IF(!ofs.is_open(), "Failed to open file: {}", path);

        write_payload(engine_, ofs, t);
    }

    void IO::load_bin(Tensor& t, const std::string& path)
    {
        std::ifstream ifs(path, std::ios::binary);
        SB_THROW_IF(!ifs.is_open(), "Failed to open file: {}", path);

        read_payload(engine_, ifs, t, path);
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
        else if (t.dtype == Core::DataType::HALF) dict += "<f2";
        else if (t.dtype == Core::DataType::INT32) dict += "<i4";
        else if (t.dtype == Core::DataType::INT64) dict += "<i8";
        // Every dtype now names itself. The chain used to fall through silently for HALF,
        // writing a header with an empty 'descr' and a payload numpy could not read - a
        // corrupt file with no error at either end. Anything unrecognised says so instead.
        else SB_THROW_IF(true, "Cannot write a .npy header for dtype {}.", Core::to_string(t.dtype));


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
        write_payload(engine_, ofs, t);

        SB_LOG_INFO("Exported NumPy file: {}", path);
    }

    void IO::load_npy(Tensor& t, const std::string& path)
    {
        std::ifstream ifs(path, std::ios::binary);
        SB_THROW_IF(!ifs.is_open(), "Failed to open file for reading: {}", path);

        char magic[6];
        read_exact(ifs, magic, 6, "magic string", path);
        SB_THROW_IF(std::string(magic, 6) != "\x93NUMPY", "Invalid .npy file magic.");

        uint8_t major, minor;
        read_exact(ifs, reinterpret_cast<char*>(&major), 1, "format version", path);
        read_exact(ifs, reinterpret_cast<char*>(&minor), 1, "format version", path);

        uint16_t header_len;
        read_exact(ifs, reinterpret_cast<char*>(&header_len), 2, "header length", path);

        std::string header(header_len, ' ');
        read_exact(ifs, &header[0], header_len, "header", path);

        // TODO: Implement comprehensive .npy header parsing for metadata verification (dtype, fortran_order).
        // TODO: Enhance shape verification logic to ensure exact matching.
        size_t shape_pos = header.find("'shape': (");
        if (shape_pos != std::string::npos)
            for (int i = 0; i < t.rank; ++i)
                if (header.find(std::to_string(t.shape[i]), shape_pos) == std::string::npos)
                    SB_THROW_IF(true, "Shape dimension {} mismatch in .npy header.", i);

        // TODO: Optimize data loading path for different allocation strategies and layout orders.
        read_payload(engine_, ifs, t, path);

        SB_LOG_INFO("Loaded NumPy file: {}", path);
    }

    std::string IO::to_string(const Tensor& t, int precision, int edge_items)
    {
        engine_.execute().wait();

        size_t bpe = Core::bytes_per_element(t.dtype);

        std::vector<char> host_buf(t.num_elements * bpe);
        engine_.get_context().get_queue().memcpy(host_buf.data(), t.data(), t.num_elements * bpe).wait();

        std::vector<std::pair<float, float>> host_data(t.num_elements);

        for (int64_t i = 0; i < t.num_elements; ++i)
        {
            if (t.dtype == Core::DataType::HALF) host_data[i] = {static_cast<float>(reinterpret_cast<sycl::half*>(host_buf.data())[i]), 0.0f};
            else if (t.dtype == Core::DataType::FLOAT32) host_data[i] = {reinterpret_cast<float*>(host_buf.data())[i], 0.0f};
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
            // Printed through the same float pipeline as everything else, which is lossy
            // past 2^24 for INT32 and 2^53 for INT64. Acceptable for a human-readable dump
            // and only there - `save_npy` above writes the raw bytes, so nothing round-trips
            // through this.
            else if (t.dtype == Core::DataType::INT32) host_data[i] = {static_cast<float>(reinterpret_cast<int32_t*>(host_buf.data())[i]), 0.0f};
            else if (t.dtype == Core::DataType::INT64) host_data[i] = {static_cast<float>(reinterpret_cast<int64_t*>(host_buf.data())[i]), 0.0f};
            else SB_THROW_IF(true, "Cannot render a tensor of dtype {} as text.", Core::to_string(t.dtype));
        }

        std::stringstream ss;
        ss << "Tensor(shape=[";

        for (int i = 0; i < t.rank; ++i)
            ss << t.shape[i] << (i == t.rank - 1 ? "" : ", ");

        ss << "], dtype=";

        if (t.dtype == Core::DataType::HALF) ss << "f16";
        else if (t.dtype == Core::DataType::FLOAT32) ss << "f32";
        else if (t.dtype == Core::DataType::FLOAT64) ss << "f64";
        else if (t.dtype == Core::DataType::COMPLEX32) ss << "c32";
        else if (t.dtype == Core::DataType::COMPLEX64) ss << "c64";
        else if (t.dtype == Core::DataType::INT32) ss << "i32";
        else if (t.dtype == Core::DataType::INT64) ss << "i64";

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
