/**************************************************************************/
/* io_file_stream.cpp                                                     */
/**************************************************************************/
/*                          This file is part of:                         */
/*                                SushiBLAS                               */
/*                https://github.com/SushiSystems/SushiBLAS               */
/*                         https://sushisystems.io                        */
/**************************************************************************/
/* Copyright (c) 2026-present Mustafa Garip & Sushi Systems               */
/*                                                                        */
/* Licensed under the Apache License, Version 2.0 (the "License"); you    */
/* may not use this file except in compliance with the License. You may   */
/* obtain a copy of the License at                                        */
/*                                                                        */
/*     https://www.apache.org/licenses/LICENSE-2.0                        */
/*                                                                        */
/* Unless required by applicable law or agreed to in writing, software    */
/* distributed under the License is distributed on an "AS IS" BASIS,      */
/* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or        */
/* implied. See the License for the specific language governing           */
/* permissions and limitations under the License.                         */
/**************************************************************************/

/**
 * @file io_file_stream.cpp
 * @brief Implements the file-stream helpers the IO calls share.
 * @author Mustafa Garip
 * @see modules/operations/io/README.md
 */

#include "io_file_stream.hpp"

#include <algorithm>
#include <cstdint>
#include <memory>
#include "io_tensor_access.hpp"

namespace SushiBLAS
{
    namespace Detail
    {
        namespace
        {
            /** @brief Holds the bytes one file transfer stages on the host for device storage. */
            constexpr size_t K_IO_CHUNK_BYTES = size_t{4} << 20;

            /**
             * @brief Returns the bytes left in the stream after its read position.
             * @throws IOFailure with FileUnreadable if the stream cannot seek.
             */
            std::uintmax_t remaining_bytes(std::ifstream& ifs, const std::string& path)
            {
                const std::streampos here = ifs.tellg();
                ifs.seekg(0, std::ios::end);
                const std::streampos end = ifs.tellg();
                ifs.seekg(here);
                if (!ifs || here == std::streampos(-1) || end < here)
                    fail_io(IOStatus::FileUnreadable, "The length of file {} cannot be measured.",
                            path);
                return static_cast<std::uintmax_t>(end - here);
            }

            /**
             * @brief Reads bytes of tensor data into destination.
             * @throws IOFailure with FileUnreadable if the stream fails; destination then holds a
             *         partial load.
             */
            void read_tensor_bytes(std::ifstream& ifs, char* destination, size_t bytes,
                                   const std::string& path)
            {
                ifs.read(destination, static_cast<std::streamsize>(bytes));

                const size_t bytes_read = static_cast<size_t>(ifs.gcount());
                if (bytes_read != bytes)
                    fail_io(IOStatus::FileUnreadable,
                            "Reading the tensor data of {} failed after {} of {} bytes; the tensor "
                            "holds a partial load.",
                            path, bytes_read, bytes);
            }
        } // namespace

        void read_exact(std::ifstream& ifs, char* destination, size_t bytes,
                        const char* part, const std::string& path)
        {
            ifs.read(destination, static_cast<std::streamsize>(bytes));

            const size_t bytes_read = static_cast<size_t>(ifs.gcount());
            if (bytes_read != bytes)
                fail_io(IOStatus::Truncated,
                        "File {} ends inside its {}: {} bytes expected, {} read; nothing was "
                        "loaded.",
                        path, part, bytes, bytes_read);
        }

        void write_checked(std::ofstream& ofs, const void* source, size_t bytes, const char* part,
                           const std::string& path)
        {
            ofs.write(static_cast<const char*>(source), static_cast<std::streamsize>(bytes));
            if (!ofs)
                fail_io(IOStatus::FileUnwritable,
                        "Writing the {} of {} failed after {} bytes were requested.", part, path,
                        bytes);
        }

        void write_payload(std::ofstream& ofs, const Tensor& t, const std::string& path)
        {
            const size_t bytes = payload_bytes(t);
            if (bytes == 0)
                return;

            const char* source = static_cast<const char*>(t.data());
            if (!is_device_allocated(t))
            {
                write_checked(ofs, source, bytes, "tensor data", path);
                return;
            }

            const size_t chunk_bytes = std::min(bytes, K_IO_CHUNK_BYTES);
            const std::unique_ptr<char[]> chunk(new char[chunk_bytes]);
            sycl::queue& queue = storage_queue(t);
            for (size_t done = 0; done < bytes; done += chunk_bytes)
            {
                const size_t step = std::min(chunk_bytes, bytes - done);
                queue.memcpy(chunk.get(), source + done, step).wait();
                write_checked(ofs, chunk.get(), step, "tensor data", path);
            }
        }

        void read_payload(std::ifstream& ifs, Tensor& t, const std::string& path)
        {
            const size_t bytes = payload_bytes(t);
            if (bytes == 0)
                return;

            const std::uintmax_t available = remaining_bytes(ifs, path);
            if (available < bytes)
                fail_io(IOStatus::Truncated,
                        "File {} ends inside its tensor data: {} bytes expected, {} present; "
                        "nothing was loaded.",
                        path, bytes, available);

            char* destination = static_cast<char*>(t.data());
            if (!is_device_allocated(t))
            {
                read_tensor_bytes(ifs, destination, bytes, path);
                return;
            }

            const size_t chunk_bytes = std::min(bytes, K_IO_CHUNK_BYTES);
            const std::unique_ptr<char[]> chunk(new char[chunk_bytes]);
            sycl::queue& queue = storage_queue(t);
            for (size_t done = 0; done < bytes; done += chunk_bytes)
            {
                const size_t step = std::min(chunk_bytes, bytes - done);
                read_tensor_bytes(ifs, chunk.get(), step, path);
                queue.memcpy(destination + done, chunk.get(), step).wait();
            }
        }
    } // namespace Detail
} // namespace SushiBLAS
