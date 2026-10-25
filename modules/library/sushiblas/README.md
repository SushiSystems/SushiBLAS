# sushiblas

The library a consumer links. `CMakeLists.txt` composes the `sushiblas` archive from the object
library of every module, installs the headers of the modules that are not internal, and exports
`SushiBLAS::sushiblas` with the two SYCL targets of `cmake/SyclTarget.cmake`:
`SushiBLAS::sycl_headers`, which the archive links, and `SushiBLAS::device_code`, which a
consumer's unit that instantiates a kernel compiles with.

The module has no header and no source of its own, and declares no dependency: it reads the
object libraries and include roots that `sushiblas_add_module()` records
(`cmake/SushiBLASModule.cmake`). The header a consumer includes is the project umbrella,
`<SushiBLAS/SushiBLAS.hpp>`, in the `umbrella` module.
