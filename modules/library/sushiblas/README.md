# sushiblas

The library a consumer links and the one header a consumer needs. `CMakeLists.txt` composes the
`sushiblas` archive from the object library of every module, installs the headers of the modules
that are not internal, and exports `SushiBLAS::sushiblas`.

It depends on every module the umbrella header includes. Its headers are included as
`<SushiBLAS/SushiBLAS.h>`.

| File | Declares |
| --- | --- |
| `SushiBLAS.h` | Includes the core, storage, tensor, engine and version headers and every operation family of SushiBLAS |
