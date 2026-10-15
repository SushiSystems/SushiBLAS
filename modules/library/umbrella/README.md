# umbrella

The project umbrella, the one header a consumer needs. `SushiBLAS.hpp` includes every public
header of SushiBLAS: the header of every module that is not internal.

It depends on every installed module, from `core` up to the operation families. Its header is
included as `<SushiBLAS/SushiBLAS.hpp>`. The library a consumer links is composed by
`library/sushiblas`.

| File | Declares |
| --- | --- |
| `SushiBLAS.hpp` | Includes every public header of SushiBLAS |

A header added to a module that is not internal gets its line in `SushiBLAS.hpp` in the same
change.
