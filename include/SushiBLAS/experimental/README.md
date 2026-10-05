# Experimental headers

`include/SushiBLAS/experimental` holds declarations that no source file implements. `engine.hpp`
does not include them.

Each section below holds the reasoning a header carried in its comments until 2026-10-05, under the
name of the file. The header's first class block cites this page.

## `lapack/linalg.hpp`

`linalg.hpp` is experimental and quarantined under decision SB-2/D1. `engine.hpp` does not include
it, and `Engine::linalg()`, the accessor that reached it, was removed. No `.cpp` file implements any
method it declares. The LAPACK backend strategy is the named follow-up P8; the presence of
`linalg.hpp` does not settle it.

## `signal/transforms.hpp`

`transforms.hpp` is experimental and quarantined under decision SB-2/D1. `engine.hpp` does not
include it, and `Engine::signal()`, the accessor that reached it, was removed. No `.cpp` file
implements any method it declares. The FFT backend strategy is the named follow-up P8; the presence
of `transforms.hpp` does not settle it.
