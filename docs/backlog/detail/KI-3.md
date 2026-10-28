# KI-3 Host `HALF` truncates below 2^-13 where a device rounds

The row of this entry is in `docs/backlog/BEHAVIOUR.md`. This file holds what did not fit the row.

- Origin: toolchain
- Where: `modules/foundation/reference/include/SushiBLAS/reference/serial_1d.hpp`
- Cause: the toolchain's host `sycl::half` treats the lowest normal exponent as subnormal
- Recognise: `HALF` parity is one unit in the last place apart below about 1.2e-4, bitwise above
- Workaround: compare `HALF` below that magnitude to one unit in the last place

`RW-50` rounds the host conversion.
