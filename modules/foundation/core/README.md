# core

The foundation module every other module is written in: element types, traits, the format
stand-in, the logger, the error type and the version macros. It depends on no other module; its
logger is a front on SushiLog's. Its headers are included as `<SushiBLAS/core/<name>.hpp>`.

| File | Declares |
| --- | --- |
| `common.hpp` | `Layout`, `DataType` and the helpers that classify and size an element type |
| `blas_types.hpp` | The flags of a BLAS call: `Transpose`, `Uplo`, `Diag` and `Side` |
| `sparse_types.hpp` | `IndexBase`, which says what value names the first row, column or position |
| `errors.hpp` | `UnsupportedExecution`, the exception a mode or device refusal throws |
| `traits.hpp` | The compile-time type traits the library shares |
| `export.hpp` | `SUSHIBLAS_API`, the symbol visibility macro of each platform |
| `logger.hpp` | The `SB_LOG_*` macros on SushiLog, `SB_ASSERT` and `SB_THROW_IF` |
| `sushiblas_log_project.hpp` | `K_SUSHIBLAS_LOG_PROJECT`, the project of every SushiBLAS log category |
| `version.hpp` | The `SUSHIBLAS_VERSION_*` macros; `CMakeLists.txt` refuses a value that differs from `project()` |

Each section below holds the reasoning a header carried in its comments until 2026-10-05, under the
name of the file and of the symbol it belongs to. The header states the rule in one line and cites
this page.

## `common.hpp`

### `DataType`

**The integer types are not general-purpose arithmetic types here.** They exist because some results
are indices rather than numbers: `iamax`, `argmax` and `argmin` all answer "which element", and
answering that in a FLOAT32 tensor meant either writing eight bytes into a four-byte element (which
is what `iamax` did, corrupting whatever a memory planner had packed next to it) or rounding the
index through a mantissa (which is what `argmax` did, silently losing exactness past 2^24). They are
also the right element type for a comparison result, which is a truth value and not a float.

So only the operations for which an integer element means something accept them: `copy`, `constant`,
`add`, `sub`, `mul`, `neg`, `abs`, `min`, `max`, `clamp`, the sparse gathers and scatter, the
channel copies, the comparison and logic family, the index reductions, and `io`. Everything else,
every elementwise transcendental, every non-linearity, every BLAS product, every folding reduction,
rejects them by name at record time. That is deliberate and it is the *point*: a dtype that is
quietly accepted and then silently mishandled is worse than one that is not accepted at all. See
`SushiBLAS::Dispatch::Domain`, which is how each op states which element types it will instantiate
for.

**The numeric values are a file-format contract.** Each enumerator is written out with its value,
`Half = 0` to `Int64 = 6`. No value changes, and a new element type is appended after the last.

### `to_string`

Exists so that a "this operation does not support that dtype" failure can name the offending type
instead of printing an integer the reader has to decode.

### `is_integer_dtype`

The counterpart to `is_complex_dtype`, and used the same way: to reject a dtype at record time in
the handful of places that check a whole family at once rather than switching arm by arm.

### `bytes_per_element`

The one place the dtype-to-width mapping is written down. Three callers need it (the Tensor
constructor's capacity check, Tensor::data(), and io.cpp); a chain written in each is three
chances for a new dtype to be added to two of them.

The `return 4;` after the switch in `bytes_per_element` is unreachable for any enumerator. It
matches the shape of `to_string()`, which needs a trailing return for a `DataType` holding a value
outside the enumeration.

## `blas_types.hpp`

### `Transpose`, `Uplo`, `Diag`, `Side`

The enumerators carry the CBLAS numbers: `NoTrans`, `Trans` and `ConjTrans` are 111 to 113,
`Upper` and `Lower` 121 and 122, `NonUnit` and `Unit` 131 and 132, `Left` and `Right` 141 and
142. A value-initialised flag is therefore 0 and names nothing. `is_valid` reports that, and the
record-time flag check in `operands/argument_check.hpp` throws `std::invalid_argument` for it,
so a flag a caller forgot to set never reads as `NoTrans` or `Upper`. Each enumeration is one
byte and fits one `TaskDescription` slot. `is_transposed` and `is_conjugated` answer the two
questions a kernel asks of a `Transpose`; for a real element type `ConjTrans` is `Trans`.

## `sparse_types.hpp`

### `IndexBase`

`Zero` is 221 and `One` is 222, so a value-initialised base is 0 and `is_valid` refuses it, as
with the BLAS flags. `first_index` turns a base into the stored value that names row 0, column 0
and position 0. The sparse operand describers carry that value to the kernel, which subtracts
it on each read through `Kernels::Sparse::CsrPattern` and `IndexList`.

## `errors.hpp`

### `UnsupportedExecution`

Thrown when an operation cannot run under the execution mode or on the device: a device-only
operand under `ForceReference`, FLOAT64 or COMPLEX64 storage on a device without the fp64 aspect,
HALF storage on a device without fp16, a SYRK or SYR2K on a device that cannot host the local
tile of its SYCL kernel, `random.poisson` on a device without fp64. It derives from `std::invalid_argument`, so a `catch (const
std::invalid_argument&)` still catches it, and a caller that wants to tell a refusal from a wrong
operand puts `catch (const SushiBLAS::UnsupportedExecution&)` first. It is thrown at record time
only, by `Dispatch::Detail::throw_execution_error` in `engine/operand_check.hpp`.

## `logger.hpp`

SushiBLAS keeps no logger of its own. The `SB_LOG_*` macros are a front on SushiLog, the one
logger of the process, built on `SL_LOG_FRONT_AT`: SushiBLAS's records go through the queue,
thread and sink SushiRuntime's records use, and SushiBLAS's levels are its own. SushiBLAS links
SushiLog directly, at the version `cli/sushistack.deps.toml` pins. The rules are the SushiSkills `logging`
standard.

| Part | SushiBLAS |
| --- | --- |
| Project | `K_SUSHIBLAS_LOG_PROJECT`, in `sushiblas_log_project.hpp`; `SB_LOG_LEVEL` sets its levels |
| Category | One `K_<MODULE>_LOG_CATEGORY` per module that logs, in `<module>_log_category.hpp` |
| Call | `SB_LOG_TRACE` to `SB_LOG_FATAL`, each taking the category, a format string and its arguments |
| Ceiling | `K_LOG_LEVEL_CEILING`: ERROR when `SB_ENABLE_LOGGING` is 0, DEBUG under `NDEBUG`, TRACE otherwise |

`SB_LOG_LEVEL=warn,blas_kernels=debug` shows every kernel launch and its chosen shape while
leaving the other modules at WARN. The test programs call `SushiLog::set_log_synchronous(true)` so records arrive in order;
SushiBLAS re-exports no SushiLog name.

The `SB_ENABLE_LOGGING` CMake option (`cmake/SushiBLASLogging.cmake`) defaults to ON;
`sb build -D SB_ENABLE_LOGGING=OFF` compiles out every call below ERROR. The definition is
PUBLIC, so a consumer compiles the same calls the library does.

### `SB_ASSERT`

`SB_ASSERT(category, condition, msg)`: in a build without `NDEBUG`, a false condition writes a
FATAL record in the caller's category naming the condition and the message, then aborts.
SushiLog delivers a FATAL record on the calling thread after draining its queue, so the line is
written before the process ends. Under `NDEBUG` the macro compiles to nothing.

### `SB_THROW_IF`

Formats the message with SushiRuntime's `sr_format` (`<SushiRuntime/core/text_format.hpp>`) and
throws `std::runtime_error` with it. It logs nothing: a failure is logged once, by the code that
handles it.

## `traits.hpp`

The project's compile-time type vocabulary.

`is_complex` has this one definition. The BLAS helpers of `operands` and `blas`, the
elementwise seam and the random seam all need the trait, and their headers are public, so a
consumer includes them side by side: a copy in each would be several identical primary templates
of one namespace in one translation unit. A header that uses the trait includes this file.
