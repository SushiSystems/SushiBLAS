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
| `format.hpp` | The C++17 stand-in for `std::format` that log and error messages use |
| `logger.hpp` | The `SB_LOG_*` macros on SushiLog, `SB_ASSERT` and `SB_THROW_IF` |
| `sushiblas_log_project.hpp` | `K_SUSHIBLAS_LOG_PROJECT`, the project of every SushiBLAS log category |
| `core_log_category.hpp` | `K_CORE_LOG_CATEGORY`, the category `SB_ASSERT` and `SB_THROW_IF` log under |
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

## `format.hpp`

`std::format` is C++20 and this library is C++17, so the placeholder substitution the logger has
always done has to be spelled without it. This is that spelling, in its own header rather than
inside logger.hpp, because "turn a format string and some values into a string" is not logging: it
is the one thing logging happens to need, and keeping it separate is what lets it be read, tested
and eventually **deleted** on its own. When the whole stack can assume C++20 again, this file goes
away and `SushiBLAS::Core::format` becomes a call to `std::vformat`; nothing else moves.

SushiRuntime carries the same stand-in (`SushiRuntime::Detail::sr_format`), and this is
deliberately *not* a call into it: that one lives in SushiRuntime's own `Detail` namespace inside
its logger header, internal by this stack's own convention, not a utility it publishes, so reaching
into it would couple SushiBLAS's diagnostics to another repo's private implementation detail. The
approach is mirrored; the code is not shared.

SushiAI, which links SushiBLAS unconditionally, *does* share this one: through the public
`SushiBLAS::Core::format` entry below. See `Core::format` below for why a third copy of this file would
have been the wrong answer.

**What it supports, exactly.** Positional `{}` fields filled left to right, `{{` and `}}` for
literal braces, and the replacement-field spec subset `{:[[fill]align][0][width][.precision][type]}`
where `align` is one of `<`, `>` or `^` and `type` is one of `f`, `e`, `g` (floating point) or `d`,
`o`, `x`, `X` (integer). That is a strict subset of `std::format`'s grammar chosen to cover what the
two codebases that use it actually write, `{}` everywhere, `{:.4f}` on a handful of SushiBLAS log
lines, and SushiAI's report tables (`{:<18}`, `{:>8.3f}`, `{:08x}`), so every existing format string
means the same thing it meant under `std::format`. Anything richer, explicit argument indices, `+`,
`#`, `b`, `L`, chrono and range specs, is **not** supported and is ignored rather than diagnosed.

Defaults follow `std::format`, which mostly means they follow `operator<<`: an unaligned field pads
on the left (`std::right`), which is what both agree on for numbers. `std::format` left-aligns
*strings* by default and `operator<<` does not, so a padded string field must spell its `<`; every
call site in this stack already does.

**What it gives up, and why that is acceptable.** `std::format` with a literal format string is
`consteval`-checked: a placeholder count that disagrees with the argument count is a compile error.
C++17 has no way to recover that check for a string that arrives through a variadic macro, so a
mismatch degrades here to a best-effort message, a surplus placeholder is dropped, a surplus
argument is unused, exactly as it does in SushiRuntime today. Substituting a message is never
allowed to be the reason a program fails, which is why the failure mode is "a slightly worse
diagnostic" rather than a throw.

### `parse_format_spec`

At `if (i + 1 < field.size() && is_align_char(field[i + 1]))`: fill-then-align first, because the
fill character may itself be a digit or a '0' and must not be read as the zero flag or the width.
Two characters are tested before one so `{:0<4}` pads with '0' on the right rather than reading as
the zero flag followed by a stray '<'.

The early return on `field.front() != ':'` in `parse_format_spec`: no leading `:` means the field is
not a spec at all (an explicit argument index, say). That form is unsupported, so the field renders
as a plain one.

### `apply_format_spec`

`std::boolalpha` is unconditional rather than part of the spec: `std::format` renders `true` and
`false` by name, `operator<<` renders them as 1 and 0 by default, and a log line that reads
"contiguous: true" under `std::format` must not read "contiguous: 1" under the stand-in.

Centring (`^`) is not expressible as stream state, iostreams has no such adjustfield, so it is
absent here and handled by `FormatArg::write_centred` instead.

At `if (spec.fill != '\0') os << std::setfill(spec.fill);`: an explicit fill wins; the bare '0' flag
is `std::format`'s sign-aware zero padding, which is what std::internal spells, and which
`std::format` itself ignores once an alignment is given.

### `FormatArg`

A pointer to the value plus a function pointer that streams it, which is the same shape
`std::vformat` uses and is here for the same reason: it keeps the substitution loop a single
non-template function. The alternative, a recursive variadic template, instantiates a fresh loop for
every distinct argument list, and this library has 130-odd `SB_LOG_*` / `SB_THROW_IF` call sites, so
that cost lands on every build.

Warning: Non-owning. A `FormatArg` is only valid for the full-expression that created it, which is
all `format` needs and all it is used for.

### `FormatArg::write`

Saves and restores `os`'s formatting state around the insertion, so a `{:.4f}` field cannot leak
`std::fixed` onto the fields that follow it.

### `FormatArg::write_centred`

The one field shape stream state cannot express, so the value is rendered unpadded into a scratch
stream and the padding is split around it by hand: the odd character to the right, as `std::format`
places it.

### `vformat`

The type-erased half of `format`, so the parsing loop is compiled once for the whole library rather
than once per argument list.

At `if (close == std::string_view::npos)`: an unterminated field is a typo in the format string, not
a reason to lose the message: emit the rest of it verbatim so the author can see what they wrote.

The `next < arg_count` test in `vformat`: a field with no argument left to fill it is dropped.
Formatting a message never fails the program; the opening of the `format.hpp` section says why.

### `format` without arguments

Still goes through `vformat` rather than returning `fmt` directly, so `{{` and `}}` mean the same
thing whether or not the call has arguments.

### `format` with arguments

Spelled with a leading named parameter rather than a bare pack so the argument array in its body is
never zero-sized; the no-argument case is the overload without arguments.

### `Core::format`

**Why this is public.** Everything in a `Detail` namespace is internal by this stack's convention,
and `Detail::format` stayed there while SushiBLAS was the only caller. It is not any more: SushiAI
moved to C++17 too, links SushiBLAS unconditionally, and needs exactly this substitution for its own
`SA_THROW_IF` and its report tables. The three options were a third copy of this file, SushiAI
reaching into SushiBLAS's `Detail`, or one documented public entry, and one entry is the only one of
the three that still has a single place to delete when the stack returns to C++20 and this becomes a
call to `std::vformat`.

That makes it an **addition** to SushiBLAS's public API, not a change to it: `Detail::format` is
untouched, every existing caller still resolves to it, and this forwards rather than reimplements.
The supported grammar is the one the opening of the `format.hpp` section gives, exactly.

- `fmt`: The format string; `{}` fields are filled left to right.
- `args`: The values to substitute, in order.

Returns: The substituted string.

## `logger.hpp`

SushiBLAS keeps no logger of its own. The `SB_LOG_*` macros are a front on SushiLog, the one
logger of the process, built on `SL_LOG_FRONT_AT`: SushiBLAS's records go through the queue,
thread and sink SushiRuntime's records use, and SushiBLAS's levels are its own. SushiBLAS links
SushiLog directly, at the version `cli/sushistack.deps.toml` pins. The rules are the SushiSkills `logging`
standard.

| Part | SushiBLAS |
| --- | --- |
| Project | `K_SUSHIBLAS_LOG_PROJECT`, in `sushiblas_log_project.hpp`; `SB_LOG_LEVEL` sets its levels |
| Category | One `K_<MODULE>_LOG_CATEGORY` per module that logs, in `<module>_log_category.hpp`; core's is `K_CORE_LOG_CATEGORY` |
| Call | `SB_LOG_TRACE` to `SB_LOG_FATAL`, each taking the category, a format string and its arguments |
| Ceiling | `K_LOG_LEVEL_CEILING`: ERROR when `SB_ENABLE_LOGGING` is 0, DEBUG under `NDEBUG`, TRACE otherwise |

`SB_LOG_LEVEL=warn,blas_kernels=debug` shows every kernel launch and its chosen shape while
leaving the other modules at WARN. The test programs call `SushiLog::set_log_synchronous(true)` so records arrive in order;
SushiBLAS re-exports no SushiLog name.

The `SB_ENABLE_LOGGING` CMake option (`cmake/SushiBLASLogging.cmake`) defaults to ON;
`sb build -D SB_ENABLE_LOGGING=OFF` compiles out every call below ERROR. The definition is
PUBLIC, so a consumer compiles the same calls the library does.

### `SB_ASSERT`

In a build without `NDEBUG`, a false condition writes a FATAL record naming the condition and
the message, then aborts. SushiLog delivers a FATAL record on the calling thread after
draining its queue, so the line is written before the process ends. Under `NDEBUG` the macro
compiles to nothing.

### `SB_THROW_IF`

Formats the message with `Core::Detail::format`, writes it as an ERROR record under
`K_CORE_LOG_CATEGORY`, then throws `std::runtime_error` with the same text.

## `traits.hpp`

The project's compile-time type vocabulary.

`is_complex` has this one definition. The BLAS helpers of `operands` and `blas`, the
elementwise seam and the random seam all need the trait, and their headers are public, so a
consumer includes them side by side: a copy in each would be several identical primary templates
of one namespace in one translation unit. A header that uses the trait includes this file.
