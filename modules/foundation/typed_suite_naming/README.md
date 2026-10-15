# typed_suite_naming

The name generator a typed GoogleTest suite is declared with. The module is internal and test-only: its headers are not
installed, and it is added only when `SB_BUILD_TESTS` is on.

It depends on GoogleTest only. Its headers are included as `<SushiBLAS/typed_suite_naming/<name>.hpp>`.

| File | Declares |
| --- | --- |
| `default_type_names.hpp` | `DefaultTypeNames` and `SB_TYPED_TEST_SUITE` |

## default_type_names.hpp

### DefaultTypeNames and SB_TYPED_TEST_SUITE

`TYPED_TEST_SUITE(Suite, Types)` looks like a two-argument macro and is declared
`TYPED_TEST_SUITE(CaseName, Types, ...)`. The `...` is an optional name generator. Passing no
argument at all for a `...` is allowed from C++20 and is a diagnosed extension before it. On this
project's C++17 baseline the two-argument spelling is an error under `-pedantic -Werror`, raised
inside a third-party header the project does not own and cannot fix.

Naming a generator explicitly settles it. `DefaultTypeNames` restates GoogleTest's default
against its public API, without reaching into its `internal` namespace. The suffix is the type's
index, which is the name GoogleTest's own default gives.
