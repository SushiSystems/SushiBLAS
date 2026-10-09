# Documentation style guide

Prose in this repository follows the `humanizer` skill of SushiSkills, in the language it is
written in. Where a document goes, how a changelog entry reads and how an agent's work folder is
named follow the `documentation` skill. Neither is restated here.

## What this repository adds

- A symbol, path, CMake option or command is written in a code span and spelled as the code
  spells it: `Core::DataType`, `SB_SYCL_TARGETS`, `sb test --suite package`.
- A path is written from the repository root, such as `include/SushiBLAS/engine/task.hpp`. A
  header named alone, such as `engine/task.hpp`, is relative to `include/SushiBLAS/`.
- A figure states what was measured, on which device and when. A claim that was derived and not
  measured says so; [the GPU validation design](design/GPU_VALIDATION.md) holds the claims that
  wait for hardware.
- A decision or work item cited by its short name (D1, D3, SB-6) is defined in
  [the glossary](reference/GLOSSARY.md).
- The CLI's manual is [`cli/README.md`](../cli/README.md), beside the code it describes.
- `docs/reference/MAINPAGE.md` is the page Doxygen opens the generated reference with. The Doxyfile,
  `.config/doxygen/Doxyfile`, reads it by that path.

## Words

English, British spelling in prose ("licence" the noun, "optimisation"). Identifiers keep the
spelling the code uses, such as `K_LICENSE_LINES` or the `LICENSE` file.

## Pages not yet through the humanizer pass

`architecture/README.md`, `guides/INTEGRATION.md` and `design/GPU_VALIDATION.md` predate
this guide and were moved without a rewrite.
[`design/REMAINING_WORK.md`](design/REMAINING_WORK.md) lists the pass.
