# SushiBLAS manual

The front door is the [root README](../README.md): what the library is, how to build it and how
to run its tests. This index reaches every other document.

## Architecture

| Document | Holds |
| --- | --- |
| [Architecture](architecture/ARCHITECTURE.md) | The layers, the execution modes, the tensor, storage and memory model, the engine, the recorder and the views, the dispatch seams and fusion hooks, the kernels and routines of each family, the SushiRuntime dependency, packaging and the exception types |

## Guides

| Document | Holds |
| --- | --- |
| [Integration](guides/INTEGRATION.md) | How another CMake project consumes SushiBLAS and what it must supply |
| [FAQ](guides/FAQ.md) | The questions a newcomer asks, each answered from this manual |
| [The CLI](../cli/README.md) | Every `sb` command and option; the CLI's own README |

## Reference

| Document | Holds |
| --- | --- |
| [Changelog](reference/CHANGELOG.md) | What changed, one line per change |
| [Glossary](reference/GLOSSARY.md) | The words these documents use |
| [Known issues](reference/KNOWN_ISSUES.md) | Defects that are recorded and not yet fixed |
| [Security policy](reference/SECURITY.md) | How to report a vulnerability and what is in scope |
| [Code of conduct](reference/CODE_OF_CONDUCT.md) | The conduct expected in project spaces |
| [API reference main page](api/mainpage.md) | The page Doxygen opens the generated reference with |

## Design

| Document | Holds |
| --- | --- |
| [Design map](design/README.md) | Which design document covers which topic |
| [GPU validation](design/GPU_VALIDATION.md) | The claims that wait for a run on GPU hardware |
| [Remaining work](design/REMAINING_WORK.md) | The backlog |

## Contributing

| Document | Holds |
| --- | --- |
| [Contributing](CONTRIBUTING.md) | How a change is built, tested, documented and committed |
| [Documentation style guide](DOCUMENTATION_STYLE_GUIDE.md) | How prose is written here |

Agent work folders are under `agent/`. Finished material is under `archive/` and is not edited.
`api-site/` is Doxygen output, written by `sb docs` and ignored by git.
