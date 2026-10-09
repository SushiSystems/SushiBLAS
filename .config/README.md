# .config

Configuration a tool can be pointed at, kept out of the repository root.

Every dotfile still in the root is one a tool refuses to find anywhere else: clang-format and
EditorConfig walk up from the file being edited, and git reads `.gitignore` and `.gitattributes`
for the folder they sit in, so only the root covers the whole tree. Everything a flag or a
configuration key can redirect lives here instead.

| File | Who reads it | The flag or key that points at it | The call site that passes it |
| --- | --- | --- | --- |
| `doxygen/Doxyfile` | doxygen | the configuration file is doxygen's first positional argument; paths inside it resolve against the working directory, not against this one | `cli/sushiblas/services/project.py`, in `docs()` |

Adding a file here means adding a row. A row with an empty last column is a file nothing reads,
which is a file to delete rather than to document.

`.github/` is not here: it holds what GitHub resolves by name, the workflows and the community
files.
