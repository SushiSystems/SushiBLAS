# Backlog migration: inventory

**Status:** Shipped — 2026-10-19. The owner approved the inventory as proposed.

The migration of this repository to the backlog format of SushiSkills'
`skills/documentation/BACKLOG_FORMAT.md`, step 2 of section 8 of SushiSkills'
`docs/agent/2026_10_19_BACKLOG_FORMAT/SPEC.md`.

## Fact check

Both files were checked against the tree at `ef6efd7` (2026-10-19, "bring the backlog and the
known issues in line with the tree"), and no commit has landed since. That check stands for
every entry; this migration re-checked only what it changes:

| Entry | Verdict |
| --- | --- |
| Confirm that SushiLog's sink sends log output to stderr (C6) | Closed: with no sink set, `sink_holder.cpp` writes each record to stderr, as SushiLog's guide states |
| Ask SushiRuntime for a "run after" edge and wider `Recorder::task` overloads | Foreign: closed by the runtime; added there as `sushiruntime RW-60` |
| Tag SushiRuntime v0.4.0 | Foreign: the runtime's `RW-3` |
| The batch feed needs an operand-address late binding the runtime lacks | Its runtime half added there as `sushiruntime RW-59`; this entry is blocked by it |
| Group the includes, in step with the runtime | Split: the runtime half is `sushiruntime RW-61` |
| Take build, test, run and clean from a shared registrar | Blocked by `sushicore RW-15` |
| Document colour, theme and exit codes once in sushicore | The page is `sushicore RW-40`; linking it here is blocked by it |
| Settle panel names and options across the command lines | Foreign to the repository that coordinates the command lines; it stays as `RW-81` until that repository migrates |
| Migrate SushiAI onto the views | Foreign: it stays as `RW-91` until SushiAI migrates |
| Decide the licence of the code of conduct text | Foreign: the file is SushiSkills' shared copy, so the decision goes there |
| Get CI green on the commit to be tagged; cut the changelog | Folded into the decision `RW-7` and the tag `RW-8` |

## Shape

- The thirteen rows of "Verification state" in the known issues are built work never measured;
  they become backlog entries `RW-10` to `RW-17`, merged with the backlog's own copies of the same
  runs. "The open items each design lists" becomes `RW-18`.
- "No release is tagged" and "the release workflow has never run" are not defects; they are
  `RW-8` and `RW-88`.
- The kernel audit's thirty items become four entries, `RW-51` to `RW-54`, each citing the
  audit's `REPORT.md`; the entries' bodies keep every item, since the report does not name them
  all.
- Every own defect names its fix: an existing entry, or `none, <reason>` where the behaviour is
  the design (`KI-2`, `KI-4`, `KI-11`, `KI-14`, `KI-16`). Four defects are the toolchain's or the
  runtime's: `KI-1` cites `sushiruntime KI-1`.

The backlog holds `RW-1` to `RW-91`, nine of them decisions, twenty-two unmeasured and six
blocked; the known issues hold `KI-1` to `KI-29`.

## Changes in sibling repositories

| Repository | Change |
| --- | --- |
| sushiruntime | `RW-59` operand-address late binding, `RW-60` a "run after" edge and wider task overloads, `RW-61` include grouping |
| sushiskills | A backlog entry for the licence of the shared code of conduct text |

## Status lines

Ten files took the four forms: three design documents (`GPU_VALIDATION.md` open,
`MODULE_SPLIT.md` and `PUBLIC_SEAM.md` shipped on 2026-10-09) and seven work folders, of which
`2026_10_19_KERNEL_AUDIT` stays open.

## Other files

| File | Finding |
| --- | --- |
| Roadmap, journal, handover, milestone or priority files | None |
| Local branches `docs/root_readme`, `fix/packed_product_parity`, `pre-clean-main` | Not in any entry; left to the owner |

## Outcome

The owner approved the inventory on 2026-10-19.

```
$ python tools/check_all.py .
clean: 13 checkers, tests
```
