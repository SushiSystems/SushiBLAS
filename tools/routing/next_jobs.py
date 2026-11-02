# next_jobs.py
# SushiSkills - https://github.com/SushiSystems/SushiSkills
# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.
# Commercial use requires a licence from Sushi Systems.
"""Prints the jobs of a plan that the orchestrator may start now.

Usage: python tools/routing/next_jobs.py PLAN.md [--free N]

Exit 0 when at least one job is printed, 1 when none is ready and jobs remain, 2 when every
job is done and 3 when the table is malformed.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from routing.job_table import read_jobs  # noqa: E402
from routing.ready_jobs import K_DONE, graph_problems, ready_jobs  # noqa: E402

K_EXIT_READY = 0
K_EXIT_WAITING = 1
K_EXIT_FINISHED = 2
K_EXIT_MALFORMED = 3


def main(argv: list[str] | None = None) -> int:
    """Prints the ids of the ready jobs of a plan, one per line, and returns the exit code."""
    parser = argparse.ArgumentParser(description="Lists the jobs of a plan that may start now.")
    parser.add_argument("plan", type=Path)
    parser.add_argument("--free", type=int, default=1, help="how many subagents are free")
    arguments = parser.parse_args(argv)
    jobs, problems = read_jobs(arguments.plan.read_text(encoding="utf-8"))
    if not problems:
        problems = graph_problems(jobs)
    for problem in problems:
        print(f"{arguments.plan}:{problem.line}: {problem.rule}: {problem.message}")
    if problems:
        return K_EXIT_MALFORMED
    if all(job.status == K_DONE for job in jobs):
        return K_EXIT_FINISHED
    ready = ready_jobs(jobs, arguments.free)
    for job_id in ready:
        print(job_id)
    return K_EXIT_READY if ready else K_EXIT_WAITING


if __name__ == "__main__":
    sys.exit(main())
