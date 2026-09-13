"""Deprecated legacy entrypoint.

The maintained experiment entrypoints are:
    python -m src.run_pretrain
    python -m src.run_section25

The former §2.3/§2.4 coordinator depended on archived modules and is no longer
part of the active code path. Keep this shim so accidental invocation fails
immediately with an actionable message instead of importing `_archive` and
breaking deep inside the pipeline.
"""
from __future__ import annotations


def main(argv=None) -> int:
    print(
        "src.run_recomb is deprecated and intentionally disabled. "
        "Use `python -m src.run_pretrain` for Stage 0 or "
        "`python -m src.run_section25` for the maintained §2.5 pipeline."
    )
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
