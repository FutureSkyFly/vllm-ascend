# SPDX-License-Identifier: Apache-2.0
"""Grade saved generations on CPU using SGLang's unchanged math scorer."""

import argparse
import importlib.metadata
import json
import os

from benchmarks.uno.sglang_math_grader import score_math


def main():
    from pathlib import Path

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()
    if os.name != "posix":
        parser.error(
            "Run grading on Linux: math-verify's Windows timeout workers cannot serialize their nested callback."
        )
    summary_path = args.output_dir / "summary.json"
    summary = json.loads(summary_path.read_text())
    assert summary["grading_status"] == "pending", "Preserve completed grading evidence"
    rows = [
        json.loads(line) for line in (args.output_dir / "generations.jsonl").read_text(encoding="utf8").splitlines()
    ]
    graded, accuracy = score_math(rows)
    with (args.output_dir / "graded.jsonl").open("x", encoding="utf8") as file:
        for row in graded:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")
    summary.update(
        grading_status="complete",
        accuracy=accuracy,
        grader_versions={name: importlib.metadata.version(name) for name in ("math-verify", "sympy")},
    )
    summary_path.write_text(json.dumps(summary, indent=2))
    print(json.dumps(accuracy), flush=True)


if __name__ == "__main__":
    main()
