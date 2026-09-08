# SPDX-License-Identifier: Apache-2.0
"""Download pinned SGLang UNO math datasets and write their integrity manifest."""

import argparse
import hashlib
import json
from pathlib import Path

from benchmarks.uno.sglang_math_data import DATASET_REVISIONS, get_benchmark, prepare_benchmark_data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--benchmarks", nargs="+", choices=["gsm8k", "math500", "aime25"], required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        parser.error("--output-dir must be new; preserve existing data and its manifest")
    args.output_dir.mkdir(parents=True)
    repositories = {"gsm8k": "openai/gsm8k", "math500": "HuggingFaceH4/MATH-500", "aime25": "math-ai/aime25"}
    manifest = []
    for name in dict.fromkeys(args.benchmarks):
        path = prepare_benchmark_data(name, output_dir=args.output_dir)
        repo = repositories[name]
        manifest.append(
            dict(
                benchmark=name,
                repo=repo,
                revision=DATASET_REVISIONS[repo],
                prepared_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                rows=get_benchmark(name).expected_rows,
            )
        )
        # Persist completed datasets even if a later network request fails.
        (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf8")
        print("DATASET_PREPARED", name, manifest[-1]["rows"], flush=True)


if __name__ == "__main__":
    main()
