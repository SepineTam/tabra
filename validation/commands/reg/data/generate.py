#!/usr/bin/env python3
"""Generate the deterministic cross-engine fixture for the reg pilot."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
from pathlib import Path


SEED = 20260722
N_ROWS = 120


def standard_normal(random_generator: random.Random) -> float:
    """Draw one standard normal using a frozen Box-Muller transform."""
    first_uniform = random_generator.random()
    second_uniform = random_generator.random()
    return math.sqrt(-2.0 * math.log(first_uniform)) * math.cos(
        2.0 * math.pi * second_uniform
    )


def generate_rows() -> list[dict[str, float | int]]:
    """Return data from MT uniform draws and a frozen normal transform."""
    random_generator = random.Random(SEED)
    rows: list[dict[str, float | int]] = []
    for row_id in range(1, N_ROWS + 1):
        x1 = standard_normal(random_generator)
        x2 = 0.35 * x1 + 0.9 * standard_normal(random_generator)
        scale = 0.35 + 0.08 * abs(x1)
        noise = scale * standard_normal(random_generator)
        y = 1.75 + 2.4 * x1 - 0.85 * x2 + noise
        rows.append({"row_id": row_id, "y": y, "x1": x1, "x2": x2})
    return rows


def write_fixture(output_path: Path) -> str:
    """Write stable CSV bytes, a SHA-256 sidecar, and generation metadata."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.DictWriter(
            output_file,
            fieldnames=["row_id", "y", "x1", "x2"],
            lineterminator="\n",
        )
        writer.writeheader()
        for row in generate_rows():
            writer.writerow(
                {
                    "row_id": row["row_id"],
                    "y": format(float(row["y"]), ".17g"),
                    "x1": format(float(row["x1"]), ".17g"),
                    "x2": format(float(row["x2"]), ".17g"),
                }
            )

    digest = hashlib.sha256(output_path.read_bytes()).hexdigest()
    output_path.with_suffix(output_path.suffix + ".sha256").write_text(
        f"{digest}  {output_path.name}\n",
        encoding="ascii",
    )
    metadata = {
        "generator": "validation/commands/reg/data/generate.py",
        "random_generator": "random.Random.random (Mersenne Twister)",
        "normal_transform": "Box-Muller v1",
        "seed": SEED,
        "rows": N_ROWS,
        "sha256": digest,
        "csv_float_format": ".17g",
        "finite_values": all(
            math.isfinite(float(value))
            for row in generate_rows()
            for key, value in row.items()
            if key != "row_id"
        ),
    }
    output_path.with_suffix(output_path.suffix + ".metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return digest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).with_name("reg_fixture.csv"),
    )
    arguments = parser.parse_args()
    print(write_fixture(arguments.output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
