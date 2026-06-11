#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
"""Prover throughput by AIR shape: Mcells/s over a (rows x cols) grid.

Sweeps the parametric `test_air_shape_prove` example over a grid of trace
shapes in two modes — constrained (one degree-2 constraint per derived column,
a real-AIR workload) and unconstrained (commitment/FRI envelope only) — against
pre-built test binaries. The timed section is the protocol itself (twiddles
through proof), excluding trace generation, as reported by the test's
`AIR_SHAPE` line.

Configurations are interleaved per experiment, every run is status-checked, and
proof hashes are compared across configurations: any mismatch fails the
campaign, so a passing run certifies bit-identical proofs on every shape.

Usage:
    uv run benchmarks/run_air_shapes.py \
        --baseline /path/to/baseline-test-binary \
        --metal /path/to/optimized-metal-binary \
        [--nometal /path/to/optimized-cpu-only-binary] \
        [--pairs 2] [--out benchmarks/results]
"""

import argparse
import datetime
import json
import logging
import os
import platform
import re
import statistics
import subprocess
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("air-shapes")

SHAPE_RE = re.compile(
    r"AIR_SHAPE log_rows=(\d+) cols=(\d+) constrained=(\d) prove_s=([0-9.]+)"
)
HASH_RE = re.compile(r"PROOF_HASH\[air_shape\]=([0-9a-f]{16})")

# (log_rows, cols) grid; both constraint modes run for every shape.
SHAPES = [
    (14, 16),
    (14, 64),
    (14, 256),
    (16, 16),
    (16, 64),
    (16, 256),
    (18, 16),
    (18, 64),
    (18, 256),
    (20, 16),
    (20, 64),
    (20, 256),
    (22, 16),
    (22, 64),
]
MODES = [1, 0]  # constrained, unconstrained


def run_once(binary: str, log_rows: int, cols: int, constrained: int) -> dict:
    env = {
        **os.environ,
        "PROOF_HASH": "1",
        "LOG_N_ROWS": str(log_rows),
        "N_COLS": str(cols),
        "CONSTRAINED": str(constrained),
    }
    try:
        out = subprocess.run(
            [binary, "test_air_shape_prove", "--nocapture"],
            capture_output=True,
            text=True,
            timeout=600,
            env=env,
        )
    except subprocess.TimeoutExpired:
        return {"status": "timeout", "prove_s": None, "hash": None}
    text = out.stdout + out.stderr
    shape_match = SHAPE_RE.search(text)
    hash_match = HASH_RE.search(text)
    if out.returncode != 0 or not shape_match or "FAILED" in text:
        return {"status": "failed", "prove_s": None, "hash": None}
    return {
        "status": "ok",
        "prove_s": float(shape_match.group(4)),
        "hash": hash_match.group(1) if hash_match else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--metal", required=True)
    parser.add_argument("--nometal")
    parser.add_argument("--pairs", type=int, default=2)
    parser.add_argument("--out", default="benchmarks/results")
    args = parser.parse_args()

    configs = {"baseline": args.baseline, "optimized+metal": args.metal}
    if args.nometal:
        configs["optimized-cpu"] = args.nometal

    stamp = datetime.datetime.now().strftime("%Y-%m-%d-%H%M")
    out_dir = Path(args.out) / f"{stamp}-air-shapes"
    out_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict] = []
    mismatches: list[str] = []

    for constrained in MODES:
        for log_rows, cols in SHAPES:
            label = f"2^{log_rows}x{cols} constrained={constrained}"
            hashes: dict[str, set[str]] = {}
            skip = False
            for pair in range(args.pairs):
                if skip:
                    break
                for config, binary in configs.items():
                    result = run_once(binary, log_rows, cols, constrained)
                    records.append(
                        {
                            "log_rows": log_rows,
                            "cols": cols,
                            "constrained": constrained,
                            "config": config,
                            "pair": pair,
                            **result,
                        }
                    )
                    log.info(
                        "%s [%s] pair %d: %s %s",
                        label,
                        config,
                        pair,
                        result["status"],
                        f"{result['prove_s']}s" if result["prove_s"] else "",
                    )
                    if result["status"] != "ok":
                        skip = True
                        break
                    if result["hash"]:
                        hashes.setdefault(config, set()).add(result["hash"])
            all_hashes = {h for hs in hashes.values() for h in hs}
            if len(hashes) >= 2 and len(all_hashes) > 1:
                mismatches.append(f"{label}: {hashes}")
                log.error("HASH MISMATCH at %s: %s", label, hashes)

    (out_dir / "runs.jsonl").write_text(
        "\n".join(json.dumps(r) for r in records) + "\n"
    )
    with open(out_dir / "results.csv", "w") as f:
        f.write("log_rows,cols,constrained,config,pair,status,prove_s,hash\n")
        for r in records:
            f.write(
                f"{r['log_rows']},{r['cols']},{r['constrained']},{r['config']},"
                f"{r['pair']},{r['status']},{r['prove_s'] or ''},{r['hash'] or ''}\n"
            )

    # One table per mode: median prove time and Mcells/s per configuration.
    lines = []
    for constrained in MODES:
        mode = "constrained (degree-2 per column)" if constrained else "unconstrained (commit/FRI only)"
        lines.append(f"\n## {mode}\n")
        header = ["rows", "cols", "Mcells"]
        for config in configs:
            header += [f"{config} (s)", f"{config} (Mc/s)"]
        header += ["speedup (metal)", "bit-identical"]
        lines.append("| " + " | ".join(header) + " |")
        lines.append("|" + "---|" * len(header))
        for log_rows, cols in SHAPES:
            cells = (1 << log_rows) * cols
            row = [f"2^{log_rows}", str(cols), f"{cells / 1e6:.1f}"]
            medians = {}
            for config in configs:
                times = [
                    r["prove_s"]
                    for r in records
                    if r["log_rows"] == log_rows
                    and r["cols"] == cols
                    and r["constrained"] == constrained
                    and r["config"] == config
                    and r["status"] == "ok"
                ]
                if times:
                    medians[config] = statistics.median(times)
                    row += [
                        f"{medians[config]:.3f}",
                        f"{cells / medians[config] / 1e6:.0f}",
                    ]
                else:
                    row += ["—", "—"]
            if (
                "baseline" in medians
                and "optimized+metal" in medians
                and medians["optimized+metal"] > 0
            ):
                row.append(f"**{medians['baseline'] / medians['optimized+metal']:.1f}x**")
            else:
                row.append("—")
            run_hashes = {
                r["hash"]
                for r in records
                if r["log_rows"] == log_rows
                and r["cols"] == cols
                and r["constrained"] == constrained
                and r["hash"]
            }
            row.append("yes" if len(run_hashes) == 1 else ("**NO**" if run_hashes else "—"))
            lines.append("| " + " | ".join(row) + " |")
    (out_dir / "table.md").write_text("\n".join(lines) + "\n")

    env_info = [
        f"date: {datetime.datetime.now().isoformat()}",
        f"platform: {platform.platform()}",
        f"machine: {platform.machine()}",
    ]
    for name, path in configs.items():
        env_info.append(f"binary[{name}]: {path}")
    try:
        chip = subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True
        ).stdout.strip()
        env_info.append(f"chip: {chip}")
    except OSError:
        pass
    (out_dir / "env.txt").write_text("\n".join(env_info) + "\n")

    log.info("artifacts in %s", out_dir)
    if mismatches:
        log.error("BIT-IDENTITY FAILURES: %s", mismatches)
        raise SystemExit(1)
    log.info("all cross-config proof hashes identical")


if __name__ == "__main__":
    main()
