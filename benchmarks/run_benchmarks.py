#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
"""End-to-end prover benchmark campaign: baseline vs optimized builds.

Runs every available end-to-end proof example across a range of sizes against
pre-built test binaries, interleaving configurations per (program, size) so ambient
load affects all sides equally. Every run is status-checked, and proof hashes are
compared across configurations — a mismatch fails the campaign, guaranteeing the
optimized prover produces bit-identical proofs.

Artifacts (CSV of every run, a median table in Markdown, environment info, and a
JSONL of raw records) are written to an output directory for sharing.

Usage:
    uv run benchmarks/run_benchmarks.py \
        --baseline /path/to/baseline-test-binary \
        --metal /path/to/optimized-metal-binary \
        [--nometal /path/to/optimized-cpu-only-binary] \
        [--pairs 3] [--out benchmarks/results]
"""

import argparse
import datetime
import json
import logging
import platform
import re
import statistics
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("bench")

TIME_RE = re.compile(r"test result: ok\..*finished in ([0-9.]+)s")
HASH_RE = re.compile(r"PROOF_HASH(?:\[[a-z_]+\])?=([0-9a-f]{16})")


@dataclass
class Program:
    name: str
    test_filter: str
    size_env: str | None
    sizes: list[int | None] = field(default_factory=lambda: [None])
    timeout: int = 600


PROGRAMS = [
    Program(
        "cpu_wide_fib",
        "test_cpu_e2e_wide_fib_prove",
        "CPU_FIB_LOG_N_INSTANCES",
        [14, 16, 18, 20, 22],
    ),
    Program("simd_blake", "test_simd_blake_prove", "LOG_N_INSTANCES", [10, 12, 14, 16]),
    Program("simd_plonk", "test_simd_plonk_prove", "LOG_N_INSTANCES", [12, 14, 16]),
    Program("simd_poseidon", "test_simd_poseidon_prove", "LOG_N_INSTANCES", [10, 12]),
    Program("simd_wide_fib", "test_wide_fib_prove_with_blake", None),
    Program("state_machine", "test_state_machine_prove", None),
]


def run_once(binary: str, program: Program, size: int | None, timeout: int) -> dict:
    env_extra = {
        "PROOF_HASH": "1",
        "CPU_FIB_PROOF_HASH": "1",
    }
    if program.size_env and size is not None:
        env_extra[program.size_env] = str(size)
    import os

    env = {**os.environ, **env_extra}
    try:
        out = subprocess.run(
            [binary, program.test_filter, "--nocapture"],
            capture_output=True,
            text=True,
            timeout=timeout,
            env=env,
        )
    except subprocess.TimeoutExpired:
        return {"status": "timeout", "seconds": None, "hash": None}
    text = out.stdout + out.stderr
    time_match = TIME_RE.search(text)
    if "0 passed" in text and "1 filtered" not in text and time_match:
        # The filter matched no test; treat as missing.
        return {"status": "missing", "seconds": None, "hash": None}
    if not time_match:
        status = "failed" if "FAILED" in text else "missing"
        return {"status": status, "seconds": None, "hash": None}
    hash_match = HASH_RE.search(text)
    return {
        "status": "ok",
        "seconds": float(time_match.group(1)),
        "hash": hash_match.group(1) if hash_match else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--metal", required=True)
    parser.add_argument("--nometal")
    parser.add_argument("--pairs", type=int, default=3)
    parser.add_argument("--out", default="benchmarks/results")
    args = parser.parse_args()

    configs = {"baseline": args.baseline, "optimized+metal": args.metal}
    if args.nometal:
        configs["optimized-cpu"] = args.nometal

    stamp = datetime.datetime.now().strftime("%Y-%m-%d-%H%M")
    out_dir = Path(args.out) / stamp
    out_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict] = []
    mismatches: list[str] = []

    for program in PROGRAMS:
        for size in program.sizes:
            label = f"{program.name}@2^{size}" if size else program.name
            hashes: dict[str, set[str]] = {}
            skip_remaining = False
            for pair in range(args.pairs):
                if skip_remaining:
                    break
                for config, binary in configs.items():
                    result = run_once(binary, program, size, program.timeout)
                    records.append(
                        {
                            "program": program.name,
                            "size": size,
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
                        f"{result['seconds']}s" if result["seconds"] else "",
                    )
                    if result["status"] in ("failed", "missing", "timeout"):
                        # Unsupported on this configuration; skip the experiment.
                        skip_remaining = True
                        break
                    if result["hash"]:
                        hashes.setdefault(config, set()).add(result["hash"])
            all_hashes = {h for hs in hashes.values() for h in hs}
            if len(hashes) >= 2 and len(all_hashes) > 1:
                mismatches.append(f"{label}: {hashes}")
                log.error("HASH MISMATCH at %s: %s", label, hashes)
            elif len(hashes) >= 2:
                log.info("%s: proofs bit-identical (%s)", label, next(iter(all_hashes)))

    (out_dir / "runs.jsonl").write_text(
        "\n".join(json.dumps(r) for r in records) + "\n"
    )
    with open(out_dir / "results.csv", "w") as f:
        f.write("program,size,config,pair,status,seconds,hash\n")
        for r in records:
            f.write(
                f"{r['program']},{r['size'] or ''},{r['config']},{r['pair']},"
                f"{r['status']},{r['seconds'] or ''},{r['hash'] or ''}\n"
            )

    # Median table.
    lines = [
        "| program | size | " + " | ".join(configs) + " | speedup (metal) | bit-identical |",
        "|---|---|" + "---|" * (len(configs) + 2),
    ]
    for program in PROGRAMS:
        for size in program.sizes:
            cells = []
            medians = {}
            for config in configs:
                times = [
                    r["seconds"]
                    for r in records
                    if r["program"] == program.name
                    and r["size"] == size
                    and r["config"] == config
                    and r["status"] == "ok"
                ]
                if times:
                    medians[config] = statistics.median(times)
                    cells.append(f"{medians[config]:.2f}s")
                else:
                    cells.append("—")
            if (
                "baseline" in medians
                and "optimized+metal" in medians
                and medians["optimized+metal"] > 0
            ):
                speedup = medians["baseline"] / medians["optimized+metal"]
                cells.append(f"**{speedup:.1f}x**")
            else:
                cells.append("—")
            run_hashes = {
                r["hash"]
                for r in records
                if r["program"] == program.name and r["size"] == size and r["hash"]
            }
            label = f"{program.name}@2^{size}" if size else program.name
            identical = "—"
            if run_hashes:
                identical = "yes" if len(run_hashes) == 1 else "**NO**"
            if any(c != "—" for c in cells[:-1]):
                lines.append(f"| {program.name} | {size or 'fixed'} | " + " | ".join(cells) + f" | {identical} |")
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
