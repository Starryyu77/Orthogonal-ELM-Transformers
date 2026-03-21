#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import json
import subprocess
from pathlib import Path
from typing import Any

from common import ensure_dir


FIELDNAMES = [
    "phase",
    "method",
    "seed",
    "dataset",
    "job_id",
    "gpu_type",
    "gpu_count",
    "cpu_count",
    "req_mem_gb",
    "node",
    "submit_utc",
    "start_utc",
    "end_utc",
    "wait_seconds",
    "elapsed_seconds",
    "gpu_hours",
    "steps",
    "tokens_seen",
    "tokens_per_second",
    "max_gpu_mem_gb",
    "state",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Update OELM V2 resource ledger from manifests and sacct")
    parser.add_argument(
        "--roots",
        nargs="+",
        required=True,
        help="Directories to scan for run_manifest.json files",
    )
    parser.add_argument("--output_csv", type=str, required=True)
    parser.add_argument("--skip_sacct", action="store_true")
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def parse_datetime(value: str | None) -> str | None:
    if value in {None, "", "Unknown"}:
        return None
    return value


def parse_sacct(job_id: str) -> dict[str, Any]:
    cmd = [
        "sacct",
        "-j",
        job_id,
        "--parsable2",
        "--noheader",
        "--format=JobIDRaw,State,Submit,Start,End,ElapsedRaw,NodeList,ReqTRES",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    lines = [line for line in result.stdout.splitlines() if line.strip()]
    primary = None
    for line in lines:
        columns = line.split("|")
        if columns[0] == job_id:
            primary = columns
            break
    if primary is None:
        raise RuntimeError(f"No sacct row found for job_id={job_id}")
    return {
        "job_id": primary[0],
        "state": primary[1],
        "submit_utc": parse_datetime(primary[2]),
        "start_utc": parse_datetime(primary[3]),
        "end_utc": parse_datetime(primary[4]),
        "elapsed_seconds": float(primary[5]) if primary[5] not in {"", "Unknown"} else None,
        "node": parse_datetime(primary[6]),
        "req_tres": primary[7],
    }


def parse_gpu_count(resource_usage: dict[str, Any], sacct_row: dict[str, Any] | None) -> int | None:
    if resource_usage.get("gpu_count") is not None:
        return int(resource_usage["gpu_count"])
    if sacct_row and "gres/gpu=" in sacct_row.get("req_tres", ""):
        try:
            return int(sacct_row["req_tres"].split("gres/gpu=", 1)[1].split(",", 1)[0])
        except (IndexError, ValueError):
            return None
    return None


def parse_wait_seconds(submit_utc: str | None, start_utc: str | None) -> float | None:
    if not submit_utc or not start_utc:
        return None
    from datetime import datetime

    submit_dt = datetime.fromisoformat(submit_utc.replace(" ", "T"))
    start_dt = datetime.fromisoformat(start_utc.replace(" ", "T"))
    return max((start_dt - submit_dt).total_seconds(), 0.0)


def scan_manifest_paths(roots: list[str]) -> list[Path]:
    manifests: list[Path] = []
    for root in roots:
        manifests.extend(Path(root).rglob("run_manifest.json"))
    return sorted(set(manifests))


def build_row(manifest: dict[str, Any], resource_usage: dict[str, Any], sacct_row: dict[str, Any] | None) -> dict[str, Any]:
    submit_utc = sacct_row.get("submit_utc") if sacct_row else None
    start_utc = sacct_row.get("start_utc") if sacct_row else resource_usage.get("start_utc")
    end_utc = sacct_row.get("end_utc") if sacct_row else resource_usage.get("end_utc")
    elapsed_seconds = sacct_row.get("elapsed_seconds") if sacct_row and sacct_row.get("elapsed_seconds") is not None else resource_usage.get("elapsed_seconds")
    gpu_count = parse_gpu_count(resource_usage, sacct_row)
    return {
        "phase": manifest.get("phase"),
        "method": manifest.get("method"),
        "seed": manifest.get("seed"),
        "dataset": manifest.get("dataset"),
        "job_id": manifest.get("slurm", {}).get("job_id") or resource_usage.get("job_id"),
        "gpu_type": resource_usage.get("gpu_type_detected") or resource_usage.get("gpu_type_requested"),
        "gpu_count": gpu_count,
        "cpu_count": resource_usage.get("cpu_count"),
        "req_mem_gb": resource_usage.get("req_mem_gb"),
        "node": sacct_row.get("node") if sacct_row else resource_usage.get("node"),
        "submit_utc": submit_utc,
        "start_utc": start_utc,
        "end_utc": end_utc,
        "wait_seconds": parse_wait_seconds(submit_utc, start_utc),
        "elapsed_seconds": elapsed_seconds,
        "gpu_hours": (elapsed_seconds * gpu_count / 3600.0) if elapsed_seconds is not None and gpu_count is not None else resource_usage.get("gpu_hours"),
        "steps": resource_usage.get("steps"),
        "tokens_seen": resource_usage.get("tokens_seen"),
        "tokens_per_second": resource_usage.get("tokens_per_second"),
        "max_gpu_mem_gb": resource_usage.get("max_gpu_mem_gb"),
        "state": sacct_row.get("state") if sacct_row else resource_usage.get("state"),
    }


def write_rows(output_csv: Path, rows: list[dict[str, Any]]) -> None:
    ensure_dir(output_csv.parent)
    with output_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> None:
    args = parse_args()
    rows: list[dict[str, Any]] = []
    for manifest_path in scan_manifest_paths(args.roots):
        resource_path = manifest_path.parent / "resource_usage.json"
        if not resource_path.exists():
            continue
        manifest = read_json(manifest_path)
        resource_usage = read_json(resource_path)
        sacct_row = None
        job_id = manifest.get("slurm", {}).get("job_id") or resource_usage.get("job_id")
        if job_id and not args.skip_sacct:
            try:
                sacct_row = parse_sacct(str(job_id))
            except (subprocess.CalledProcessError, FileNotFoundError, RuntimeError):
                sacct_row = None
        rows.append(build_row(manifest, resource_usage, sacct_row))

    write_rows(Path(args.output_csv), rows)


if __name__ == "__main__":
    main()
