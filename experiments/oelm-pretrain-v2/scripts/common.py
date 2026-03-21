from __future__ import annotations

import csv
import json
import logging
import os
import random
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import torch


UTC = timezone.utc
SGT = ZoneInfo("Asia/Singapore")


def ensure_dir(path: str | Path) -> Path:
    resolved = Path(path)
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def utc_now() -> datetime:
    return datetime.now(tz=UTC)


def sgt_now() -> datetime:
    return datetime.now(tz=SGT)


def iso_timestamp(value: datetime | None = None) -> str:
    target = value or utc_now()
    return target.isoformat(timespec="seconds")


def timestamp_slug(value: datetime | None = None) -> str:
    target = value or utc_now()
    return target.astimezone(UTC).strftime("%Y%m%d_%H%M%S")


def write_json(path: str | Path, payload: Any) -> Path:
    resolved = Path(path)
    ensure_dir(resolved.parent)
    resolved.write_text(json.dumps(payload, indent=2, sort_keys=True))
    return resolved


def read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text())


def setup_logging(output_dir: str | Path, prefix: str) -> tuple[logging.Logger, Path]:
    ensure_dir(output_dir)
    log_path = Path(output_dir) / f"{prefix}_{timestamp_slug()}.log"
    logger_name = f"{prefix}_{log_path.stem}"
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    logger.propagate = False

    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    return logger, log_path


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def slurm_context() -> dict[str, Any]:
    job_gres = os.environ.get("SLURM_JOB_GRES", "")
    return {
        "job_id": os.environ.get("SLURM_JOB_ID"),
        "job_name": os.environ.get("SLURM_JOB_NAME"),
        "node_name": os.environ.get("SLURMD_NODENAME") or os.environ.get("HOSTNAME"),
        "submit_host": os.environ.get("SLURM_SUBMIT_HOST"),
        "partition": os.environ.get("SLURM_JOB_PARTITION"),
        "cpus_per_task": getenv_int("SLURM_CPUS_PER_TASK"),
        "req_mem_mb": parse_slurm_mem(os.environ.get("SLURM_MEM_PER_NODE")),
        "gpus_on_node": getenv_int("SLURM_GPUS_ON_NODE"),
        "job_gres": job_gres,
        "gpu_type_requested": parse_gpu_type(job_gres),
    }


def getenv_int(name: str) -> int | None:
    value = os.environ.get(name)
    if value is None:
        return None
    try:
        return int(value)
    except ValueError:
        return None


def parse_slurm_mem(value: str | None) -> int | None:
    if not value:
        return None
    normalized = value.strip().upper()
    if normalized.endswith("G"):
        return int(normalized[:-1]) * 1024
    if normalized.endswith("M"):
        return int(normalized[:-1])
    try:
        return int(normalized)
    except ValueError:
        return None


def parse_gpu_type(job_gres: str) -> str | None:
    if "gpu:" not in job_gres:
        return None
    try:
        part = job_gres.split("gpu:", 1)[1]
        return part.split(":", 1)[0]
    except IndexError:
        return None


def detect_gpu_type() -> str | None:
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            check=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None

    names = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    if not names:
        return None
    if len(set(names)) == 1:
        return names[0]
    return ",".join(sorted(set(names)))


def build_run_name(explicit_run_name: str | None = None) -> str:
    if explicit_run_name:
        return explicit_run_name
    job_id = os.environ.get("SLURM_JOB_ID")
    if job_id:
        return f"job-{job_id}"
    return timestamp_slug()


def build_run_dir(
    root: str | Path,
    phase: str,
    method: str,
    seed: int,
    run_name: str,
) -> Path:
    return ensure_dir(
        Path(root)
        / f"phase={phase}"
        / f"method={method}"
        / f"seed={seed}"
        / f"run={run_name}"
    )


def base_manifest(
    *,
    phase: str,
    method: str,
    seed: int,
    dataset: str,
    output_dir: str | Path,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    manifest = {
        "phase": phase,
        "method": method,
        "seed": seed,
        "dataset": dataset,
        "output_dir": str(Path(output_dir).resolve()),
        "created_at_utc": iso_timestamp(),
        "created_at_sgt": sgt_now().isoformat(timespec="seconds"),
        "slurm": slurm_context(),
    }
    if extra:
        manifest.update(extra)
    return manifest


def base_resource_usage(
    *,
    phase: str,
    method: str,
    seed: int,
    dataset: str,
    start_time_utc: datetime,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload = {
        "phase": phase,
        "method": method,
        "seed": seed,
        "dataset": dataset,
        "job_id": os.environ.get("SLURM_JOB_ID"),
        "partition": os.environ.get("SLURM_JOB_PARTITION"),
        "gpu_type_requested": parse_gpu_type(os.environ.get("SLURM_JOB_GRES", "")),
        "gpu_type_detected": detect_gpu_type(),
        "gpu_count": getenv_int("SLURM_GPUS_ON_NODE") or (1 if torch.cuda.is_available() else 0),
        "cpu_count": getenv_int("SLURM_CPUS_PER_TASK"),
        "req_mem_gb": to_gb(parse_slurm_mem(os.environ.get("SLURM_MEM_PER_NODE"))),
        "node": os.environ.get("SLURMD_NODENAME") or os.environ.get("HOSTNAME"),
        "start_utc": iso_timestamp(start_time_utc),
        "start_sgt": start_time_utc.astimezone(SGT).isoformat(timespec="seconds"),
    }
    if extra:
        payload.update(extra)
    return payload


def finalize_resource_usage(
    usage: dict[str, Any],
    *,
    end_time_utc: datetime,
    state: str,
    steps: int,
    tokens_seen: int,
    peak_gpu_mem_gb: float,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    elapsed_seconds = max((end_time_utc - datetime.fromisoformat(usage["start_utc"])).total_seconds(), 0.0)
    usage.update(
        {
            "end_utc": iso_timestamp(end_time_utc),
            "end_sgt": end_time_utc.astimezone(SGT).isoformat(timespec="seconds"),
            "elapsed_seconds": elapsed_seconds,
            "gpu_hours": elapsed_seconds * float(usage.get("gpu_count") or 0) / 3600.0,
            "steps": steps,
            "tokens_seen": tokens_seen,
            "tokens_per_second": (tokens_seen / elapsed_seconds) if elapsed_seconds > 0 else 0.0,
            "max_gpu_mem_gb": peak_gpu_mem_gb,
            "state": state,
        }
    )
    if extra:
        usage.update(extra)
    return usage


def to_gb(mem_mb: int | None) -> float | None:
    if mem_mb is None:
        return None
    return round(mem_mb / 1024.0, 3)


def peak_gpu_memory_gb() -> float:
    if not torch.cuda.is_available():
        return 0.0
    peak_bytes = torch.cuda.max_memory_allocated()
    return round(peak_bytes / (1024**3), 4)


def load_config(path: str | Path) -> dict[str, Any]:
    return read_json(path)


def append_csv_rows(path: str | Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    resolved = Path(path)
    ensure_dir(resolved.parent)
    write_header = not resolved.exists()
    with resolved.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        for row in rows:
            writer.writerow(row)
