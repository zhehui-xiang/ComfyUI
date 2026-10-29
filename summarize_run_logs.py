import argparse
import csv
import re
import sys
from pathlib import Path


WORKLOAD_RE = re.compile(
    r"WORKLOAD_COMPLETE.*?\brun=(\d+).*?:\s*([0-9]+(?:\.[0-9]+)?)\s*sec"
)
SPEED_RE = re.compile(r"[0-9]+(?:\.[0-9]+)?s/it")
PROGRESS_RE = re.compile(r"[0-9]+(?:\.[0-9]+)?/[0-9]+(?:\.[0-9]+)?")
ANSI_RE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")


def extract_runs(log_path: Path) -> list[dict[str, str]]:
    runs = []
    last_progress = ""
    last_speed = ""

    with log_path.open(encoding="utf-8", errors="replace") as log_file:
        for line in log_file:
            clean_line = ANSI_RE.sub("", line).strip()
            speed_match = SPEED_RE.search(clean_line)
            if speed_match:
                progress_matches = list(PROGRESS_RE.finditer(clean_line[:speed_match.start()]))
                last_progress = progress_matches[-1].group() if progress_matches else ""
                last_speed = speed_match.group().removesuffix("s/it")

            workload_match = WORKLOAD_RE.search(clean_line)
            if workload_match:
                runs.append(
                    {
                        "file": log_path.name,
                        "run": workload_match.group(1),
                        "progress": last_progress,
                        "speed": last_speed,
                        "workload_duration_sec": workload_match.group(2),
                    }
                )
                last_progress = ""
                last_speed = ""

    return runs


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Summarize the last tqdm s/it value and WORKLOAD_COMPLETE duration per run."
    )
    parser.add_argument(
        "--logs-dir",
        type=Path,
        default=Path.home() / "xzh" / "ComfyUI",
        help="Directory containing log files (default: ~/xzh/ComfyUI).",
    )
    parser.add_argument(
        "--pattern",
        default="*opt*.log",
        help="Filename glob (default: *opt*.log).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="CSV output path (default: <logs-dir>/run_summary.csv).",
    )
    args = parser.parse_args()

    log_paths = sorted(args.logs_dir.glob(args.pattern))
    if not log_paths:
        print(f"No files matched {args.pattern!r} in {args.logs_dir}", file=sys.stderr)
        return 1

    rows = []
    for log_path in log_paths:
        runs = {row["run"]: row for row in extract_runs(log_path)}
        summary = {"file": log_path.name}
        for run_number in ("1", "2"):
            run = runs.get(run_number, {})
            summary.update(
                {
                    f"run{run_number}_progress": run.get("progress", ""),
                    f"run{run_number}_speed": run.get("speed", ""),
                    f"run{run_number}_workload_duration_sec": run.get(
                        "workload_duration_sec", ""
                    ),
                }
            )
        rows.append(summary)

    output_path = args.output or args.logs_dir / "run_summary.csv"
    with output_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=(
                "file",
                "run1_progress",
                "run1_speed",
                "run1_workload_duration_sec",
                "run2_progress",
                "run2_speed",
                "run2_workload_duration_sec",
            ),
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"Processed {len(log_paths)} log file(s), wrote {len(rows)} row(s).")
    print(f"CSV: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())