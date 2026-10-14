import argparse
import json
from pathlib import Path

script_dir = Path(__file__).resolve().parent
parser = argparse.ArgumentParser(
    prog="set_raylight_workload.py",
    description="Update duration, GPU and ulysses_degree in the JSON file in place.",
)
parser.add_argument("--duration", type=int, default=5, help="Duration in seconds")
parser.add_argument("--gpu", type=int, default=1, help="GPU count and ulysses_degree")
parser.add_argument(
    "--json_file",
    type=Path,
    default=script_dir / "video_minimax_h3_t2v_raylight_api.json",
    help="JSON file to modify (default: workflow beside this script)",
)
args = parser.parse_args()

workflow = json.loads(args.json_file.read_text(encoding="utf-8"))
workflow["140:133"]["inputs"]["value"] = args.duration
try:
    workflow["140:141"]["inputs"]["GPU"] = args.gpu
    workflow["140:141"]["inputs"]["ulysses_degree"] = args.gpu
except KeyError as e:
    print(f"Warning: Missing key in JSON workflow: {e}")

output = json.dumps(workflow, ensure_ascii=False, indent=2) + "\n"
args.json_file.write_text(output, encoding="utf-8")
print(f"Updated {args.json_file}: duration={args.duration}s, GPU={args.gpu}, ulysses_degree={args.gpu}")