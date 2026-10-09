#!/usr/bin/env python3
"""Execute a ComfyUI API-format workflow without starting the web server."""
print("start of script",flush=True)
import argparse
import asyncio
import copy
import importlib.util
import json
import logging
import os
import sys
import time
import uuid


def parse_runner_args():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--workflow", required=True, help="Workflow saved by ComfyUI as API Format JSON")
    parser.add_argument("--runs", type=int, default=2)
    parser.add_argument("--seed-step", type=int, default=1)
    runner_args, comfy_args = parser.parse_known_args()
    if runner_args.runs < 1:
        parser.error("--runs must be at least 1")
    sys.argv = [sys.argv[0], *comfy_args]
    return runner_args


RUNNER_ARGS = parse_runner_args()

import comfy.options

comfy.options.enable_args_parsing()

from comfy.cli_args import args
import folder_paths
import utils.extra_config


def configure_paths():
    if args.extra_model_paths_config:
        for config_paths in args.extra_model_paths_config:
            for config_path in config_paths:
                utils.extra_config.load_extra_path_config(config_path)
    if args.output_directory:
        folder_paths.set_output_directory(os.path.abspath(args.output_directory))
    if args.input_directory:
        folder_paths.set_input_directory(os.path.abspath(args.input_directory))
    if args.user_directory:
        folder_paths.set_user_directory(os.path.abspath(args.user_directory))
    os.makedirs(folder_paths.get_output_directory(), exist_ok=True)


def run_prestartup_scripts():
    for custom_node_path in folder_paths.get_folder_paths("custom_nodes"):
        for module_name in os.listdir(custom_node_path):
            script_path = os.path.join(custom_node_path, module_name, "prestartup_script.py")
            if not os.path.isfile(script_path):
                continue
            spec = importlib.util.spec_from_file_location(script_path, script_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)


class HeadlessServer:
    def __init__(self):
        self.client_id = "headless-benchmark"
        self.timed_node_id = None
        self.last_event_time = None

    def send_sync(self, event, data, sid=None):
        if event != "executing" or data.get("node") is None:
            return
        now = time.perf_counter()
        if self.last_event_time is not None:
            logging.info("NODE_TIMING node=%s seconds=%.3f", self.timed_node_id, now - self.last_event_time)
        self.timed_node_id = data["node"]
        self.last_event_time = now


def load_api_prompt(path):
    with open(path, encoding="utf-8") as workflow_file:
        prompt = json.load(workflow_file)
    if not isinstance(prompt, dict) or not all("class_type" in node for node in prompt.values()):
        raise ValueError(
            "Workflow must be ComfyUI API Format JSON. In the frontend, use Save (API Format); "
            "the normal workflow JSON is a canvas document and cannot be executed by PromptExecutor."
        )
    return prompt


def update_seeds(prompt, run_number):
    if run_number == 1 or RUNNER_ARGS.seed_step == 0:
        return
    for node in prompt.values():
        inputs = node.get("inputs", {})
        if isinstance(inputs.get("noise_seed"), int):
            inputs["noise_seed"] += RUNNER_ARGS.seed_step * (run_number - 1)


async def execute_prompt(executor, prompt, run_number):
    from execution import validate_prompt

    prompt_id = str(uuid.uuid4())
    valid, error, outputs, _ = await validate_prompt(prompt_id, prompt, None)
    if not valid:
        raise RuntimeError(f"Workflow validation failed: {error['message']}: {error['details']}")
    executor.server.timed_node_id = None
    executor.server.last_event_time = time.perf_counter()
    start = time.perf_counter()
    await executor.execute_async(prompt, prompt_id, {"client_id": executor.server.client_id}, outputs)
    if executor.server.timed_node_id is not None:
        logging.info(
            "NODE_TIMING last node=%s seconds=%.3f",
            executor.server.timed_node_id,
            time.perf_counter() - executor.server.last_event_time,
        )
    logging.info(
        "WORKLOAD_COMPLETE run=%d cached_models=%s success=%s seconds=%.3f",
        run_number,
        run_number > 1,
        executor.success,
        time.perf_counter() - start,
    )
    if not executor.success:
        raise RuntimeError("Workflow execution failed; inspect execution_error messages above")


async def main():
    print("begin of main()", flush=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    configure_paths()
    run_prestartup_scripts()
    from execution import CacheType, PromptExecutor
    import nodes

    await nodes.init_extra_nodes(init_custom_nodes=True, init_api_nodes=not getattr(args, "disable_partner_nodes", False))
    prompt = load_api_prompt(os.path.abspath(RUNNER_ARGS.workflow))
    executor = PromptExecutor(
        HeadlessServer(),
        cache_type=CacheType.CLASSIC,
        cache_args={"lru": 0, "ram": 0, "ram_inactive": 0},
    )
    for run_number in range(1, RUNNER_ARGS.runs + 1):
        run_prompt = copy.deepcopy(prompt)
        update_seeds(run_prompt, run_number)
        await execute_prompt(executor, run_prompt, run_number)


if __name__ == "__main__":
    print("start of __main__",flush=True)
    asyncio.run(main())