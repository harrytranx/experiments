"""
Usage:
python eval_runner.py -c eval_watcher_config.json
"""
import sys 
lib_path = '/fsx_0/user/tranx/rsync/llm_mm_aligner/experiments/aws'
if lib_path not in sys.path:
    sys.path.append(lib_path)
    
from pprint import pprint
from typing import Any
import time 
import json
import launch_evals as eval_helper
import argparse

def run_evals(
    configs_path: str,
    watch_path: str,
    max_jobs: int,
    conda_env: str | None = None,
    prefix_name: str | None = None,
    start_step: int | None = None,
    stop_step: int | None = None,
    every_steps: int | None = None,
    base_eval_output_dir: str = "evals",
    overrides: list[list[tuple[str, Any]]] | None = None,
    rerun_if_exists: bool = False
):
    num_jobs = 0
    
    available_steps = eval_helper._get_available_checkpoints(watch_path)
    
    for step in available_steps:
        if (start_step is not None and step < start_step) or (
            every_steps is not None and step % every_steps != 0
        ):
            continue
        
        new_jobs = eval_helper.run_eval_steps(
            configs_path=configs_path,
            watch_path=watch_path,
            max_num_jobs=max_jobs - num_jobs,
            conda_env=conda_env,
            prefix_name=prefix_name,
            steps=[step],
            base_eval_output_dir=base_eval_output_dir,
            overrides=overrides,
            rerun_if_exists=rerun_if_exists,
        )
        
        num_jobs += new_jobs
        if num_jobs == max_jobs:
            # will not launch more job in this iteration
            break

        if stop_step is not None and step == stop_step:
            break
        
    return num_jobs

def main(config_file: str):
    while True:
        # re-read latest config
        # with open('eval_watcher_config.json', 'r') as file:
        print(f"Reading eval watcher config from: {config_file}")
        with open(config_file, 'r') as file:
            global_config = json.load(file)

        watch_jobs = global_config.pop("watch_jobs")
        wait_seconds = global_config.pop("wait_seconds")
        read_format = global_config.pop("read_format")
        
        for job in watch_jobs:
            if "ignore" in job:
                if job["ignore"]:
                    continue 
                else:
                    job.pop("ignore")
                
            job_config = global_config.copy()
            
            # update with local config
            job_config.update(job)
            wandb_project = job_config.pop("wandb_project")
            
            print("="*20)
            pprint(job_config)
            print(f"Scanning for new checkpoints in {job_config['watch_path']}")
            try:
                num_jobs = run_evals(**job_config)
                print(f"Launched {num_jobs} jobs in {job_config['watch_path']}")
                
                # read results and publish to wandb
                eval_helper.read_eval_results(
                    watch_path=job_config['watch_path'],
                    base_eval_output_dir=job_config.get('base_eval_output_dir', "evals"),
                    read_format=read_format,
                    wandb_project=wandb_project
                )
            except Exception as e:
                print(e)
            
        time.sleep(wait_seconds)

        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Eval Watcher Config")
    parser.add_argument("-c", "--config", help="Path to config file", required=True)
    args = parser.parse_args()
    main(args.config)