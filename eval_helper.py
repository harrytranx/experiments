import os
import re
import pandas as pd
import json
from typing import Optional, List

import utils
from slurm import run_sbatch_job

ALL_BENCHMARKS = [
    "mmmu",
    "docvqa",
    "mathvista",
    "ai2d",
    "chartqa",
    "vqa",
    "textvqa",
    "infographics_w_ocr",
    "infographics",
    "mmbench"
]


def run_eval_plan(
    eval_base_sbatch: str,
    eval_plan: str,
    eval_config_dir: str,
    checkpoint_dir: str,
    checkpoints: list[int],
    benchmarks: Optional[List[str]] = None
):

    job_dict = {}

    job_dict_json = f'job_dict_{eval_plan}.json'
    assert not os.path.exists(job_dict_json)

    if benchmarks is None:
        benchmarks = ALL_BENCHMARKS

    for benchmark in benchmarks:
        job_dict[benchmark] = {}

        for chk in checkpoints:
            params = {
                "eval_plan": eval_plan,
                "json_config": f"{eval_config_dir}/eval_{benchmark}.json",
                "checkpoint_dir": checkpoint_dir,
                "benchmark_name": benchmark,
                "checkpoint_id": str(chk)
            }

            assert os.path.exists(params["json_config"])
            assert os.path.exists(
                f"{params['checkpoint_dir']}/checkpoint-{chk}")

            job_id = run_sbatch_job(
                sbatch_base_script=eval_base_sbatch,
                sbatch_overwrite={
                    "job-name": f"eval_{benchmark}"
                },
                positional_env_vars=list(params.values())
            )

            job_dict[benchmark][chk] = int(job_id)

    with open(job_dict_json, 'w') as f:
        json.dump(job_dict, f, indent=4)

    print(f"job_dict saved to {job_dict_json}")

    return job_dict


def extract_values(filename):
    if not os.path.exists(filename):
        return None

    # Define the pattern to match the line and capture the required parts
    pattern = r"Writing (\S+) with value ([\d\.]+) to TensorBoard"

    # List to hold the extracted values
    extracted_values = []

    # Open the file and read line by line
    with open(filename, 'r', encoding='utf-8', errors='replace') as file:
        # with open(filename, 'r') as file:
        for line in file:
            # Search for the pattern in each line
            match = re.search(pattern, line)
            if match:
                # Extract the path and the value
                path = match.group(1)
                value = float(match.group(2))
                extracted_values.append((path, value))

    # if len(extracted_values) == 0:
    #     return None

    return extracted_values


def get_eval_scores(job_dict, output_csv=None):
    results = {}
    for b in job_dict:
        results[b] = {}
        for chk in job_dict[b]:
            job_id = job_dict[b][chk]
            log = f"/fsx_0/user/tranx/output/slurm_logs/output_{job_id}.txt"
            # print(log)

            res = extract_values(log)
            if res:
                print(f"Got result for {b} - {chk}")
                results[b][chk] = res

    scores = {
        "mmmu": ["accuracy", "mllm_eval_accuracy"],
        "docvqa": ["anls_total_score", "mllm_evaluation_anls_score"],
        "mathvista": ["accuracy"],
        "ai2d": ["accuracy"],
        "chartqa": ["accuracy"],
        "vqa": ["accuracy", "mllm_evaluation_accuracy"],
        "textvqa": ["accuracy", "mllm_eval_accuracy"],
        "infographics_w_ocr": ["anls_total_score", "mllm_evaluation_anls_score"],
        "infographics": ["anls_total_score", "mllm_evaluation_anls_score"],
        "mmbench": ["overall"]
    }

    component_scores = []
    for k, v in scores.items():
        for vi in v:
            component_scores.append(f"{k}/{vi}")

    df = pd.DataFrame(columns=component_scores)

    for b in results:
        for chk in results[b]:
            res = results[b][chk]
            # print(b, chk, results[b][chk])
            for x in res:
                component, val = x[0], x[1]
                if component in component_scores:
                    df.loc[chk, component] = round(val, 4)

    if output_csv is not None:
        df.to_csv(output_csv, index=False)

    return df
