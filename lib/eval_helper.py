import os
import re
import pandas as pd
import json
from typing import Optional, List, Dict, Any
import glob
import copy

from lib import utils
from lib.slurm import run_sbatch_job, get_log_file

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

REPORT_BENCHMARKS = {
    'mmmu/mllm_eval_accuracy': 'mmmu_v2',
    'mmmu/accuracy': 'mmmu_v1',
    'docvqa/anls_total_score': 'docvqa',
    'mathvista/accuracy': 'mathvista',
    'ai2d/accuracy': 'ai2d',
    'chartqa/accuracy': 'chartqa',
    'vqa/accuracy': 'vqa',
    'textvqa/accuracy': 'textvqa',
    'infographics_w_ocr/anls_total_score': 'infographics_w_ocr',
    'infographics/anls_total_score': 'infographics',
    'mmbench/overall': 'mmbench'
}


def get_report_benchmarks(all_benchmark_df: pd.DataFrame) -> pd.DataFrame:
    df = all_benchmark_df[list(REPORT_BENCHMARKS.keys())]
    df = df.rename(columns=REPORT_BENCHMARKS)
    df['textvqa'] = round(df['textvqa']/100, 4)
    return df


def run_eval_plan(
    eval_base_sbatch: str,
    aligner_parent_dir: str,
    eval_config_dir: str,
    checkpoint_dir: str,
    checkpoints: list[int],
    save_eval_jobs: str,
    benchmarks: Optional[List[str]] = None,
    rerun_if_exists: bool = False,
    update_if_exists: bool = False,
    print_cmd: bool = False,
    print_job_dict: bool = False
):

    job_dict = {}

    # if os.path.exists(save_eval_jobs) and not rerun_if_exists:
    #     raise RuntimeError(f"Found existing eval_jobs at: {save_eval_jobs}")
    
    if os.path.exists(save_eval_jobs):
        if update_if_exists:
            job_dict = utils.read_json(save_eval_jobs) 
        else:
            raise RuntimeError(f"Found existing eval_jobs at: {save_eval_jobs} and update_if_exists=False")

    if benchmarks is None:
        benchmarks = ALL_BENCHMARKS

    for benchmark in benchmarks:
        job_dict[benchmark] = {}

        for chk in checkpoints:
            params = {
                "aligner_parent_dir": aligner_parent_dir,
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
                positional_env_vars=list(params.values()),
                print_cmd=print_cmd
            )

            job_dict[benchmark][chk] = int(job_id)
            
    if print_job_dict:
        print(job_dict)

    os.makedirs(os.path.dirname(save_eval_jobs), exist_ok=True)
    with open(save_eval_jobs, 'w') as f:
        json.dump(job_dict, f, indent=4)

    print(f"eval jobs saved to: {save_eval_jobs}")


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


# def get_eval_scores(job_dict, output_csv=None) -> pd.DataFrame:
def get_eval_scores(checkpoint_dir: str, checkpoint_id: int, output_csv=None, verbose: bool=False) -> pd.DataFrame:
    results = {}
    
    job_dict_file = get_eval_jobs_record(checkpoint_dir, checkpoint_id)
    with open(job_dict_file, 'r') as f:
        job_dict = json.load(f)
    
    if verbose:    
        print(job_dict)
    
    for b in job_dict:
        results[b] = {}
        for chk in job_dict[b]:
            job_id = job_dict[b][chk]
            # log = f"/fsx_0/user/tranx/output/slurm_logs/output_{job_id}.txt"
            # print(log)
            log = get_log_file(int(job_id))

            res = extract_values(log)
            if res:
                if verbose:
                    print(f"Got result for {b} - {chk}: {res}")
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

    df = get_report_benchmarks(df)

    if output_csv is not None:
        df.to_csv(output_csv, index=False)

    return df


def get_eval_scores_all(checkpoint_dir: str, verbose: bool = False, sorted_by: Optional[str]=None) -> pd.DataFrame:
    checkpoints = get_checkpoints(checkpoint_dir)
    if verbose:
        print(checkpoints)
    df = pd.DataFrame()
    
    for c in checkpoints:
        try:
            df_c = get_eval_scores(checkpoint_dir, c, verbose=verbose)
            df = pd.concat([df, df_c])   
            if verbose:  
                print(c)   
        except Exception:
            pass
        
    if sorted_by is not None:
        df = df.sort_values(by=[sorted_by])
        
    return df


def get_checkpoints(output_dir: str) -> List[int]:
    """
    List all checkpoints in a given output folder
    """
    checkpoint_dirs = glob.glob(os.path.join(output_dir, '*/'))
    checkpoints = []

    for d in checkpoint_dirs:
        checkpoint_name = d.split('/')[-2]  # e.g. checkpoint-1800
        if checkpoint_name.startswith('checkpoint'):
            checkpoint_number = int(checkpoint_name.split('-')[-1])  # 800
            checkpoints.append(checkpoint_number)

    checkpoints = sorted(checkpoints)

    return checkpoints


def get_checkpoints_eval(output_dir: str) -> List[int]:
    """
    List all checkpoints with eval jobs
    """
    checkpoints = []
    for file in glob.glob(f"{output_dir}/evals/*.json"):
        c = file.replace(".json", "").split("-")[-1]
        checkpoints.append(int(c))
    
    checkpoints = sorted(checkpoints)
    
    return checkpoints


def get_eval_jobs_record(checkpoint_dir: str, checkpoint_id: int):
    return f"{checkpoint_dir}/evals/eval_jobs_checkpoint-{checkpoint_id}.json"
    
    
def run_eval_sweep(
    output_dir: str, 
    eval_sbatch: str, 
    eval_config_dir: str, 
    aligner_parent_dir:str, 
    print_cmd:bool = False, 
    min_checkpoint: int=0, 
    benchmarks: Optional[List[str]] = None,
    update_if_exists: bool = False
):
    """
    Start eval sweep on new checkpoints
    """
    
    checkpoints = get_checkpoints(output_dir)
    checkpoints_eval = get_checkpoints_eval(output_dir)
    
    if update_if_exists:
        new_checkpoints = checkpoints 
    else:
        new_checkpoints = [int(c) for c in checkpoints if c not in checkpoints_eval]
        
    new_checkpoints = sorted(new_checkpoints)
    
    print(f"New checkpoints: {new_checkpoints}")
    
    for c in new_checkpoints:
        if c < min_checkpoint:
            continue 
        
        # if c not in checkpoints_eval:
        print(f"Start eval for {output_dir}/checkpoint-{c}")
            
        run_eval_plan(
            eval_base_sbatch=eval_sbatch,
            aligner_parent_dir=aligner_parent_dir,
            eval_config_dir=eval_config_dir,
            checkpoint_dir=output_dir,
            checkpoints=[c],
            benchmarks=benchmarks,
            # save_eval_jobs=f"{output_dir}/evals/eval_jobs_checkpoint-{c}.json"
            save_eval_jobs=get_eval_jobs_record(output_dir, c),
            print_cmd=print_cmd,
            update_if_exists=update_if_exists
        )


def get_eval_config_overwrite(train_config: Dict[str, Any], eval_config: Dict[str, Any]) -> Dict[str, Any]:
    delete_keys = []
    for k in train_config["trainer_args"]:
        if k not in eval_config["eval_args"]:
            delete_keys.append(k)


    update_keys = {}
    for k, v in eval_config["eval_args"].items():
        if (k not in train_config["trainer_args"]) or (v != train_config["trainer_args"][k]):
            update_keys[k] = v
            
    overwrite = {
        "delete": delete_keys,
        "update": update_keys
    }
    
    return overwrite


def gen_eval_config(train_config, overwrite):
    
    eval_config = {
        "eval_args": copy.deepcopy(train_config["trainer_args"]),
        "fsdp_config": copy.deepcopy(train_config["fsdp_config"])
    }
    
    if "delete" in overwrite:
        for k in overwrite["delete"]:
            if k in eval_config["eval_args"]:
                eval_config["eval_args"].pop(k)
            
    eval_config["eval_args"].update(overwrite["update"])
    
    eval_config["eval_args"].update({
        "checkpoints_perception_tokenizer": train_config["trainer_args"]["checkpoints_perception_tokenizer"],
        "lora_checkpoint": "CHECKPOINT_PATH",
        "lora_tokenizer_checkpoint": "CHECKPOINT_PATH/adapter_tokenizer"
    })
    
    return eval_config