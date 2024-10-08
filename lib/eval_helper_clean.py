import os
import re
import pandas as pd
import numpy as np 
import json
from typing import Optional, List, Dict, Any
import glob
import copy
import subprocess
from collections import OrderedDict

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

READ_BENCHMARK_SCORES = {
    "mmmu": ["accuracy", "mllm_eval_accuracy"],
    "docvqa": ["anls_total_score", "mllm_evaluation_anls_score", "mmllm_fixed_anls_score"],
    "mathvista": ["accuracy"],
    "ai2d": ["accuracy"],
    "chartqa": ["accuracy"],
    "vqa": ["accuracy", "mllm_evaluation_accuracy"],
    "textvqa": ["accuracy", "mllm_eval_accuracy"],
    "infographics_w_ocr": ["anls_total_score", "mllm_evaluation_anls_score"],
    "infographics": ["anls_total_score", "mllm_evaluation_anls_score"],
    "mmbench": ["overall"]
}

REPORT_BENCHMARKS_V1 = {
    'mmmu_v2': 'mmmu/mllm_eval_accuracy',
    'mmmu_v1': 'mmmu/accuracy',
    'docvqa': 'docvqa/anls_total_score',
    'mathvista': 'mathvista/accuracy',
    'ai2d': 'ai2d/accuracy',
    'chartqa': 'chartqa/accuracy',
    'vqa': 'vqa/accuracy',
    'textvqa': 'textvqa/accuracy',
    'infographics_w_ocr': 'infographics_w_ocr/anls_total_score',
    'infographics': 'infographics/anls_total_score',
    'mmbench': 'mmbench/overall'
}

# https://fburl.com/gsheet/r5hn8r1k
REPORT_BENCHMARKS_V2 = {
    'mmmu_v2': ['mmmu/mllm_eval_accuracy'],
    'mmmu_v1': ['mmmu/accuracy'],
    'docvqa': ["docvqa/anls_total_score", "docvqa/mllm_evaluation_anls_score", "docvqa/mmllm_fixed_anls_score"],
    'mathvista': ['mathvista/accuracy'],
    'ai2d': ['ai2d/accuracy'],
    'chartqa': ['chartqa/accuracy'],
    'vqa': ['vqa/accuracy', 'vqa/mllm_evaluation_accuracy'],
    'textvqa': ['textvqa/accuracy', 'textvqa/mllm_eval_accuracy'],
    'infographics_w_ocr': ['infographics_w_ocr/anls_total_score', 'infographics_w_ocr/mllm_evaluation_anls_score'],
    'infographics': ['infographics/anls_total_score', 'infographics/mllm_evaluation_anls_score'],
    'mmbench': ['mmbench/overall']
}

def get_log_file(job_id: int):
    return f"/fsx_0/user/tranx/slurm_logs/output_{job_id}.txt"


def read_json(file):
    with open(file, 'r') as f:
        # data = json.load(f)
        data = json.load(f, object_pairs_hook=OrderedDict)
        
    return data 


def get_bash_output(cmd: str, print_cmd: bool = False, print_output: bool = False):
    """
    Get output of a bash command 
    """
    if print_cmd:
        print(cmd)

    cmd = cmd.split(" ")

    try:
        output = subprocess.check_output(cmd).decode()
        if print_output:
            print(output)

        return output
    except Exception as e:
        print(f"{e}")
        return None
    
    
def run_sbatch_job(sbatch_base_script, sbatch_overwrite, positional_env_vars, print_cmd=False):
    sbatch_vars_string = []
    for k, v in sbatch_overwrite.items():
        sbatch_vars_string.append(f"--{k}={v}")
    sbatch_vars_string = ' '.join(sbatch_vars_string)

    positional_env_string = " ".join([str(x) for x in positional_env_vars])

    cmd = f"sbatch --parsable {sbatch_vars_string} {sbatch_base_script} {positional_env_string}"

    job_id = get_bash_output(cmd, print_output=False)
    job_id = int(job_id)
    
    if print_cmd:
        print(cmd)
        print("JOB ID:", job_id)

    return job_id


class EvalHelper():
    def __init__(self, code_dir: str, eval_sbatch: str, eval_config_dir: str):
        self.code_dir = code_dir
        self.eval_sbatch = eval_sbatch 
        self.eval_config_dir = eval_config_dir
        
    def run_eval_sweep(self, 
        checkpoint_dir, 
        eval_config_dir: Optional[str]=None, 
        print_cmd: bool = False, 
        checkpoints: Optional[List[int]] = None,
        min_checkpoint: int=0, 
        checkpoint_interval: Optional[int]=None,
        benchmarks: Optional[List[str]] = None,
        update_if_exists: bool = False,
        eval_plan: Optional[str]=None,
        slurm_qos: Optional[str]=None,
        max_num_jobs: int = 10
    ):
        if eval_config_dir is None:
            eval_config_dir = self.eval_config_dir
            
        run_eval_sweep(
            output_dir=checkpoint_dir,
            eval_sbatch=self.eval_sbatch,
            eval_config_dir=eval_config_dir,
            aligner_parent_dir=self.code_dir,
            print_cmd=print_cmd,
            checkpoints=checkpoints,
            min_checkpoint=min_checkpoint,
            checkpoint_interval=checkpoint_interval,
            benchmarks=benchmarks,
            update_if_exists=update_if_exists,
            eval_plan=eval_plan,
            slurm_qos=slurm_qos,
            max_num_jobs=max_num_jobs
        )
        
    def get_scores(self, 
        checkpoint_dir: str, 
        checkpoints:Optional[List[int]]=None, 
        min_checkpoint: Optional[int]=None,
        verbose: bool = False, 
        sorted_by: Optional[str]=None, 
        eval_plan=None, 
        report_version=None
    ) -> pd.DataFrame:
        
        print(f"Using REPORT_BENCHMARK_VERSION={report_version}")
        
        if checkpoints is None:
            checkpoints = get_checkpoints(checkpoint_dir)
            
        if verbose:
            print(checkpoints)
        df = pd.DataFrame()
        
        for c in checkpoints:
            if (min_checkpoint is not None) and (c < min_checkpoint):
                continue
            try:
                df_c = get_eval_scores(checkpoint_dir, c, verbose=verbose, eval_plan=eval_plan, report_version=report_version)
                df = pd.concat([df, df_c])   
                if verbose:  
                    print(c)   
            except Exception:
                pass
            
        if sorted_by is not None:
            df = df.sort_values(by=[sorted_by])
            
        return df
    

def get_report_benchmarks(all_benchmark_df: pd.DataFrame, report_version=None) -> pd.DataFrame:
    """
    report_version is one of ["v1", "v2", "hybrid"]
    """
    get_report_benchmarks_fn = {
        "v1": get_report_benchmarks_v1,
        "v2": get_report_benchmarks_v2,
        "hybrid": get_report_benchmarks_hybrid
    }
    
    if report_version is None:
        report_version = "v1"
        
    return get_report_benchmarks_fn[report_version](all_benchmark_df)


def get_report_benchmarks_hybrid(df):
    df1 = get_report_benchmarks_v1(df)
    df2 = get_report_benchmarks_v2(df)
    
    for col in df1.columns:
        df1[col] = pd.to_numeric(df1[col], errors='coerce')
        df2[col] = pd.to_numeric(df2[col], errors='coerce')
    
    df_report = df1.copy()
    
    for k in ['docvqa', 'vqa', 'textvqa', 'infographics_w_ocr', 'infographics']:
        df_report[k] = np.round(df1[k], 4).astype(str) + ' (' + np.round(df2[k], 4).astype(str) + ' )'
        
    return df_report


def get_report_benchmarks_v1(df: pd.DataFrame) -> pd.DataFrame:
    df_report = pd.DataFrame()
    
    for k, v in  REPORT_BENCHMARKS_V1.items():
        df_report[k] = df[v]
    
    # normalize textvqa score to be in the same scale as other benchmarks 
    df_report['textvqa'] = round(df_report['textvqa']/100, 4)
    
    return df_report


def get_report_benchmarks_v2(df: pd.DataFrame) -> pd.DataFrame:
    # df = all_benchmark_df[list(REPORT_BENCHMARKS.keys())]
    df_report = pd.DataFrame()
    
    # for k in READ_BENCHMARK_SCORES:
    #     if k == "mmmu":
    #         df_report["mmmu_v2"] = df["mmmu/mllm_eval_accuracy"]
    #         df_report["mmmu_v1"] = df["mmmu/accuracy"]
    #     else:
    #         read_columns = [f"{k}/{v}" for v in READ_BENCHMARK_SCORES[k]]
    #         df_k = df[read_columns]
    #         df_report[k] = df_k.max(axis=1)
    
    for k, v in REPORT_BENCHMARKS_V2.items():
        df_report[k] = df[v].max(axis=1)
            
    # normalize textvqa score to be in the same scale as other benchmarks 
    df_report['textvqa'] = round(df_report['textvqa']/100, 4)
    
    return df_report

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
    print_job_dict: bool = False,
    eval_plan: Optional[str]=None,
    slurm_qos: Optional[str]=None,
    conda_env_path: Optional[str]=None
):

    job_dict = {}

    # if os.path.exists(save_eval_jobs) and not rerun_if_exists:
    #     raise RuntimeError(f"Found existing eval_jobs at: {save_eval_jobs}")
    
    if os.path.exists(save_eval_jobs):
        if update_if_exists:
            job_dict = read_json(save_eval_jobs) 
        else:
            raise RuntimeError(f"Found existing eval_jobs at: {save_eval_jobs} and update_if_exists=False")

    if benchmarks is None:
        benchmarks = ALL_BENCHMARKS

    for benchmark in benchmarks:
        job_dict[benchmark] = {}

        for chk in checkpoints:
            # positional parameter, need to match with launch_eval_sbatch.sh
            if eval_plan is None:
                eval_plan = "\"\""  # passing empty string "" to sbatch script
            
            if conda_env_path is None:
                conda_env_path = "\"\"" # passing empty string "" to sbatch script
                
            params = {
                "aligner_parent_dir": aligner_parent_dir,
                "json_config": f"{eval_config_dir}/eval_{benchmark}.json",
                "checkpoint_dir": checkpoint_dir,
                "benchmark_name": benchmark,
                "checkpoint_id": str(chk),
                "conda_env_path": conda_env_path,
                "eval_plan": eval_plan
            } 
            # if eval_plan is not None:
            #     params = {
            #         "aligner_parent_dir": aligner_parent_dir,
            #         "json_config": f"{eval_config_dir}/eval_{benchmark}.json",
            #         "checkpoint_dir": checkpoint_dir,
            #         "benchmark_name": benchmark,
            #         "checkpoint_id": str(chk),
            #         "eval_plan": eval_plan
            #     }
            # else: # master branch does not accept eval_plan yet
            #     params = {
            #         "aligner_parent_dir": aligner_parent_dir,
            #         "json_config": f"{eval_config_dir}/eval_{benchmark}.json",
            #         "checkpoint_dir": checkpoint_dir,
            #         "benchmark_name": benchmark,
            #         "checkpoint_id": str(chk),
            #     }
            

            assert os.path.exists(params["json_config"])
            assert os.path.exists(
                f"{params['checkpoint_dir']}/checkpoint-{chk}")

            if slurm_qos is not None:
                sbatch_overwrite={
                    "job-name": f"eval_{benchmark}",
                    "qos": slurm_qos,
                    "account": slurm_qos
                }
            else:
                sbatch_overwrite={
                    "job-name": f"eval_{benchmark}"
                }
                
            job_id = run_sbatch_job(
                sbatch_base_script=eval_base_sbatch,
                sbatch_overwrite=sbatch_overwrite,
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
def get_eval_scores_v1(checkpoint_dir: str, checkpoint_id: int, output_csv=None, verbose: bool=False, eval_plan: Optional[str]=None)  -> pd.DataFrame:
    results = {}
    # print(f"get_eval_scores {eval_plan}/{checkpoint_id}")
    job_dict_file = get_eval_jobs_record(checkpoint_dir, checkpoint_id, eval_plan)
    with open(job_dict_file, 'r') as f:
        job_dict = json.load(f)
    
    if verbose:    
        print(job_dict)
    
    for b in job_dict:
        results[b] = {}
        for chk in job_dict[b]:
            job_id = job_dict[b][chk]
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
            # print(b, chk, res)
            for x in res:
                component, val = x[0], x[1]
                
                if component.startswith("eval_"):
                    component = component[5:]

                if component in component_scores:
                    df.loc[chk, component] = round(val, 4)

    df = get_report_benchmarks(df)

    if output_csv is not None:
        df.to_csv(output_csv, index=False)

    return df

def get_eval_scores(checkpoint_dir: str, checkpoint_id: int, report_version=None, output_csv=None, verbose: bool=False, eval_plan: Optional[str]=None)  -> pd.DataFrame:
    results = {}
    # print(f"get_eval_scores {eval_plan}/{checkpoint_id}")
    job_dict_file = get_eval_jobs_record(checkpoint_dir, checkpoint_id, eval_plan)
    with open(job_dict_file, 'r') as f:
        job_dict = json.load(f)
    
    if verbose:    
        print(job_dict)
    
    for b in job_dict:
        results[b] = {}
        for chk in job_dict[b]:
            job_id = job_dict[b][chk]
            log = get_log_file(int(job_id))
            res = extract_values(log)
            if res:
                if verbose:
                    print(f"Got result for {b} - {chk}: {res}")
                results[b][chk] = res

    # scores = {
    #     "mmmu": ["accuracy", "mllm_eval_accuracy"],
    #     "docvqa": ["anls_total_score", "mllm_evaluation_anls_score", "mmllm_fixed_anls_score"],
    #     "mathvista": ["accuracy"],
    #     "ai2d": ["accuracy"],
    #     "chartqa": ["accuracy"],
    #     "vqa": ["accuracy", "mllm_evaluation_accuracy"],
    #     "textvqa": ["accuracy", "mllm_eval_accuracy"],
    #     "infographics_w_ocr": ["anls_total_score", "mllm_evaluation_anls_score"],
    #     "infographics": ["anls_total_score", "mllm_evaluation_anls_score"],
    #     "mmbench": ["overall"]
    # }
    component_scores = []
    for k, v in READ_BENCHMARK_SCORES.items():
        for vi in v:
            component_scores.append(f"{k}/{vi}")

    df = pd.DataFrame(columns=component_scores)
    
    for b in results:
        for chk in results[b]:
            res = results[b][chk]
            # print(b, chk, res)
            for x in res:
                component, val = x[0], x[1]
                
                if component.startswith("eval_"):
                    component = component[5:]

                if component in component_scores:
                    df.loc[chk, component] = round(val, 4)

    df = get_report_benchmarks(df, report_version=report_version)
    # df = get
    # if REPORTING_SCORE_METHOD == 1:
    #     df = get_report_benchmarks(df)
    # elif REPORTING_SCORE_METHOD == 2:
    #     df = get_report_benchmarks_v2(df)
    # else:
    #     raise ValueError("Unsupported REPORTING_SCORE_METHOD")

    if output_csv is not None:
        df.to_csv(output_csv, index=False)

    return df

def get_eval_scores_v0(checkpoint_dir: str, checkpoint_id: int, output_csv=None, verbose: bool=False, eval_plan: Optional[str]=None)  -> pd.DataFrame:
    results = {}
    print(f"get_eval_scores {eval_plan}/{checkpoint_id}")
    job_dict_file = get_eval_jobs_record(checkpoint_dir, checkpoint_id, eval_plan)
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
            print(log)
            res = extract_values(log)
            print(res)
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
    print(results)
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

def get_eval_scores_all(checkpoint_dir: str, checkpoints:Optional[List[int]]=None, verbose: bool = False, sorted_by: Optional[str]=None, eval_plan=None) -> pd.DataFrame:
    
    if checkpoints is None:
        checkpoints = get_checkpoints(checkpoint_dir)
        
    if verbose:
        print(checkpoints)
    df = pd.DataFrame()
    
    for c in checkpoints:
        try:
            df_c = get_eval_scores(checkpoint_dir, c, verbose=verbose, eval_plan=eval_plan)
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


def get_checkpoints_eval(output_dir: str, eval_plan=None) -> List[int]:
    """
    List all checkpoints with eval jobs
    """
    if eval_plan is None:
        eval_plan = "evals"
        
    checkpoints = []
    for file in glob.glob(f"{output_dir}/{eval_plan}/*.json"):
        c = file.replace(".json", "").split("-")[-1]
        checkpoints.append(int(c))
    
    checkpoints = sorted(checkpoints)
    
    return checkpoints


def get_eval_jobs_record(checkpoint_dir: str, checkpoint_id: int, eval_plan=None):
    if eval_plan is None:
        eval_plan = "evals"
        
    return f"{checkpoint_dir}/{eval_plan}/eval_jobs_checkpoint-{checkpoint_id}.json"
    
    
def run_eval_sweep(
    output_dir: str, 
    eval_sbatch: str, 
    eval_config_dir: str, 
    aligner_parent_dir:str, 
    print_cmd:bool = False, 
    checkpoints: Optional[List[int]] = None,
    min_checkpoint: int=0, 
    checkpoint_interval: Optional[int]=None,
    benchmarks: Optional[List[str]] = None,
    update_if_exists: bool = False,
    eval_plan: Optional[str]=None,
    slurm_qos: Optional[str]=None,
    max_num_jobs: Optional[int]=10
):
    """
    Start eval sweep on new checkpoints
    """
    
    if benchmarks is None:
        benchmarks = ALL_BENCHMARKS
    
    if checkpoints is None:
        checkpoints = get_checkpoints(output_dir)
        
    checkpoints_eval = get_checkpoints_eval(output_dir, eval_plan)
    
    if update_if_exists:
        new_checkpoints = checkpoints 
    else:
        new_checkpoints = [int(c) for c in checkpoints if c not in checkpoints_eval]
        
    if checkpoint_interval is not None:
        new_checkpoints = [c for c in new_checkpoints if c % checkpoint_interval == 0]
        
    new_checkpoints = sorted(new_checkpoints)
    
    print(f"New checkpoints: {new_checkpoints}")
    
    if max_num_jobs is None:
        max_num_jobs = 10000
        
    launched_jobs = 0
    
    for c in new_checkpoints:
        if c < min_checkpoint:
            continue 
        
        
        if launched_jobs + len(benchmarks) > max_num_jobs:
            print(f"Will not launch more jobs due to exceeding max_num_jobs={max_num_jobs}.")
            break
        
        # if c not in checkpoints_eval:
        print(f"Start eval for {output_dir}/checkpoint-{c}")
            
        run_eval_plan(
            eval_base_sbatch=eval_sbatch,
            aligner_parent_dir=aligner_parent_dir,
            eval_config_dir=eval_config_dir,
            checkpoint_dir=output_dir,
            checkpoints=[c],
            benchmarks=benchmarks,
            save_eval_jobs=get_eval_jobs_record(output_dir, c, eval_plan),
            print_cmd=print_cmd,
            update_if_exists=update_if_exists,
            eval_plan=eval_plan,
            slurm_qos=slurm_qos
        )

        launched_jobs += len(benchmarks)