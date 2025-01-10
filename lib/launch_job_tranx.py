import ast
import copy
import datetime
import json
import logging
import os
import subprocess
import sys
from argparse import ArgumentParser

from itertools import chain
from typing import Any, TextIO

"""
Example usage:
To run a job, navigate to the appropriate directory and execute the script with necessary arguments:
cd llm_mm_aligner/experiments/aws
python launch_job.py --config pretrain/pretrain_8B_Llama3_336px.json

For local running on a compute node with debugging, 
python launch_job.py --config pretrain/pretrain_8B_Llama3_336px.json --debug_launch_script --local --env /data/home/kapilk/.conda/envs/aligner_20240822_v2/ --name test_local --devices 0,1 --debug


This module provides functionality for launching and managing SLURM jobs for training and evaluating  models with LLM MM Aligner framework. 
It includes capabilities to parse command-line arguments, generate SLURM batch scripts based on provided configurations, submit these scripts to a SLURM scheduler, and handle job configurations and environment setups.
The module supports launching jobs with specific configurations including the selection of compute resources, setting up the environment, and managing output and error logs. 
It also includes debugging support to attach a debugger to the running job or the launch script itself.
"""


logger = logging.getLogger(__name__)

DEFAULT_ACCOUNT = "midpri"
DEFAULT_QOS = "midpri"
DEFAULT_CONDA_ENVIRONMENT = "/fsx_0/shared/conda/latest"
JOB_LOG = "run_log.txt"
SLURM_CONFIG_KEY = "slurm_args"
TRAIN_CONFIG_KEY = "trainer_args"
EVAL_CONFIG_KEY = "eval_args"
OUTPUT_SCRIPT_NAME = "launch_script.sh"

# [tranx]
ALIGNER_PARENT_DIR = "/fsx_0/user/tranx/rsync"

# cluster tuning, read more here: https://fburl.com/gdoc/y8qarud4
CLUSTER_TUNING_FLAGS = """
export LOGLEVEL=INFO
export CUDA_LAUNCH_BLOCKING=0
export FI_EFA_SET_CUDA_SYNC_MEMOPS=0
export NCCL_BUFFSIZE=8388608 
export NCCL_P2P_NET_CHUNKSIZE=524288
export NCCL_DEBUG=INFO
export NCCL_SOCKET_IFNAME="enp"
export OMPI_MCA_btl_tcp_if_exclude="lo,docker0,nerdctl0,veth*"
export NCCL_IB_DISABLE=1
export CUDA_CACHE_PATH="/fsx_0/user/$USER/.nv/ComputeCache"
export LD_PRELOAD=/usr/local/cuda-12.3/lib/libnccl.so
export LD_LIBRARY_PATH=/usr/local/cuda-12.3/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/opt/aws-ofi-nccl/lib/:$LD_LIBRARY_PATH
export CUDA_DEVICE_MAX_CONNECTIONS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
"""


def parse_args():
    args = ArgumentParser()
    args.add_argument(
        "--config",
        "-c",
        type=str,
        required=True,
        help="The training / evaluation config json file.",
    )
    args.add_argument(
        "--name", "-n", type=str, default=None, help="The name of the job."
    )
    args.add_argument(
        "--env",
        type=str,
        default=None,
        help="Override the conda environment specified in the config.",
    )
    args.add_argument(
        "--devices",
        type=str,
        default=None,
        help="[For Local] A comma-separated list of cuda devices to run on locally. Default is all devices.",
    )
    args.add_argument(
        "--local",
        action="store_true",
        default=False,
        help="Run locally. NOTE: This should be run on a compute node. Don't run on submit node.",
    )
    args.add_argument(
        "--debug",
        "-d",
        action="store_true",
        default=False,
        help="Activate debugger for local running job",
    )
    args.add_argument(
        "--debug_launch_script",
        action="store_true",
        default=False,
        help="Attach debugger to the launch script.",
    )
    args.add_argument(
        "overrides",
        nargs="*",
        help="Dot separated overrides like trainer_args.per_device_train_batch_size 2",
    )
    return args.parse_args()


def run_job(
    config_file: str,
    name: str | None = None,
    local: bool = False,
    devices: str | None = None,
    debug: bool = False,
    conda_env: str | None = None,
    overrides: list[list[tuple[str, Any]]] | None = None,
    aligner_parent_dir: str | None = None
) -> str | None:
    """Loads the job configuration from a specified JSON file, processes it, and submits the job
    either locally or to a SLURM cluster based on the configuration and provided arguments.
    Args:
        config_file (str): The path to the JSON configuration file.
        name (str | None, optional): The name of the job to be used in the SLURM configuration.
        local (bool, optional): If True, the job will be executed locally instead of being submitted to SLURM.
        devices (str | None, optional): Specifies the devices to be used for the job. Currently not implemented.
        debug (bool, optional): If True, enables debugpy attaching to the running job from 5679 (see llm_mm_aligner/main.py).
            Independent from the debug_launch_script.
            Probably only should be run with local jobs, but could facilitate with remote debugging as well.
        conda_env (str | None, optional): The name of the conda environment to use for the job.
        overrides (list[list[tuple[str, Any]]] | None, optional): A list of list of tuples containing key-value pairs to override
            Mainly used when calling this module as a helper instead of from this script.
            e.g. [[
                ("trainer_args.per_device_train_batch_size", bs),
                ("eval_args.resume_from_checkpoint", ckpt),
            ]]
    """
    # [tranx]
    if aligner_parent_dir is not None:
        global ALIGNER_PARENT_DIR
        ALIGNER_PARENT_DIR = aligner_parent_dir
            
    with open(config_file, "r") as file:
        config = json.load(file)

    # using wd_datarecipe_file to populate wd_datarecipe
    if ("trainer_args" in config) and ("wd_datarecipe_file" in config["trainer_args"]):
        if "wd_datarecipe" in config["trainer_args"]:  # already populated
            raise ValueError(
                "Cannot use both wd_datarecipe_file and wd_datarecipe in trainer_args."
            )

        with open(config["trainer_args"]["wd_datarecipe_file"], "r") as file:
            config["trainer_args"]["wd_datarecipe"] = json.load(file)
            
            
    # use same run_name as slurm job name for consistency
    if name is None:
        if "trainer_args" in config:
            name = config["trainer_args"]["output_dir"].split("/")[-1]
        else:
            name = "unnamed"

    overrides[0].append(("trainer_args.run_name", name))

    # Setting up Configs
    config = _add_and_update_slurm_config_if_empty(
        config, name=name, conda_env=conda_env
    )
    config = override_args(config, overrides=overrides)
    config = process_config(config, name=name, debug=debug)
    validate_config(config)

    # Set up the working directory
    work_dir = os.path.dirname(config[SLURM_CONFIG_KEY]["output"])
    os.makedirs(work_dir, exist_ok=True)

    # write config file to the working dir. Make sure all configs are set by this point
    config_path = os.path.join(work_dir, "config.json")
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    logger.warning(f"Config file written to: {color_if_tty(config_path)}")

    # Create the launch bash scripts
    if local:
        logger.info("Creating local run script since --local is set. ")
        script_to_run = create_local_run_script(config, devices=devices)
    else:
        script_to_run = create_sbatch_script(config)

    # write sbatch script to the working dir
    script_path = os.path.join(work_dir, OUTPUT_SCRIPT_NAME)
    with open(script_path, "w") as file:
        file.write(script_to_run)
    logger.warning(f"script written to: {color_if_tty(script_path)}")

    if local:
        run_job_locally(script_path)
        return None
    else:
        job_id = submit_job(script_path)
        return int(job_id), work_dir


def create_sbatch_script(config: dict) -> str:
    """Generate the contents of an sbatch script based on the provided configuration.

    Args:
        config (dict): The job configuration.

    Returns:
        str: The sbatch script.
    """
    slurm_config = config[SLURM_CONFIG_KEY]
    file_to_run = _get_file_to_run(config)

    work_dir = os.path.dirname(slurm_config["output"])
    config_file = os.path.join(work_dir, "config.json")
    aligner_parent_folder = _get_aligner_parent_repo_path()
    output_file_str = _get_output_file_for_evals(config)

    sbatch_script = f"""#!/bin/bash
#SBATCH --job-name={slurm_config['job_name']}
#SBATCH --nodes={slurm_config['nodes']}
#SBATCH --ntasks={slurm_config['nodes']}
#SBATCH --gpus-per-task={slurm_config['gpus_per_task']}
#SBATCH --cpus-per-task={slurm_config['cpus_per_task']}
#SBATCH --mem={slurm_config['mem']}
#SBATCH --output={slurm_config['output']}
#SBATCH --error={slurm_config['error']}
#SBATCH --time={slurm_config['time']}
#SBATCH --account={slurm_config['account']}
#SBATCH --qos={slurm_config['qos']}
#SBATCH --wait-all-nodes={slurm_config['wait_all_nodes']}
#SBATCH --exclusive
#SBATCH --requeue
#SBATCH --exclude h100-st-p548xlarge-421
# The set -e command causes the bash script to exit immediately if a command exits with a non-zero status (which indicates an error).
set -eo pipefail

# Activate conda environment
CONDA_ENV={slurm_config['conda_env']}
eval "$(conda shell.bash hook)"
source activate $CONDA_ENV
echo Using conda environment: $CONDA_ENV
echo Node list: $SLURM_JOB_NODELIST

{CLUSTER_TUNING_FLAGS}

# launcher
ALIGNER_PARENT_DIR={aligner_parent_folder}
ALIGNER_DEP_DIR=$ALIGNER_PARENT_DIR/llm_mm_aligner/replicated
CONDA_PYTHON_PKGS=$CONDA_PREFIX/python-packages
head_node_ip=$(srun --nodes=1 --ntasks=1 hostname --ip-address)
echo Node IP: $head_node_ip
JSON_CONFIG={config_file}
echo "starting with the following json config: \n\n"
cat $JSON_CONFIG


PYTHONPATH=$PYTHONPATH:$ALIGNER_PARENT_DIR:$ALIGNER_DEP_DIR:$CONDA_PYTHON_PKGS srun --cpus-per-gpu 24 torchrun \
    --nnodes $SLURM_NNODES \
    --nproc_per_node 8 \
    --rdzv_id $RANDOM \
    --rdzv_backend c10d \
    --rdzv_endpoint $head_node_ip:29500 \
    $ALIGNER_PARENT_DIR/llm_mm_aligner/{file_to_run} $JSON_CONFIG {output_file_str}
"""
    return sbatch_script


def create_local_run_script(config: dict, devices: str | None = None) -> str:
    """Generate the contents of an sbatch script based on the provided configuration.

    Args:
        config (dict): The job configuration.
        devices (str | None): The devices to run on. Defaults to None.
            e.g. "0,1,2,3" for 4 gpus.

    Returns:
        str: The sbatch script.
    """
    slurm_config = config[SLURM_CONFIG_KEY]

    file_to_run = _get_file_to_run(config)
    work_dir = os.path.dirname(slurm_config["output"])
    config_file = os.path.join(work_dir, "config.json")
    aligner_parent_folder = _get_aligner_parent_repo_path()
    output_file_str = _get_output_file_for_evals(config)

    num_processes = 8
    cuda_visible_devices_str = ""
    if devices is not None:
        device_list = devices.split(",")
        num_processes = len(device_list)
        cuda_visible_devices_str = f"CUDA_VISIBLE_DEVICES={devices}"

    # NOTE: If debugging the shell scripts, consider using set -x for further debugging.
    # TODO: find a way to be able to utilize set -u option, but right now $PYTHONPATH is set externally.
    run_script = f"""#!/bin/bash

set -o pipefail

JSON_CONFIG={config_file}
CONDA_ENV={slurm_config['conda_env']}

# Activate conda environment
echo "Activating conda env from $CONDA_ENV"
eval "$(conda shell.bash hook)"
source activate $CONDA_ENV

echo "Config: $JSON_CONFIG"
echo "Conda env: $CONDA_PREFIX"

{CLUSTER_TUNING_FLAGS}

# Auto-infer the path
ALIGNER_PARENT_DIR={aligner_parent_folder}

PYTHON_PATH_EXTRAS=$PYTHONPATH:\
$ALIGNER_PARENT_DIR:\
$ALIGNER_PARENT_DIR/llm_mm_aligner/replicated:\
$CONDA_PREFIX/python-packages

echo "starting with the following json config: \n\n"
cat $JSON_CONFIG

{cuda_visible_devices_str}
PYTHONPATH=${{PYTHONPATH}}:${{PYTHON_PATH_EXTRAS}} torchrun --standalone \
    --nproc_per_node={num_processes} ${{ALIGNER_PARENT_DIR}}/llm_mm_aligner/{file_to_run} $JSON_CONFIG {output_file_str}

echo "DONE"
"""
    return run_script


def _get_output_file_for_evals(config) -> str:
    """Gets the string of the output file for the bash scripts to run shim_evaluate.py from the config."""
    job_type = _get_job_type(config)
    if job_type != "eval":
        return ""
    output_file = config[SLURM_CONFIG_KEY]["output_file"]
    if not output_file:
        raise ValueError(f"Pass valid output file to shim_evaluate.py {output_file}")
    return output_file


def _get_file_to_run(config: dict) -> str:
    """Get the file to run based on the job type.

    Args:
        config (dict): The job configuration.

    Returns:
        str: The file to run.
    """
    slurm_config = config[SLURM_CONFIG_KEY]
    job_type = slurm_config["job_type"]
    if job_type == "train":
        file_to_run = "main.py"
    elif job_type == "eval":
        file_to_run = "aws/shim_evaluate.py"
    else:
        raise ValueError(
            f"Invalid job type {job_type!r}. Must be either 'train' or 'eval'."
        )
    return file_to_run


def submit_job(sbatch_script_path: str) -> str:
    """Submit the sbatch script using sbatch.

    Args:
        sbatch_script_path (str): The path to the sbatch script.

    Raises:
        subprocess.CalledProcessError: If the sbatch command fails.
    """
    try:
        output = subprocess.check_output(["sbatch", "--parsable", sbatch_script_path])
    except subprocess.CalledProcessError as e:
        logger.exception(f"Failed to submit job: {e}")
        raise
    logger.info("Job submitted successfully.")
    logger.info(output.decode("utf-8"))
    
    return output

def run_job_locally(script_path: str) -> None:
    """Run the script locally.

    Args:
        script_path (str): The path to the launch script.

    Raises:
        subprocess.CalledProcessError: If the sbatch command fails.
    """
    try:
        output = subprocess.check_output(["bash", script_path])
    except subprocess.CalledProcessError:
        logger.exception("Job Failed to run.")
    logger.info("Job run successfully.")
    logger.info(output.decode("utf-8"))


def _get_date_time_str() -> str:
    """Get a string representation of the current date and time.

    Returns:
        str: The current date and time.
    """
    now = datetime.datetime.now()
    str_now = now.strftime("%y%m%d_%H_%M_%S_%f")  # "yymmdd_HH_MM_SS_microseconds"
    return str_now


def _get_aligner_parent_repo_path() -> str:
    """Gets the aligner parent directory location based on known location of this code.
    Returns: aligner path
    """
    # current_directory = os.path.dirname(os.path.abspath(__file__))
    # aligner_directory = os.path.abspath(
    #     os.path.join(current_directory, os.pardir, os.pardir, os.pardir)
    # )
    # return aligner_directory
    
    # [tranx]
    return ALIGNER_PARENT_DIR


def _substitute_placeholders(config: dict, substitutions: list[str]) -> dict:
    config = copy.deepcopy(config)
    username = os.environ["USER"]
    for key in substitutions:
        if key in config:
            config[key] = config[key].replace("$USER", username)
    return config


def default_slurm_config():
    return {
        "nodes": 1,
        "gpus_per_task": 8,
        "cpus_per_task": 24,
        "mem": 0,
        "time": "168:00:00",
        "account": DEFAULT_ACCOUNT,
        "qos": DEFAULT_QOS,
        "wait_all_nodes": 1,
        "conda_env": DEFAULT_CONDA_ENVIRONMENT,
        "job_name": "test_name",
        "job_type": "train",
        "output": "",
        "error": "",
        ## NOTE: Used only for evals to pass in workdir for shim_evaluate.py
        "output_file": None,
    }


def _add_and_update_slurm_config_if_empty(
    config: dict, name: str | None = None, conda_env: str | None = None
) -> dict:
    """Ensures that the provided configuration dictionary includes a SLURM configuration.
    If the SLURM configuration is missing, it adds a default SLURM configuration.
    Additionally, it allows for specifying a job name and a conda environment, which
    will be added or updated in the SLURM configuration.
    Parameters:
        config (dict): The initial configuration dictionary.
        name (str | None, optional): The name of the SLURM job. If provided, it updates
                                     the job name in the SLURM configuration.
        conda_env (str | None, optional): The name of the conda environment to use.
                                          If provided, it updates the conda environment
                                          in the SLURM configuration.
    Returns:
        dict: The updated configuration dictionary containing the SLURM configuration.
    """
    config = config.copy()
    # create the default config.
    slurm_config = default_slurm_config()
    if SLURM_CONFIG_KEY in config:
        slurm_config.update(config[SLURM_CONFIG_KEY])
        config[SLURM_CONFIG_KEY] = slurm_config
    else:
        logging.warning(
            f"Slurm config is missing. Default values are used: {slurm_config}"
        )

    config[SLURM_CONFIG_KEY] = slurm_config
    if name is not None:
        config[SLURM_CONFIG_KEY]["job_name"] = name
    if conda_env is not None:
        config[SLURM_CONFIG_KEY]["conda_env"] = conda_env

    return config


def validate_config(config: dict) -> None:
    """Verify selected fields of the job configuration.

    Args:
        config (dict): The job configuration.

    Raises:
        ValueError: If one or more fields of the input confile is invalid.
    """
    slurm_config = config[SLURM_CONFIG_KEY]
    if (job_type := _get_job_type(config)) not in ["train", "eval"]:
        raise ValueError("job_type must either be 'train' or 'eval'")
    if slurm_config["nodes"] < 1:
        raise ValueError("Number of nodes must be at least 1.")
    if not slurm_config.get("conda_env"):
        raise ValueError("Conda environment must be specified.")
    if not slurm_config.get("job_name"):
        raise ValueError("Job name must be specified.")

    if job_type == "train":
        trainer_args = config.get("trainer_args")
        if not trainer_args.get("output_dir"):
            raise ValueError("ouput_dir must be specified.")
    else:
        # eval
        eval_args = config.get("eval_args")
        # TODO: eval validations
        checkpoint_path = eval_args.get("resume_from_checkpoint")
        if not os.path.isdir(checkpoint_path):
            raise ValueError(
                f"resume_from_checkpoint must be a valid checkpoint directory: {checkpoint_path}"
            )
        eval_ckpt_id = eval_args.get("eval_ckpt")
        if not eval_ckpt_id and isinstance(eval_ckpt_id, int):
            raise ValueError(
                f"Please set eval_ckpt to be valid. eval_ckpt is: {eval_ckpt_id}"
            )
        eval_type = eval_args.get("eval_type")
        if not eval_type and isinstance(eval_ckpt_id, str):
            raise ValueError(
                f"Please set eval_type to be the benchmark name. eval_type is : {eval_type}"
            )


def override_args(
    config: dict[str, Any],
    overrides: list[list[tuple[str, Any]]] | None = None,
) -> dict[str, Any]:
    """Overrides the values in a nested configuration dictionary based on a list of key-value pairs.
    This function takes a configuration dictionary and a list of tuples, where each tuple contains
    a dot-separated key path and a value. The function updates the configuration dictionary by
    setting the value at the specified key path.
    Args:
        config: The original configuration dictionary to be modified.
        overrides: key path (as a string) and the value to set at that path in list of list of overrides.
    Returns:
        dict[str, Any]: The updated configuration dictionary with the overridden values.
    Example:
        original_config = {
            'trainer_args': {
                'per_device_train_batch_size': 10
            }
        }
        overrides = [[("trainer_args.per_device_train_batch_size", 20)]]

        updated_config = override_args(original_config, overrides)
        # updated_config will be:
        # {
        #     'trainer_args': {
        #         'per_device_train_batch_size': 20
        #         }
        #     }
        # }
    Note:
        - The function uses `deepcopy` to avoid modifying the original configuration dictionary.
        - If a key path does not exist in the original dictionary, it will be created.
    """
    if overrides is None:
        return config
    config = copy.deepcopy(config)
    list_args = list(chain.from_iterable(overrides))
    for path, value in list_args:
        keys = path.split(".")
        temp = config
        for key in keys[:-1]:
            temp = temp.setdefault(key, {})
        try:
            temp[keys[-1]] = ast.literal_eval(value)
        except (ValueError, SyntaxError):
            # NOTE: fallback for all string values. If we try except, then we're going to spam the logs a bunch.
            temp[keys[-1]] = value
    return config


def process_override_pairs(args_list: list[str]) -> list[list[tuple[str, Any]]]:
    """
    Create a list of tuples from a list of arguments.

    Args:
    args_list: A list containing keys and values alternately.

    Returns:
    list: each tuple is a (key, value) pair of overrides.

    Raises:
    ValueError: If the number of elements in args_list is not even.
    """
    # Check if the number of arguments is even (key-value pairs)
    if len(args_list) % 2 != 0:
        raise ValueError(
            f"Please provide an even number of arguments (key-value pairs). {args_list}"
        )

    # Create a list of tuples from the arguments
    key_value_pairs = [
        (args_list[i], args_list[i + 1]) for i in range(0, len(args_list), 2)
    ]
    return [key_value_pairs]


def process_config(config: dict, name: str, debug: bool = False) -> dict:
    """Processes the input configuration dictionary for a job submission by performing specific
    substitutions and setting default values for certain fields if they are missing.
    The function performs the following operations:
    1. Replaces placeholder values such as $USER with the actual current user's name.
    2. Sets default paths for 'output' and 'error' logs in the SLURM configuration if they are not specified.
    3. Substitutes placeholders in various fields within the 'trainer_args' and SLURM configuration.
    4. Updates debug flags if debug argument is set.
    Args:
        config (dict): The job configuration dictionary which contains both SLURM and trainer-specific settings.
    Returns:
        dict: The updated job configuration dictionary with processed values.
    """
    config = config.copy()
    slurm_config = config[SLURM_CONFIG_KEY]

    slurm_config.setdefault("output", "")
    slurm_config.setdefault("error", "")

    substitutions = ["output", "error"]
    config[SLURM_CONFIG_KEY] = _substitute_placeholders(slurm_config, substitutions)
    slurm_config = config[SLURM_CONFIG_KEY]

    job_type = _get_job_type(config)
    job_config_key = _get_job_config_key(job_type)
    if job_type == "train":
        config = _substitute_train_config(config)
    elif job_type == "eval":
        config = _substitute_eval_config(config)
    else:
        raise ValueError(f"Invalid job type {job_type}.")

    # TODO: for evals what should the output directory be.
    output_dir = config[job_config_key]["output_dir"]

    # create work_dir name following wandb naming pattern
    work_dir = os.path.join(output_dir, _get_date_time_str())

    log_file = os.path.join(work_dir, JOB_LOG)
    if not slurm_config["output"]:
        slurm_config["output"] = log_file
        logger.warning(f"stdout is redirected to {color_if_tty(log_file)}")
    if not slurm_config["error"]:
        slurm_config["error"] = log_file
        logger.warning(f"stderr is redirected to {color_if_tty(log_file)}")

    if debug:
        config[job_config_key]["attach_debugger"] = True

    config[SLURM_CONFIG_KEY] = slurm_config
    return config


def _substitute_train_config(config: dict) -> dict:
    """
    Substitutes placeholders in the configuration dictionary for train jobs.
    Args:
        config (dict): The job configuration dictionary.
    Returns:
        dict: The updated job configuration dictionary with substituted placeholders.
    """
    substitutions = [
        "model_name_or_path",
        "tokenizer_path",
        "modality_tokenizer_name",
        "checkpoints_perception_tokenizer",
        "output_dir",
        "train_file",
        "validation_file",
    ]
    job_config = _get_job_config(config)
    job_config = _substitute_placeholders(job_config, substitutions)
    config[TRAIN_CONFIG_KEY] = job_config
    return config


def _substitute_eval_config(config: dict) -> dict:
    """
    Substitutes placeholders in the configuration dictionary for evaluation jobs.
    Args:
        config (dict): The job configuration dictionary.
    Returns:
        dict: The updated job configuration dictionary with substituted placeholders.
    """
    job_config = _get_job_config(config)
    benchmark_name = job_config["eval_type"]

    evals_path = config[EVAL_CONFIG_KEY]["output_dir"]

    result_path = os.path.join(evals_path, f"{benchmark_name}_eval_results.txt")
    logger.warning(f"Eval output is at: {color_if_tty(result_path)}")

    config[SLURM_CONFIG_KEY]["output_file"] = result_path
    config[EVAL_CONFIG_KEY] = job_config
    return config


def _get_job_type(config: dict) -> str:
    """Retrieves the job type (train, eval) from the config.
    Returns: a string like `train` or `eval`"""
    return config[SLURM_CONFIG_KEY]["job_type"]


def _get_job_config_key(job_type: str) -> str:
    """Retrieves the job type (train, eval) key like TRAIN_CONFIG_KEY or EVAL_CONFIG_KEY"""
    if job_type == "train":
        return TRAIN_CONFIG_KEY
    elif job_type == "eval":
        return EVAL_CONFIG_KEY
    else:
        raise ValueError(f"Invalid job type used {job_type}")


def _get_job_config(config: dict) -> dict:
    """Retrieves the configuration for a specific job type from the provided config dictionary.
    Args:
        config (dict): A dictionary containing the overall configuration, including Slurm and job-specific settings.
    Returns:
        dict: The configuration dictionary for the specified job type.
    Raises:
        ValueError: If an invalid job type is encountered.
    Notes:
        The job type is determined by the value of 'job_type' under the '{SLURM_CONFIG_KEY}' key in the config dictionary.
        Currently supported job types are 'train' and 'eval'.
    """
    job_type = _get_job_type(config)
    config_key = _get_job_config_key(job_type)
    return config[config_key]


CYAN = "\033[36m"


def color_if_tty(
    message: str,
    color: str = CYAN,  # https://fburl.com/code/gvq1muh1
    stream: TextIO | None = None,
) -> str:
    """Returns an ANSI-colored message if stream is a TTY.

    Args:
        message: message to optionally color.
        color: ANSI color to use. ansicolor package to get the color code.
        stream: stream that the message will be written to. Defaults to stderr.
    """
    CLEAR = "\033[0m"
    if stream is None:
        stream = sys.stderr
    if hasattr(stream, "isatty") and stream.isatty():
        return f"{color}{message}{CLEAR}"
    return message


def attach_debugger(debug: bool = False, port: int = 5686) -> None:
    """Attaches a debugger to the current process if debug is True.

    # NOTE: do not use the same port as the default llm_mm_aligner port which is 5679.
    Args:
        debug: Whether to attach a debugger. Defaults to False.
    """
    if debug:
        import debugpy

        logger.warning(f"debugpy: {port}. Make sure to ssh tunnel to the node. ")
        debugpy.listen(port)
        debugpy.wait_for_client()  # blocks execution until client is attached


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    args = parse_args()
    attach_debugger(args.debug_launch_script)
    overrides = process_override_pairs(args.overrides)
    run_job(
        config_file=args.config,
        name=args.name,
        local=args.local,
        devices=args.devices,
        debug=args.debug,
        conda_env=args.env,
        overrides=overrides,
    )
