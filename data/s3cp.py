import subprocess
from typing import Dict, Any, List, TextIO
import pandas as pd
from datetime import datetime
import os
import pytz
import json
from collections import OrderedDict
import sys 
import multiprocessing

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
    

def get_file_list(s3_dir: str):
    # example s3_dir = s3://fb-m2c2/metaclip_v2/metaclip_v2_2b_090924/0/
    
    if not s3_dir.endswith("/"):
        s3_dir += "/"
    
    output = get_bash_output(f"aws s3 ls {s3_dir}")

    files = []
    for item in output.split("\n"):
        item = item.split(" ")[-1].strip()
        item = f"{s3_dir}{item}"
        files.append(item)
    
    return files

def sync_dir(s3_dir: str, local_dir: str):
    """
    aws s3 sync s3://your-bucket-name/your-folder-path/ /local/destination/path/
    """
    assert s3_dir.endswith("/")
    assert local_dir.endswith("/")
    
    if not os.path.exists(local_dir):
        os.makedirs(local_dir)
        print(f"Directory '{local_dir}' created.")
    
    cmd = f"aws s3 sync {s3_dir} {local_dir}"
    output = get_bash_output(cmd)
    


def start_multiprocess(worker_function, args_list):
    processes = []
    
    for args in args_list:
        process = multiprocessing.Process(target=worker_function, args=args)
        processes.append(process)
        process.start()

    for process in processes:
        process.join()
        
    print("All processes have completed.")
    
def num_to_str_id(num: int, str_len: int) -> str:
    """
    Convert a number to a string ID with a specified length.
    Args:
        num (int): The number to convert.
        str_len (int): The desired length of the resulting string ID.
    Returns:
        A string ID with the specified length.
    """
    str_num = str(num)
    str_id = "0" * (str_len - len(str_num)) + str_num
    return str_id


def download_fb_m2c2(n):
    
    args = []
    for i in range(n):
        args.append((
            f"s3://fb-m2c2/metaclip_v2/metaclip_v2_2b_090924/{i}/",
            f"/fsx_3/dataset01/metaclip_v2_2b_090924/{i}/"
        ))
        
    start_multiprocess(
        worker_function=sync_dir,
        args_list=args
    )
    
    
def download_mmai_data(
    s3_base_dir: str,
    from_shard_index: int,
    to_shard_index: int,
    local_base_dir: str):
    
    args = []
    for i in range(from_shard_index, to_shard_index + 1):
        shard_id = num_to_str_id(i, 4)
        src_path = os.path.join(s3_base_dir, shard_id)
        target_path = os.path.join(local_base_dir, shard_id)
        
        print(src_path, target_path)
        args.append((src_path, target_path))
        
    start_multiprocess(
        worker_function=sync_dir,
        args_list=args
    )
    

if __name__ == "__main__":
    
    # clear partial files
    # cmd = "find /fsx_3/dataset01/metaclip_v2_2b_090924/ -type f -name '*.tar.*' -exec rm -f {} +"
    # get_bash_output(cmd)
    
    n = 100
    download_mmai_data(
        s3_base_dir="s3://ar-ai-s3-use2/datasets_30days/sg_mmllm_stage2_compliant_cap_qa_exp28_kosher_v2/20240820/",
        from_shard_index=0,
        to_shard_index=99,
        local_base_dir="/fsx_3/datasets_30days/sg_mmllm_stage2_compliant_cap_qa_exp28_kosher_v2/20240820"
    )


        