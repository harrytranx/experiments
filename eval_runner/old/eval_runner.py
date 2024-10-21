"""
Usage:

nohup python eval_runner.py > output.log & 
nohup python eval_runner.py > output.log 2>&1 &

ps aux | grep eval_runner
pgrep -f "eval_runner.py" | xargs kill
"""
import sys 
lib_path = '/fsx_0/user/tranx/experiments'
if lib_path not in sys.path:
    sys.path.append(lib_path)
    
import time 
from lib import utils
from lib import eval_helper

def main_old():
    # ev_master = eval_helper.EvalHelper(
    #     code_dir="/fsx_0/user/tranx/rsync",
    #     eval_sbatch="/fsx_0/user/tranx/rsync/llm_mm_aligner/experiments/aws/launch_eval_sbatch.sh",
    #     eval_config_dir="/fsx_0/user/tranx/rsync/llm_mm_aligner/experiments/aws/mm10/stage2/eval_overwrite"
    # )
    
    ev_unfreeze = eval_helper.EvalHelper(
        code_dir="/fsx_0/user/tranx/rsync",
        eval_sbatch="/fsx_0/user/tranx/rsync/llm_mm_aligner/experiments/aws/launch_eval_sbatch.sh",
        eval_config_dir="/fsx_0/user/tranx/rsync/llm_mm_aligner/experiments/aws/mm10/stage2/eval_overwrite_w_checkpoints_perception"
    )
    
    # watch_path = "/fsx_0/checkpoints/mm10/MM10-Stage1-70B/MH21_70B_224px_norm_R3"
    watch_paths = [
        # "/fsx_0/checkpoints/mm10/MM10-Stage2-70B/MH21_70B_336px_exp32a",
        # "/fsx_0/checkpoints/mm10/MM10-Stage2-70B/MH21_70B_336px_exp32a_bz48"
        # "/fsx_0/checkpoints/mm10/MM10-Stage2-70B/MH21_70B_336px_exp32a_unfreeze"
        "/fsx_0/checkpoints/mm10/MM10-Stage2-70B/MH21_70B_336px_exp30d_m2c2_036_unfreeze",
        "/fsx_0/checkpoints/mm10/MM10-Stage2-70B/MH21_70B_336px_exp30d_m2c2_036_unfreeze_lr1",
        "/fsx_0/checkpoints/mm10/MM10-Stage2-70B/MH21_70B_336px_exp32a_lr1_n128",
        "/fsx_0/checkpoints/mm10/MM10-Stage2-70B/MH21_70B_336px_exp30d_m2c2_036_unfreeze_lr1_n128",
        "/fsx_0/checkpoints/mm10/MM10-Stage2-70B/MH21_70B_336px_exp30d_plus_unfreeze_lr1_n128"
    ]
    
    while True:  
        for path in watch_paths:
            ts = utils.get_local_time()
            print(f"[{ts}] scanning for new checkpoints in: {path}")
            
            ev_unfreeze.run_eval_sweep(
                checkpoint_dir=path,
                # benchmarks=["mmmu"],
                benchmarks=eval_helper.ALL_BENCHMARKS,
                slurm_qos="midpri",
                # checkpoints=[100],
                update_if_exists=False,
                print_cmd=True
            )
        
        time.sleep(300) 

def main():

    while True:
        eval_watcher_config = utils.read_json("eval_watcher_config.json")
        print(eval_watcher_config)
    
        eval_runner = eval_helper.EvalHelper(
            code_dir=eval_watcher_config["code_dir"],
            eval_sbatch=eval_watcher_config["eval_sbatch"],
            eval_config_dir=eval_watcher_config["eval_config_dir"]
        )
    
        for job in eval_watcher_config["watch_jobs"]:
            ts = utils.get_local_time()
            path = job["watch_path"]
            print(f"[{ts}] scanning for new checkpoints in: {path}")
            
            benchmarks = job.get("benchmarks", None)
            if benchmarks is None:
                benchmarks = eval_helper.ALL_BENCHMARKS
            assert isinstance(benchmarks, list)

            eval_runner.run_eval_sweep(
                checkpoint_dir=path,
                benchmarks=benchmarks,
                slurm_qos="midpri",
                update_if_exists=False,
                print_cmd=True
            )
        
        time.sleep(eval_watcher_config["scan_every_n_seconds"]) 
        
if __name__ == "__main__":
    main()