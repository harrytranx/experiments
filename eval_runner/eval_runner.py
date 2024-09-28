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

def main():
    ev_master = eval_helper.EvalHelper(
        code_dir="/fsx_0/user/tranx/rsync",
        eval_sbatch="/fsx_0/user/tranx/rsync/llm_mm_aligner/experiments/aws/launch_eval_sbatch.sh",
        eval_config_dir="/fsx_0/user/tranx/rsync/llm_mm_aligner/experiments/aws/mm10/stage1/eval_overwrite"
    )
    
    OUTPUT_S1_norm_r3 = "/fsx_0/checkpoints/mm10/MM10-Stage1-70B/MH21_70B_224px_norm_R3"
    
    while True:  
        ts = utils.get_local_time()
        print(ts)
        
        ev_master.run_eval_sweep(
            checkpoint_dir=OUTPUT_S1_norm_r3,
            benchmarks=["mmmu"],
            # benchmarks=eval_helper.ALL_BENCHMARKS,
            slurm_qos="midpri",
            # checkpoints=[100],
            update_if_exists=False,
            print_cmd=True
        )
        
        time.sleep(300) 
        
if __name__ == "__main__":
    main()