import os
from typing import Any
from functools import partial
import json
from pprint import pprint
from collections import OrderedDict
from datetime import datetime
import pytz

from lib.launch_job_tranx import run_job
from lib import utils

class Launcher:
    def __init__(self, 
        aligner_parent_dir: str, 
        config_base_dir: str 
    ):
        self.aligner_parent_dir = aligner_parent_dir
        self.launch_fn = run_job
        self.config_base_dir = config_base_dir 
        self.run_log_file = None
        self.run_log = None
        
    
    def cancel(self, wandb_project_name: str, experiment: str):
        """
        Cancel all running jobs of a given experiment
        """        
        run_log = self._read_run_log(wandb_project_name, experiment)
        
        for job in run_log[experiment]:
            job_id = job.get("job_id", None)
            if job_id is not None:
                print(f"Try canceling job {job_id} if running")
                utils.get_bash_output(f"scancel {job_id}")
        
    def run(
        self,
        config: str,
        nodes: int = 1,
        # overrides_dict = None,
        trainer_args = None,
        qos: str = "ar-ai-hipri",
        conda_env: str = "/fsx_0/user/ahmadyan/.conda/envs/aligner_20240822",
        note = None,
        experiment = "default",
        name = None,
        excludes=None
    ):
        config_file = os.path.join(self.config_base_dir, config)
        config_data = utils.read_json(config_file)
        
        assert "wandb_project_name" in config_data["trainer_args"]
        assert "output_dir" in config_data["trainer_args"]
        
        wandb_project_name = config_data["trainer_args"]["wandb_project_name"]
        if name is None:
            if trainer_args is not None and "output_dir" in trainer_args:
                name = trainer_args["output_dir"].split("/")[-1]
            else:
                name = config_data["trainer_args"]["output_dir"].split("/")[-1]
        
        self.run_log = self._read_run_log(wandb_project_name, experiment)
        
        overrides = [[
            ("slurm_args.qos", qos),
            ("slurm_args.account", qos),
            ("slurm_args.nodes", nodes),
            ("slurm_args.exclude", "h100-st-p548xlarge-337")
        ]]
        
        if trainer_args is not None:
            for k, v in trainer_args.items():
                overrides[0].append((
                    f"trainer_args.{k}", v
                ))
        
        # if overrides_dict is not None:
        #     for k, v in overrides_dict.items():
        #         overrides[0].append((
        #             k, v
        #         ))

        info = OrderedDict([
            ("name", name),
            ("note", note),
            ("job_id", None),
            ("nodes", nodes),
            ("timestamp", self._get_timestamp()),
            ("input_config", config),
            ("config_base", self.config_base_dir),
            ("trainer_args_overrides", trainer_args),
            ("conda_env", conda_env)
        ])
            
        pprint(info)

        job_id, work_dir = self.launch_fn(
            config_file=config_file,
            conda_env=conda_env,
            overrides=overrides,
            aligner_parent_dir=self.aligner_parent_dir,
            name=name
        )
           
        if job_id is not None:
            info.update({
                "job_id": job_id,
                "work_dir": work_dir,
                "run_config": os.path.join(work_dir, "config.json")
            })
            
            # insert job to top of the list
            if experiment not in self.run_log:
                self.run_log[experiment] = []
                
            self.run_log[experiment] = [info] + self.run_log[experiment]
            self._save_run_log(wandb_project_name, experiment)
        
    def _read_run_log(self, wandb_project_name, experiment):
        run_log_file = self._run_log_file(wandb_project_name)
        try:
            with open(run_log_file, 'r') as f:
                run_log = json.load(f)
                return run_log
        except Exception:
            return {}
    
    def _save_run_log(self, wandb_project_name, experiment):
        run_log_file = self._run_log_file(wandb_project_name)
        with open(run_log_file, 'w') as f:
            json.dump(self.run_log, f, indent=4)
            
        print(f"Save run log to {run_log_file}")
        
    
    def _run_log_file(self, wandb_project_name):
        run_log_file = f"/fsx_0/user/tranx/experiments/ablations/run_log/{wandb_project_name}.json"
        return run_log_file
            
    def _get_timestamp(self):
        timestamp = datetime.now(pytz.timezone(zone='America/New_York'))
        timestamp = timestamp.strftime('%Y-%m-%d %H:%M:%S')
        
        return timestamp
            
