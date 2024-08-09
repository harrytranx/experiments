import pandas as pd
from typing import Optional, Dict, Any, List
import json
import re
from pprint import pprint
import os

import utils


class SlurmClient():
    def __init__(self):
        self.host_prefix = "h100-st-p548xlarge"  # FIX ME, move this to config file

    @property
    def num_nodes(self):
        """
        Return total number of nodes in the cluster
        """
        info = self.get_info()
        count = 0
        for v in info.values():
            count += v[0]

        return count

    def get_info(self) -> Optional[pd.DataFrame]:
        """
        Return node info as a dictionary
        {
            state: [count, node_list]
        }
        """

        output = utils.get_bash_output("sinfo --format=%t\t%D\t%N")

        if output is None:
            raise RuntimeError("Unable to get SLURM info")

        return utils.bash_output_to_table(output)

    def get_nodes_by_state(self, state: str) -> List[str]:
        """
        Get list of nodes given state
        """
        df = self.get_info()

        try:
            node_list_expression = df[df.STATE == state].NODELIST.item()
            node_list_expression = node_list_expression.replace(
                f"{self.host_prefix}-", "")

            node_list = utils.list_expression_to_list(node_list_expression)

            return node_list
        except Exception as e:
            print(e)
            return None

    def get_queue_summary(self):
        q = self.get_queue()
        q_summary = q.groupby(['ACCOUNT', 'ST']).aggregate(
            {'NODES': 'sum'}).reset_index()

        return q_summary

    def get_recent_queue(self, account=None):
        q = self.get_queue()
        q = q[q.ST == 'R']
        q['TIME_SECS'] = q['TIME'].apply(
            lambda x: utils.get_elasped_seconds(x))

        if account is not None:
            q = q[q['ACCOUNT'] == account]

        q = q.sort_values(by='TIME_SECS', ascending=True)

        return q

    def get_queue(self) -> Optional[pd.DataFrame]:
        """
        Get SLURM queue into a dictionary
        """

        output = utils.get_bash_output(
            "squeue --format=%i\t%a\t%j\t%u\t%t\t%S\t%M\t%D")
        if output is None:
            raise RuntimeError("Unable to get SLURM queue")

        queue = utils.bash_output_to_table(output)
        queue['NODES'] = queue['NODES'].astype(int)

        return queue

    def get_job_hist_json(self, start_date: str) -> List[Dict[str, Any]]:
        """
        Get Slurm job history since a start_date (e.g., "2024-07-01")
        """
        cmd = f"sacct -a --starttime {start_date} --json"
        output = utils.get_bash_output(cmd)

        output_dict = json.loads(output)

        jobs = output_dict["jobs"]

        return jobs

    def get_job_hist_table(self, start_date: str) -> Optional[pd.DataFrame]:
        """
        Get Slurm job history since a start_date (e.g., "2024-07-01")
        """

        # run bash command to get job history output, then convert to df
        bash_command = f"sacct -a --starttime {start_date} --format=User,JobID,Jobname%50,partition,state,time,start,end,elapsed,nnodes,ncpus,nodelist"
        output = utils.get_bash_output(bash_command)
        lines = output.split("\n")

        header = lines[0].split()
        num_cols = len(header)

        data = [line.split() for line in lines[2:-1]]

        data = [row for row in data if len(row) == num_cols]

        df = pd.DataFrame(columns=header, data=data)

        # post processing
        df['NNodes'] = df['NNodes'].astype(int)
        df['NCPUS'] = df['NCPUS'].astype(int)
        df['Elapsed_hr'] = df['Elapsed'].map(utils.get_elapsed_hours)

        return df


class LogProcessor:
    @staticmethod
    def parse_tres_count(tres_list):
        tres = {}
        for item in tres_list:
            tres[item["type"]] = item["count"]

        return tres

    @staticmethod
    def parse_sbatch_config(sbatch_script_file: str, job_id: int):
        config = {}

        lines = utils.read_file_lines(sbatch_script_file)
        if lines is not None:
            for line in lines:
                line = line.strip()
                if line.startswith("#SBATCH"):
                    # ex: #SBATCH --job-name=vitg_2
                    line = line.replace("%j", str(job_id))
                    line = line.split()[-1]  # --job-name=vitg_2
                    line = line[2:]  # job-name=vitg_2
                    if '=' in line:
                        arg, value = line.split('=')
                        if value.isdigit():
                            value = int(value)

                        config[arg] = value

        return config

    @staticmethod
    def parse_job_error_log(log_file: str):
        """
        Parse error log file to identify error pattern and training progress
        Supported jobs:
            - vision encoder: /fsx_0/user/jiazhi/logs/
        """
        log_info = {
            "train_epoch": None,
            "found_error_patterns": []
        }

        # known error patterns
        error_patterns = [
            {
                "catch_phrase": "RendezvousTimeoutError",
                "note": "usually happens during launch, timeout after 10min"

            },
            {
                "catch_phrase": "NCCL  NET/OFI Request error",
                "note": ""
            },
            {
                "catch_phrase": "Watchdog caught collective operation timeout",
                "note": "followed by WorkNCCL..."
            },
            {
                "catch_phrase": "uncorrectable ECC error encountered",
                "note": "GPU error"
            }
        ]

        lines = utils.read_file_lines(log_file)
        if lines is not None:
            # read backward to find max train epoch reached
            for line in lines[::-1]:
                match = re.search(r"Train Epoch: (\d+)", line)
                if match:
                    log_info['train_epoch'] = int(match.group(1))
                    break

            # look for error patterns
            for line in lines:
                for p in error_patterns:
                    if (p["catch_phrase"] not in log_info["found_error_patterns"]):
                        if re.search(p["catch_phrase"], line):
                            log_info["found_error_patterns"].append(
                                p["catch_phrase"])

        return log_info

    @staticmethod
    def parse_job_info(info):
        """
        parse job info json
        """

        if not info['submit_line'].startswith("sbatch"):
            return

        if info['qos'] != 'ar-ai-hipri':
            return

        allocated_tres = LogProcessor.parse_tres_count(
            info["tres"]["allocated"])
        node_count = allocated_tres.get("node", None)
        if node_count is None or node_count < 10:
            return

        user = info['association']['user']
        job_id = info['job_id']

        if user not in ('jiazhi', 'tranx', 'qyh', 'samueldo'):
            return

        state = info['state']['current']
        if state == "CANCELLED":
            return

        # process sbatch file
        sbatch_file = info['submit_line'].split()[-1]
        sbatch_file = os.path.join(info['working_directory'], sbatch_file)
        config = LogProcessor.parse_sbatch_config(sbatch_file, info['job_id'])

        # print('-------')
        # print("job_id:", job_id)
        # print("node_count:", node_count)
        # print("user:", user)
        # print("config:", config)

        log_info = None
        if 'error' in config and os.path.exists(config['error']):
            log_info = LogProcessor.parse_job_error_log(config['error'])

        output = {
            "job_id": job_id,
            "node_count": node_count,
            "user": user,
            "state": state,
            "log_info": log_info,
            "start_time": utils.timestamp_to_str(info['time']['start']),
            "end_time": utils.timestamp_to_str(info['time']['end']) if state != "RUNNING" else None,
            "elapsed_seconds": info['time']['elapsed'],
            "sbatch_config": config
        }

        print('-------')
        pprint(output)

        return output
