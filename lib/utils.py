import subprocess
from typing import Dict, Any, List, TextIO
import pandas as pd
from datetime import datetime
import os
import pytz
import json
from collections import OrderedDict
import sys 

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



def timestamp_to_str(timestamp: int):
    """
    Convert timestamp to datetime string
    Ex: 1721055801 -> 2024-07-15 15:03:21
    """
    dt_object = datetime.fromtimestamp(timestamp)
    dt_str = dt_object.strftime('%Y-%m-%d %H:%M:%S')

    return dt_str


def get_local_time(zone='America/New_York'):
    """
    'America/New_York', 'US/Pacific'
    """

    timestamp = datetime.now(pytz.timezone(zone))
    timestamp = timestamp.strftime('%Y-%m-%d %H:%M:%S')

    return timestamp


def read_file_lines(file_path: str):
    try:
        with open(file_path, 'r') as f:
            lines = f.readlines()

        return lines
    except Exception:
        return None


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


def bash_output_to_table(output: str) -> pd.DataFrame:
    """
    Convert bash output data into pandas dataframe
    """

    lines = output.split("\n")
    header = lines[0].split("\t")

    data = [line.split("\t") for line in lines[1:-1]]
    df = pd.DataFrame(columns=header, data=data)

    return df


def get_elapsed_hours(elapsed_str: str) -> float:
    """
    Calculate number of hours from elapsed string
    Eg.g, 6-23:06:16, 23:06:16
    """

    if "-" in elapsed_str:
        day_substr, elapsed_str = elapsed_str.split("-")
        day = int(day_substr)
    else:
        day = 0

    hour_substr, min_substr, _ = elapsed_str.split(":")

    num_hours = 24*day + int(hour_substr) + round(int(min_substr)/60, 2)

    return num_hours


def get_elasped_seconds(elapsed_str: str) -> int:
    """
    Calculate number of seconds from elapsed string
    Eg.g, 6-23:06:16, 23:06:16
    """

    if "-" in elapsed_str:
        day_substr, elapsed_str = elapsed_str.split("-")
        day = int(day_substr)
    else:
        day = 0

    minutes = 0
    hours = 0
    if elapsed_str.count(":") == 0:
        seconds = int(elapsed_str)
    elif elapsed_str.count(":") == 1:
        minutes, seconds = elapsed_str.split(":")
        minutes = int(minutes)
        seconds = int(seconds)
    else:
        hours, minutes, seconds = elapsed_str.split(":")
        minutes = int(minutes)
        seconds = int(seconds)
        hours = int(hours)

    seconds += 60*minutes + 86400*day + 3600*hours

    return seconds


def run_bash_command(cmd: str, print_output: bool = False, print_error: bool = True):
    """
    Run a Bash command and return the output or exit code
    """

    process = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
    output, error = process.communicate()

    if error:
        error = error.decode()
        if print_error:
            print("Error:", error)
    else:
        output = output.decode()
        if print_output:
            print(output)
        return output


def list_expression_to_list(list_expression: str) -> List[int]:
    """
    Convert list expression to regular list
    e.g., "[1,2,5-7]" --> [1,2,5,6,7]
    """
    output = []

    for item in list_expression[1:-1].split(","):
        if '-' in item:
            begin, end = item.split("-")
            begin = int(begin)
            end = int(end)
            item_list = [int(x) for x in range(begin, end + 1)]
            output.extend(item_list)

        elif item.isdigit():
            output.append(int(item))

        else:
            raise ValueError("Invalid list expression: {item}")

    return output


def save_plotly_to_html(fig, output_file, width='100%', height=700):

    html_string = fig.to_html(
        include_plotlyjs=True,  # include the Plotly.js library in the HTML
        default_width=width,  # set the default width of the plot to 100%
        default_height=height  # set the default height of the plot to 500 pixels
    )

    # save the HTML string to a file
    if not output_file.endswith('.html'):
        output_file += '.html'

    with open(output_file, 'w') as f:
        f.write(html_string)

    print(f"Saved figure to {output_file}")


def save_json(data, output_file):
    with open(output_file, 'w') as f:
        json.dump(data, f, indent=4)