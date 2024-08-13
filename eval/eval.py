import argparse
import subprocess
import os
import re


def grep_files(directory, pattern):
    # List all files in the specified directory
    files = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]
    matched_files = []

    # Loop through files and run grep
    for file in files:
        try:
            # Run grep command
            result = subprocess.run(['grep', '-l', pattern, os.path.join(directory, file)], 
                                    text=True, capture_output=True, check=True)
            # If grep finds a match, it will return filenames
            if result.stdout:
                matched_files.append(result.stdout.strip())
        except subprocess.CalledProcessError:
            # This means grep did not find a match in the file
            continue
    
    return matched_files


def extract_values(filename):
    # Define the pattern to match the line and capture the required parts
    pattern = r"Writing (\S+) with value ([\d\.]+) to TensorBoard"
    
    # List to hold the extracted values
    extracted_values = []
    
    # Open the file and read line by line
    with open(filename, 'r') as file:
        for line in file:
            # Search for the pattern in each line
            match = re.search(pattern, line)
            if match:
                # Extract the path and the value
                path = match.group(1)
                value = float(match.group(2))
                extracted_values.append((path, value))
    
    return extracted_values

def main():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--directory", type=str)
    parser.add_argument("--checkpoint", type=str)

    args = parser.parse_args()

    directory = args.directory  # e.g. '/fsx_0/user/ahmadyan/logs/eval/mh19'
    checkpoint = args.checkpoint # e.g. 14000

    benchmarks=["ai2d", "vqa", "mmmu", "chartqa", "docvqa", "infographics", "infographics_w_ocr", "mathvista", "mmbench", "textvqa"]

    for benchmark in benchmarks:
        print("\n Evaluating", benchmark, checkpoint)
        matched_files = grep_files(directory, benchmark + "_" + checkpoint)    
        print(matched_files)

        for file in matched_files:
            values = extract_values(file)
            print(values)

if __name__ == "__main__":
    main()

