import os
from datasets import load_dataset, DatasetInfo, IterableDataset
from torch.utils.data import DataLoader, IterableDataset
from tqdm import tqdm
import json
from PIL import Image
import io
import math 

from typing import Any, List, Tuple, Dict
import webdataset_helper
import multiprocessing

class HFDataset():
    def __init__(
        self, 
        data_dir: str, 
        split: str = "train"
    ):
        self.data_dir = data_dir
        self.info = load_dataset(data_dir, split=split, streaming=False).info
        self.num_examples = self.info.splits[split].num_examples
        
    
    def load(self, split: str= "train", streaming: bool = True):
        # streaming does not support num_proc
        
        return load_dataset(self.data_dir, split=split, streaming=streaming)
    
    def load_slice(self, start_index: int, end_index: int, num_proc: int = 64, split: str = "train"):
        slice = load_dataset(self.data_dir, num_proc=num_proc, split=f"{split}[{start_index}:{end_index}]")
        
        return slice
    
def run_multiprocessing(target_function, args_list, processes=None):
    """
    Run a target function using multiprocessing with a list of arguments.
    
    Parameters:
        target_function (callable): The function to execute in parallel.
        args_list (list of tuple): A list of arguments (tuples) to pass to the target function.
        processes (int, optional): Number of worker processes to use. Defaults to the number of CPU cores.
    
    Returns:
        list: A list of results from the target function.
    """
    with multiprocessing.Pool(processes=processes) as pool:
        # Using starmap to unpack tuples in args_list
        results = pool.starmap(target_function, args_list)
    return results

def count_subfolders(folder_path):
    """
    Count the number of level-1 (immediate) subfolders in a given folder.

    Parameters:
        folder_path (str): Path to the folder.

    Returns:
        int: Number of level-1 subfolders.
    """
    try:
        # List only directories in the specified folder
        level1_subfolders = [entry for entry in os.listdir(folder_path) 
                             if os.path.isdir(os.path.join(folder_path, entry))]
        return len(level1_subfolders)
    except FileNotFoundError:
        print(f"Error: The folder '{folder_path}' does not exist.")
        return 0
    except PermissionError:
        print(f"Error: Permission denied to access '{folder_path}'.")
        return 0
    
def count_files_recursively(folder_path):
    """
    Count the total number of files in a folder, including all subfolders, recursively.

    Parameters:
        folder_path (str): Path to the folder.

    Returns:
        int: Total number of files in the folder and its subfolders.
    """
    try:
        # Walk through the directory and count files
        total_files = sum(len(files) for _, _, files in os.walk(folder_path))
        return total_files
    except FileNotFoundError:
        print(f"Error: The folder '{folder_path}' does not exist.")
        return 0
    except PermissionError:
        print(f"Error: Permission denied to access '{folder_path}'.")
        return 0

def image_to_bytesio(image: dict[str, Any]):
    try:
        return io.BytesIO(image['bytes'])
    except Exception:
        return None

def sanitize(input: str):
    # making sure that tags with \n go to front of the list
    image_tags = ["<img>\n", "<image>\n"] + ["<img>", "<image>"]
    
    for tag in image_tags:
        input = input.replace(tag, "")
    
    return input

def generate_qa_samples(sample: dict[str, Any]) -> List[Tuple[Dict[str, Any], io.BytesIO]]:
    """
    Generate question-answer pairs
    Return list of [(json_data, image buffer), (json_data, image buffer), ...]
    """
    qa_samples = []
    
    turns = len(sample['conversations']) // 2
    
    for i in range(turns):
        human = sample['conversations'][2*i]
        gpt = sample['conversations'][2*i+1]
        
        assert human['from'] == 'human'
        assert gpt['from'] == 'gpt'
        
        json_data = {
            'id': sample['id'],
            'data_source': sample['data_source'],
            "question": sanitize(human['value']),
            "response": sanitize(gpt['value'])
        }
    
        qa_samples.append((json_data, sample['image']))
    
    return qa_samples

def process_slice(dataset, slice_id, slice_size, samples_per_tar, output_dir):
    ds_slice = dataset.load_slice(
        start_index = slice_id*slice_size,
        end_index = (slice_id + 1)*slice_size
    )

    df = ds_slice.to_pandas()
    # df['image'] = df['image'].apply(lambda img: io.BytesIO(img['bytes']))
    df['image'] = df['image'].apply(lambda img: image_to_bytesio(img))
    
    qa_results = df.apply(generate_qa_samples, axis=1).to_list()
    qa_results = [item for sublist in qa_results for item in sublist]
    
    print(f"num samples = {len(df)}, num qa samples = {len(qa_results)}")   
    
    slice_output_dir = os.path.join(
        output_dir, 
        webdataset_helper.num_to_str_id(slice_id, 4)
    )
    os.makedirs(slice_output_dir, exist_ok=True)
    
    num_tars = math.ceil(len(qa_results) / samples_per_tar)
    slice_stats = {'num_tars': num_tars, "provided_samples": 0, "valid_samples": 0}
    
    # for i in range(num_tars):
    for i in tqdm(range(num_tars)):
        tar_id = webdataset_helper.num_to_str_id(i, str_len=5)

        samples = qa_results[i*samples_per_tar:(i+1)*samples_per_tar]
        json_data = [sample[0] for sample in samples]
        image_handles = [sample[1] for sample in samples]
        
        # print(f"{tar_id=}, {len(samples)=}")
        wds = webdataset_helper.Webdataset(json_data, image_handles)
        tar_file, stats = wds.to_file(
            output_tar_file=os.path.join(slice_output_dir, f"{tar_id}.tar"),
            sample_prefix=tar_id + "_",
            progress_bar=False
        )
        
        slice_stats['provided_samples'] += stats['provided_samples']
        slice_stats['valid_samples'] += stats['valid_samples']
    
    return slice_stats

def process_dataset(data_path, output_dir, num_slices=100, samples_per_tar=1024):
    
    print(f"Initializing HFDataset from {data_path}")
    ds = HFDataset(data_path)
    print(f"{ds.num_examples=}")
    
    slice_size = math.ceil(ds.num_examples / num_slices)
    
    args_list = []
    for slice_id in range(num_slices):
        args = (
            ds,
            slice_id,
            slice_size,
            samples_per_tar,
            output_dir
        )
        
        args_list.append(args)
        
    all_stats = run_multiprocessing(process_slice, args_list, processes=num_slices)

    print("Completed.")
    print(f"Number of subfolders = {count_subfolders(output_dir)}")
    print(f"Number of tar files = {count_files_recursively(output_dir)}")
    
    stats = {}
    for key in ['num_tars', 'provided_samples', 'valid_samples']:
        stats[key] = sum([s[key] for s in all_stats])
    
    return stats