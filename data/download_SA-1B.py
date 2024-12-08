import pandas as pd
import os
import requests
from io import BytesIO
import multiprocessing

def download_buffer(url, name):
    response = requests.get(url)

    if response.status_code == 200:
        bytes_io = BytesIO()
        bytes_io.write(response.content)
        bytes_io.seek(0)
        print(f"File downloaded successfully: {name}")
    else:
        print(f"Failed to download file. Status code: {response.status_code}: {name}")
        
        
def download_file(url, name, dir_name):
    
    file_name = os.path.join(dir_name, name)
    file_tmp = f"{file_name}.tmp"
    
    if os.path.exists(file_name):
        print(f"Found existing file {file_name}, skip downloading")
        return 
    elif os.path.exists(file_tmp):
        print(f"Removing partially downloaded file: {file_tmp}")
        os.remove(file_tmp)
    
    try:
        response = requests.get(url, stream=True)
        if response.status_code == 200:
            with open(file_tmp, "wb") as file:
                for chunk in response.iter_content(chunk_size=8192):
                    file.write(chunk)
            
            if os.path.exists(file_tmp):
                os.rename(file_tmp, file_name)
                print(f"Successfuly downloaded: {file_name}")
            else:
                print(f"Failed to find {file_tmp} after download")
        else:
            print(f"Failed to download file with tatus code: {response.status_code}: {file_name}")
    except Exception as e:
        print(e)
        # clean up to make sure file is removed if error happens
        if os.path.exists(file_name):
            os.remove(file_name)
        if os.path.exists(file_tmp):
            os.remove(file_tmp)
        
        
def start_multiprocess(worker_function, args_list):
    processes = []
    
    for args in args_list:
        process = multiprocessing.Process(target=worker_function, args=args)
        processes.append(process)
        process.start()

    for process in processes:
        process.join()
        
    print("All processes have completed.")
    
    
def main():
    print("Loading urls list")
    DIR = "/fsx_3/dataset01/SA-1B"
    df = pd.read_csv(f"{DIR}/urls.csv", delimiter="\t")
    df = df.sort_values(by=['file_name']).reset_index()
    
    print(df)
    
    download_dir = os.path.join(DIR, "raw")
    args = []
    max_workers = 100
    for i, row in df.iterrows():
        args.append((
            row.cdn_link,
            row.file_name,
            download_dir
        ))

        if len(args) == max_workers:
            start_multiprocess(
                worker_function=download_file,
                args_list=args
            )
            args = []
            
    start_multiprocess(
        worker_function=download_file,
        args_list=args
    )
                
if __name__ == "__main__":
    main()