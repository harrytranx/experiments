
import json
import tarfile
import numpy as np
import matplotlib.pyplot as plt
import random
import textwrap
from pathlib import Path
import pandas as pd
from PIL import Image
from io import BytesIO
import base64
from functools import partial

IMAGE_EXTENSIONS = {"JPG", "jpg", "png", "jpeg", "bmp", "tif", "tiff"}

def image_base64(img):
    with BytesIO() as buffer:
        img.save(buffer, 'jpeg')
        return base64.b64encode(buffer.getvalue()).decode()
    
    
def image_formatter(img, width: 500):
    return f'<img src="data:image/jpeg;base64,{image_base64(img)}" width="{width}">'


def text_formatter(value):
    """Format text with left alignment."""
    return f'<div style="text-align: left;">{value}</div>'


class WDSViewer:
    @staticmethod
    def view(path: str, num_samples=3, different_tars=False, img_width=300):

        df = WDSViewer._read_wds_samples(
            path,
            num_samples,
            different_tars
        )

        img_formatter = partial(image_formatter, width=img_width)
        
        return df.to_html(
            formatters={ 
                'image': img_formatter,  
                **{col: text_formatter for col in df.columns if col != 'image'}
            },
            escape=False 
        )
       
    @staticmethod 
    def _read_wds_samples(path:str, num_samples: int, different_tars: bool):
        all_tarpaths = list(Path(path).glob("**/*.tar"))
        
        if different_tars:
            tarfps = random.sample(all_tarpaths, min(num_samples, len(all_tarpaths)))
            tarfps_samples = [(tarfp, 1) for tarfp in tarfps]
        else:
        
            tarfps_samples = [(random.choice(all_tarpaths), num_samples)]
            
        # print(tarfps_samples)
        
        data = []
        
        for tarfp, num_samples_for_tar in tarfps_samples:
            with tarfile.open(tarfp, 'r') as tar:
                sampled_members = random.sample([member for member in tar.getmembers() if member.name.endswith(".json")],
                                                min(len(tar.getnames())//2, num_samples_for_tar))
                
                # print("here:", tarfp)
                
                for member in sampled_members:
                    # print("member.name:", member.name)
                    key, _ = member.name.split(".")

                    key = key.split("/")[-1]
                    # print("key", key)
                    
                    example = json.load(tar.extractfile(f"{key}.json"))   
                    
                    img_fn = next(
                        (
                            f"{key}.{ext}"
                            for ext in IMAGE_EXTENSIONS
                            if f"{key}.{ext}" in tar.getnames()
                        ),
                        None,
                    )
            
                    img = Image.open(tar.extractfile(img_fn)).convert("RGB")
                    img.load()
                    # example["__tarfile__"] = tarfp
                    # example["__key__"] = key
                    example["image"] = img
                    data.append(example)
                    
        cols = data[0].keys()
        data_dict = {col: [sample.get(col, None) for sample in data] for col in cols} 
        df = pd.DataFrame(data=data_dict)
        
        return df