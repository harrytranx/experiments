import json
import os
import pickle
import time
import unittest
from itertools import chain
from typing import Any
from unittest.case import TestCase

import torch
import torch.multiprocessing as mp

# @manual=fbsource//third-party/pypi/webdataset:webdataset
import webdataset as wds

from llm_mm_aligner.lib.configs import (
    DataTrainingArguments,
    ModelArguments,
    TrainingArguments,
)
from llm_mm_aligner.lib.data_collators import get_collator
from llm_mm_aligner.lib.datasets.manager import get_dataset
from llm_mm_aligner.lib.datasets.web_dataset import get_wb_dataset
from llm_mm_aligner.lib.platform import platform, platform_type
from torch.distributed import destroy_process_group, init_process_group

from torch.utils.data import DataLoader, IterableDataset

# @manual=fbsource//third-party/pypi/transformers:transformers
from transformers import HfArgumentParser


def print_green(text):
    green_color = "\033[92m"  # bright green
    reset_color = "\033[0m"  # Reset the color to default terminal color

    print(f"{green_color}{text}{reset_color}")


def get_args_list(args):
    """
    Copied from https://fburl.com/code/3pq3dn99
    Convert a dict of args to a list of strings for passing to the binary
    """
    return list(
        chain.from_iterable(
            [f"--{k}", str(v)] if v is not None else [f"--{k}"] for k, v in args.items()
        )
    )


def get_local_test_artifacts():
    """
    Getting test artifacts from manifold or AWS fsx
    """
    if platform.get_platform_type() == platform_type.PlatformType.META:
        print_green("Running on Meta")
        from iopath.common.file_io import PathManager
        from iopath.fb.manifold import ManifoldPathHandler

        pathmgr = PathManager()
        pathmgr.register_handler(ManifoldPathHandler())
        TEST_PATH = "manifold://sg_scene_ai/tree/llm_mm_aligner/tests/tranx/train_8b_mh"

        tokenizer_file = pathmgr.get_local_path(
            os.path.join(TEST_PATH, "HFMetaFormerTokenizer.pkl")
        )

        preprocessor_file = pathmgr.get_local_path(
            os.path.join(TEST_PATH, "LlavaNextImageProcessor.pkl")
        )

        params_file = "/data/sandcastle/boxes/fbsource/fbcode/fblearner/flow/projects/assistant/multimodal/llm_mm_aligner/experiments/stage1/train_8b_mh.json"

        data_path = pathmgr.get_local_path(
            os.path.join(TEST_PATH, "wds_data/"))

    elif platform.get_platform_type() == platform_type.PlatformType.AWS:
        print_green("Running on AWS")
        # TEST_PATH = "/fsx_0/user/tranx/test_data/llm_mm_aligner"

        # tokenizer_file = os.path.join(TEST_PATH, "HFMetaFormerTokenizer.pkl")

        # preprocessor_file = os.path.join(
        #     TEST_PATH, "LlavaNextImageProcessor.pkl")
        # params_file = os.path.join(TEST_PATH, "train_8b_mh.json")

        # data_path = os.path.join(TEST_PATH, "wds_data/")
        # data_path = "/fsx_1/datasets_30days/sg_vision_encoder_clip_filtered03_m2c2_metaclip_sstk_ocr_0710/20240710/0000"

        tokenizer_file = "/fsx_0/checkpoints/tranx/Aligner-Pretrain-8B/resume_fb_8B/HFMetaFormerTokenizer.pkl"
        preprocessor_file = "/fsx_0/checkpoints/tranx/Aligner-Pretrain-8B/resume_fb_8B/LlavaNextImageProcessor.pkl"
        params_file = "/fsx_0/user/tranx/llm_mm_aligner/experiments/aws_tranx/mm9_stage1/pretrain_MH_8B_resume.json"
        data_path = "/fsx_1/datasets_30days/sg_mmllm_stage1_m2c2v3_sstk_10x_arxiv_pdf_mix_v6/20240723"

    else:
        raise Exception(
            f"Unsupported platform type: {platform.get_platform_type()}. Only Meta and AWS are supported."
        )

    with open(params_file, "r") as f:
        params = json.load(f)

    with open(tokenizer_file, "rb") as f:
        tokenizer = pickle.load(f)

    with open(preprocessor_file, "rb") as f:
        preprocessor = pickle.load(f)

    trainer_args = params.get("trainer_args", None)
    parser = HfArgumentParser(
        (ModelArguments, DataTrainingArguments, TrainingArguments)
    )

    model_args, data_args, training_args = parser.parse_args_into_dataclasses(
        args=get_args_list(trainer_args)
    )

    print_green(f"data_path: {data_path}")

    data_args.use_hive_dataset = False

    data_args.wd_data_path = data_path
    data_args.wd_chunk_size = 1000

    return model_args, data_args, training_args, tokenizer, preprocessor


def data_loader_process(
    rank: int,
    world_size: int,
    model_args: ModelArguments,
    data_args: DataTrainingArguments,
    training_args: TrainingArguments,
    tokenizer: Any,
    preprocessor: Any,
):
    # setup DDP
    print_green("Setup DDP Process Group")
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    init_process_group(backend="nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

    print_green(
        f"rank: {rank}, world_size: {world_size}, device ID: {torch.cuda.current_device()}"
    )

    train_dataset = get_wb_dataset(
        preprocessor=preprocessor, model_args=model_args, data_args=data_args, training_args=training_args
    )

    data_collator = get_collator(data_args, model_args, tokenizer)

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=32,  # training_args.per_device_train_batch_size
        collate_fn=data_collator,
        num_workers=0,
        pin_memory=training_args.dataloader_pin_memory,  # False
    )

    try:
        print_green(
            f"rank: {rank}, number of batches = len(train_dataloader) = {len(train_dataloader)}"
        )
    except Exception as e:
        print_green(f"rank: {rank}, exception = {e}")

    # iterate through the entire dataset on each worker
    start = time.time()
    for step, _ in enumerate(train_dataloader):
        load_time = time.time() - start

        print_green(
            f"rank: {rank}, world_size: {world_size}, step: {step}, load_time: {load_time} seconds"
        )
        start = time.time()

    """
    rank: 2, world_size: 4, step: 42, load_time: 3.649968147277832 seconds
    rank: 3, world_size: 4, step: 17, load_time: 9.97529649734497 seconds
    rank: 0, world_size: 4, step: 41, load_time: 3.839179277420044 seconds
    rank: 1, world_size: 4, step: 78, load_time: 8.168761730194092 seconds
    rank: 2, world_size: 4, step: 43, load_time: 3.717665910720825 seconds
    rank: 0, world_size: 4, step: 42, load_time: 2.121644973754883 seconds
    rank: 0, world_size: 4, step: 43, load_time: 3.622596502304077 seconds
    rank: 2, world_size: 4, step: 44, load_time: 5.7756218910217285 seconds
    rank: 1, world_size: 4, step: 79, load_time: 6.368230581283569 seconds
    rank: 0, world_size: 4, step: 44, load_time: 3.2522003650665283 seconds
    rank: 3, world_size: 4, step: 18, load_time: 12.100693464279175 seconds
    rank: 2, world_size: 4, step: 45, load_time: 5.8367249965667725 seconds
    rank: 1, world_size: 4, step: 80, load_time: 7.276073217391968 seconds
    rank: 0, world_size: 4, step: 45, load_time: 5.66037392616272 seconds
    rank: 2, world_size: 4, step: 46, load_time: 5.108770370483398 seconds
    """
    # destroy ddp
    print_green("Destroy Process Group")
    destroy_process_group()


class TestWebDatasetLocal(TestCase):
    """
    buck2 run @//mode/opt @mode/inplace assistant/multimodal/llm_mm_aligner/lib/datasets/web_dataset/tests:test_webdataset
    """

    @unittest.skipIf(not torch.cuda.is_available(), "Skip when CUDA is not available")
    def test_01_instantiation(self):

        model_args, data_args, training_args, tokenizer, preprocessor = (
            get_local_test_artifacts()
        )

        # instantiate train_dataset
        train_dataset = get_wb_dataset(
            preprocessor=preprocessor, 
            model_args=model_args, 
            data_args=data_args, 
            training_args=training_args
        )

        # WebDataset is also an IterableDataset
        print_green(f"type of train_dataset: {type(train_dataset)}")
        print_green(
            f"isinstance(train_dataset, WebDataset): {isinstance(train_dataset, wds.WebDataset)}"
        )
        print_green(
            f"isinstance(train_dataset, IterableDataset): {isinstance(train_dataset, IterableDataset)}"
        )

        assert isinstance(train_dataset, IterableDataset)

    @unittest.skipIf(not torch.cuda.is_available(), "Skip when CUDA is not available")
    def test_02_dataloader_single(self):
        model_args, data_args, training_args, tokenizer, preprocessor = (
            get_local_test_artifacts()
        )

        train_dataset = get_wb_dataset(
            preprocessor=preprocessor, model_args=model_args, data_args=data_args, training_args=training_args
        )

        data_collator = get_collator(data_args, model_args, tokenizer)

        train_dataloader = DataLoader(
            train_dataset,
            batch_size=32,  # training_args.per_device_train_batch_size
            collate_fn=data_collator,
            num_workers=0,
            pin_memory=training_args.dataloader_pin_memory,  # False
        )

        print_green(
            f"number of steps = len(train_dataloader): {len(train_dataloader)}")
        # assert (
        #     len(train_dataloader) == 125
        # )  # 4 shards x 1000 samples / batch_size 32 = 125 steps

        batch = next(iter(train_dataloader))
        print_green(f"batch.keys: {batch.keys()}")

        assert "modality" in batch.keys()
        assert "image_sizes" in batch.keys()
        assert "chunk_ids" in batch.keys()
        assert "eoi_ids" in batch.keys()
        assert "num_chunks" in batch.keys()
        # assert "image_pos" in batch.keys()
        assert "input_ids" in batch.keys()
        assert "labels" in batch.keys()

        print_green(f"batch['modality']: {batch['modality'].shape}")

        # try iterating on few batches
        start = time.time()
        for step, batch in enumerate(train_dataloader):
            load_time = time.time() - start
            print_green(
                f"step={step}: modality.shape={batch['modality'].shape}, load_time={load_time} seconds"
            )
            start = time.time()
            if step == 10:
                break

    @unittest.skipIf(not torch.cuda.is_available(), "Skip when CUDA is not available")
    def test_03_dataloader_with_ddp(self):
        model_args, data_args, training_args, tokenizer, preprocessor = (
            get_local_test_artifacts()
        )

        num_gpus = min(8, torch.cuda.device_count())
        print_green(f"Number of GPUs: {num_gpus}")

        mp.spawn(
            data_loader_process,
            args=(
                num_gpus,
                model_args,
                data_args,
                training_args,
                tokenizer,
                preprocessor,
            ),
            nprocs=num_gpus,
        )

    @unittest.skipIf(not torch.cuda.is_available(), "Skip when CUDA is not available")
    def test_04_train_dataloader(self):
        model_args, data_args, training_args, tokenizer, preprocessor = (
            get_local_test_artifacts()
        )

        # following main.py (https://fburl.com/code/b2qe3xdf)
        train_dataset, eval_dataset = get_dataset(
            preprocessor=preprocessor,
            data_args=data_args,
            training_args=training_args,
            model_args=model_args,
        )

        data_collator = get_collator(data_args, model_args, tokenizer)

        print_green(f"data_args.task_type: {data_args.task_type}")
        print_green(f"model_args.model_type: {model_args.model_type}")
        print_green(f"training_args.fsdp: {training_args.fsdp}")

        # following base_trainer.py (https://fburl.com/code/mz2tgvgs)
        print_green(f"type(train_dataset): {type(train_dataset)}")
        print_green(
            f"isinstance(train_dataset, IterableDataset): {isinstance(train_dataset, IterableDataset)}"
        )
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=32,
            collate_fn=data_collator,
            num_workers=0,
            pin_memory=training_args.dataloader_pin_memory,
        )

        batch = next(iter(train_dataloader))
        print_green(f"batch.keys: {batch.keys()}")


if __name__ == "__main__":
    unittest.main(failfast=True, exit=False)
