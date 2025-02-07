# ==============================================================================
import os
import sys

user = os.environ.get("USER")

sys.path.append(
    f"/data/users/{user}/fbsource/fbcode/assistant/multimodal/xlformers_llama4"
)

# import_paths = [
#     # "/home/tranx/xlformers",
#     "/data/users/tranx/fbsource/fbcode/assistant/multimodal/xlformers_llama4"
# ]
# for path in import_paths:
#     if path not in sys.path:
#         sys.path.append(path)


# ==============================================================================

from core.model.encoders._image_adapter import build_vision_adapter
from core.model.encoders._image_encoder import (
    build_vision_encoder,
    ENCODER_CLASS_MAPPING,
    Llama4FlashVisionEncoder,
)

from core.model.encoders._image_utils import (
    expand_num_tokens_to_mult8,
    get_vision_encoder_config,
    resize_local_position_embedding,
    resize_local_position_embedding_no_cls_token,
)

from core.model.encoders.image import VisionEmbedding
from core.parallelism.tensor_parallel.random import (
    _MODEL_PARALLEL_RNG_TRACKER_NAME,
    CudaRNGStatesTracker,
)
from core.params.args import ImageEncoderArgs


def print_green(message):
    """
    Print the given message in green color.

    :param message: The message to print
    """
    green_color = "\033[92m"
    reset_color = "\033[0m"
    print(f"{green_color}{message}{reset_color}")


def main():
    # image_args_dict = {
    #     "enable_projection": True,
    #     "encoder_name": "llama4_flash_encoder",
    #     "encoder_params": None,
    #     "freeze_vision_encoder": True,
    #     "image_height": 336,
    #     "image_width": 336,
    #     "patch_height": 14,
    #     "patch_width": 14,
    #     "ps_ratio": 0.5,
    #     "recompute_transformer": True,
    #     "return_intermediate": None,
    #     # "use_cached_embeddings": true,
    #     "use_cached_embeddings": False,
    #     "use_dynamic_transform": True,
    #     "vision_adapter_type": "pixel_shuffle_mlp",
    #     "vision_encoder_ckpt_path": "/mnt/wsfuse/nextgen_mm/vision_encoders/llama4_flash_encoder_final_1023_ema",
    # }

    # state_tracker = CudaRNGStatesTracker()
    # state_tracker.add(_MODEL_PARALLEL_RNG_TRACKER_NAME, 123)
    # print_green(f"{state_tracker.states_=}")

    image_args_dict = {
        "enable_projection": True,
        "encoder_name": "llama4_flash_encoder_vev0p2_392",
        "encoder_params": None,
        "freeze_vision_encoder": True,
        "image_height": 392,
        "image_width": 392,
        "patch_height": 14,
        "patch_width": 14,
        "ps_ratio": 0.5,
        "recompute_transformer": True,
        "return_intermediate": None,
        "use_cached_embeddings": False,
        "use_dynamic_transform": True,
        "vision_adapter_type": "pixel_shuffle_mlp",
        "vision_encoder_ckpt_path": "/mnt/wsfuse/tranx/llama4_flash_vit_g14_392_fair_vev0_epoch_480",
    }

    img_args = ImageEncoderArgs(**image_args_dict)
    print_green(f"{img_args=}")

    vision_embedding = VisionEmbedding(img_args, init=None)

    print_green(f"{vision_embedding=}")
    print_green("Done")


if __name__ == "__main__":
    main()
