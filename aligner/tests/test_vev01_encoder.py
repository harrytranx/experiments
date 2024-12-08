import logging
import os
import sys
# Set environment variables
os.environ["XLFORMERS"] = os.path.expanduser("~/rsc/aligner/xlformers")
os.environ["CRYPTOGRAPHY_OPENSSL_NO_LEGACY"] = "1"
# Modify PYTHONPATH
xlformers_path = os.environ["XLFORMERS"]
finetune_packages_path = os.path.join(xlformers_path, "src", "finetune", "packages", "llm_common")
sys.path.insert(0, xlformers_path)
sys.path.insert(0, finetune_packages_path)
import unittest

import timm
import torch
from PIL import Image
from torchvision import transforms


from src.reloading import init_distributed_mode
from src.multimodal.vision_models.metaclip_vev01_fixed_res import METACLIP_VEV01
# from src.metaclip_tmp.transformer import VisionTransformer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


CURR_DIR = os.path.dirname(os.path.abspath(__file__))

VEV0_BASE_PATH = "/checkpoint/onevision/aligner_models/vev0_1/vev0_1_lp14_336_36ep_final_vision.pt"
VEV0_XLF_PATH = "/checkpoint/onevision/aligner_models/vev0_1/vev0_1_lp14_336_36ep_final_vision.pt"
TEST_IMAGE = f"/checkpoint/onevision/datasets_eval/coco/val2014/COCO_val2014_000000485307.jpg"


metaclip_config = {
    'image_size': 336,
    'patch_size': 14,
    'width': 1024,
    'layers': 32,
    'heads': 16,
    'mlp_ratio': 4.625,
    'global_layers': -1,
    'pos_embed_type': "learnable",
    'relative_pos_embed_type': 'rope_2d',
    'output_dim': 1024,
    'embed_cls_token': True,
    'output_tokens': True,
    'disable_ln_post': True,
}

xlformers_config = {
    'ckpt_path': VEV0_XLF_PATH,
    'image_size': 336, 
    'patch_size': 14,
    'width': 1024,
    'layers': 32,
    'heads': 16,
    'mlp_ratio': 4.625,
    'load_ckpt': True,
}

CLIP_MEAN = (0.5, 0.5, 0.5)
CLIP_STD = (0.5, 0.5, 0.5)

class TestvevoModel(unittest.TestCase):
    def setUp(self):
        init_distributed_mode(1)
        
        # self.base_model = VisionTransformer(**metaclip_config)
        # self.base_model.cuda().eval()
        # logger.info("Loaded vev0_1 metaclip format model.")
        # reuse_ckpt = torch.load(VEV0_BASE_PATH, map_location="cpu")
        # mm, uu = self.base_model.load_state_dict(reuse_ckpt, strict=False)
        # print("missed param:", mm)
        # print("unused param:", uu)
        # print("\n")

        self.xlf_model = METACLIP_VEV01(**xlformers_config)
        self.xlf_model.cuda().eval()
        logger.info("Loaded vev01 fixed res model.")
        print("\n")

        # self.xlf_model_navit = METACLIP_NAVIT(**xlformers_config)
        # self.xlf_model_navit.cuda().eval()
        # logger.info("Loaded vev0 any res model.")

    def preprocess_image_base(self, image_path):
        preprocess = transforms.Compose(
            [
                transforms.Resize(metaclip_config['image_size']),
                transforms.CenterCrop(metaclip_config['image_size']),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=CLIP_MEAN, std=CLIP_STD
                ),
            ]
        )
        image = Image.open(image_path).convert("RGB")
        return preprocess(image).unsqueeze(0)

    def preprocess_image_xlformers(self, image_path):
        preprocess = transforms.Compose(
            [
                transforms.Resize(metaclip_config['image_size']),
                transforms.CenterCrop(metaclip_config['image_size']),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=CLIP_MEAN, std=CLIP_STD
                ),
            ]
        )
        image = Image.open(image_path).convert("RGB")
        return preprocess(image).unsqueeze(0)

    def pack_window(self, img, packed_dim=7):
        patch_h = patch_w = metaclip_config['patch_size']
        _, channel, image_h, image_w = img.shape
        img = img.unfold(-2, patch_h, patch_h).unfold(-2, patch_w, patch_w)

        idx_h, idx_w = image_h // patch_h, image_w // patch_w
        img_idx = torch.arange(image_h * image_w // (patch_h * patch_w), dtype=torch.int32)
        img_idx = img_idx.reshape(idx_h * idx_w, 1)

        img = img.reshape(channel, -1, 1, patch_h, patch_w)
        img_idx = img_idx.reshape(-1, 1)

        img = torch.cat([img, img[:, :1]], dim=1)
        img_idx = torch.cat([img_idx, img_idx[:1]], dim=0)
        img_idx[-1, -1] = PACKED.ID_CLS_TOKEN

        packed_img_idx = torch.empty(img_idx.shape[0], img_idx.shape[1], packed_dim, dtype=torch.int32)

        packed_img_idx[:, :, PACKED.Y] = img_idx // idx_w
        packed_img_idx[:, :, PACKED.X] = img_idx % idx_w
        packed_img_idx[:, :, PACKED.IDX] = img_idx

        return img, packed_img_idx

    def test_forward_parity(self):
        image_tensor_base = self.preprocess_image_base(TEST_IMAGE).cuda().repeat(1, 1, 1, 1)
        image_tensor_xlf = self.preprocess_image_xlformers(TEST_IMAGE)
        
        """
        test metaclip packed input and metaclip normal input:
        """
        # packed_img, packed_img_idx = self.pack_window(image_tensor_xlf, packed_dim=PACKED.NUM_METADATA)
        # # fake collator
        # images = [packed_img, packed_img]
        # packed_img_idxs = [packed_img_idx, packed_img_idx]
        
        # num_windows = torch.tensor([packed_img.shape[1] for packed_img in images]).long()
        # packed_num_windows, packed_counts = torch.unique(num_windows, return_counts=True)
        # packed_end_idx = (packed_counts * packed_num_windows).cumsum(dim=0)
        # packing_boundaries = [packed_num_windows.tolist(), packed_end_idx.tolist()]

        # packed_img_idxs = torch.cat(packed_img_idxs, dim=0)
        # images = torch.cat(images, dim=1).permute(1, 0, 2, 3, 4).contiguous()
        # images_package = [images.cuda(), packed_img_idxs.cuda(), num_windows.cuda(), packing_boundaries]

        # with torch.no_grad():
        #     base_features = self.base_model(image_tensor_base).tokens
        #     xlf_features = self.base_model(images_package).tokens


        """
        test metaclip normal input and xlformers fixed res model:
        """
        image_tensor_xlf = image_tensor_xlf.cuda().repeat(1, 1, 1, 1)
        with torch.no_grad():
            # base_features = self.base_model(image_tensor_base).tokens
            # base_features = base_features.reshape(1, -1, base_features.shape[-1])[:1,:-1,:]
            xlf_features = self.xlf_model(image_tensor_xlf)
            base_features = xlf_features

        
        """
        test metaclip normal input and xlformers fixed res model:
        """
        # image_tensor_xlf = image_tensor_xlf.cuda().repeat(1, 1, 1, 1)
        # with torch.no_grad():
        #     base_features = self.base_model(image_tensor_base)
        #     # xlf_features = self.xlf_model(image_tensor_xlf)
        #     xlf_features = base_features[:1]


        """
        test metaclip normal input and xlformers any res model:
        """
        # packed_img, packed_img_idx = self.pack_window(image_tensor_xlf, packed_dim=PACKED.NUM_METADATA)
        # # fake collator
        # images = [packed_img, packed_img]
        # packed_img_idxs = [packed_img_idx, packed_img_idx]
        
        # num_windows = torch.tensor([packed_img.shape[1] for packed_img in images]).long()
        # packed_img_idxs = torch.cat(packed_img_idxs, dim=0)
        # images = torch.cat(images, dim=1).permute(1, 0, 2, 3, 4).contiguous()
        # images_package = [images.cuda(), packed_img_idxs.cuda(), num_windows.cuda()]
        
        # with torch.no_grad():
        #     base_features = self.base_model(image_tensor_base).tokens
        #     base_features = base_features.reshape(2, -1, base_features.shape[-1])[:,:-1,:]
        #     xlf_features = self.xlf_model_navit(images_package, None)
        #     xlf_features = torch.stack(xlf_features, dim=0)
        
        logger.info(f"base_features shape: {base_features.shape}")
        logger.info(f"xlf_features shape: {xlf_features.shape}")

        logger.info(f"base features norm: {torch.linalg.norm(base_features, dim=-1).mean()}")
        logger.info(f"xlf features norm: {torch.linalg.norm(xlf_features, dim=-1).mean()}")
        
        self.assertEqual(base_features.shape, xlf_features.shape)
        norm_gap = (
            (
                torch.linalg.norm(base_features - xlf_features, dim=-1)
                / torch.linalg.norm(base_features, dim=-1)
            ).mean(),
            torch.linalg.norm(base_features, dim=-1).mean(),
            torch.linalg.norm(xlf_features, dim=-1).mean(),
        )
        logger.info(f"Norm gap between base and xlf: {norm_gap}")


if __name__ == "__main__":
    unittest.main()