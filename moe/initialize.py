import os
import json
import torch

def initialize_moe(moe, candidates, num_layers: int = 4, num_experts: int = 4):
    """
    Initialize the MoE (Mixture of Experts) model from existing experts.

    It will assume the experts are already `packed`, i.e. they will have
    two tensors, `w12` and `w3`. The `w12` tensor will be used for the first two layers,'
    and the `w3` tensor will be used for the last two layers.

    Args:
        moe (pytorch state dictionary): The MoE model to be initialized.
        candidates (list): A list of candidate `statedicts` to be used for initialization.
                    len(candidates) = num_experts
        num_layers (int, optional): The number of layers in the MoE model. Defaults to 4.
        num_experts (int, optional): The number of experts in the MoE model. Defaults to 4.

    Returns:
        None
    """
    discarded_tensors = set()
    expert = candidates[0]

    moe_num_params = sum(p.numel() for p in moe.values())
    candidate_num_params = sum(p.numel() for p in expert.values())
    print(
        f"Number of parameters in MoE: {moe_num_params} ({moe_num_params//1e9} B), in candidate: {candidate_num_params} ({candidate_num_params//1e9} B)"
    )

    for key in expert.keys():
        if key in moe.keys() and expert[key].shape == moe[key].shape:
            print("matched keys:", key, expert[key].shape)
            moe[key] = expert[key].clone().detach()
        else:
            discarded_tensors.add(key)

    print("Update weights for each layer and expert")
    for layer_id in range(num_layers):
        print(f"Layer {layer_id}")
        for expert_id in range(num_experts):
            print(f"    expert {expert_id}")
            for weight_suffix in ["w12.weight", "w3.weight"]:
                tensor_name = (
                    f"module.layers.{layer_id}.ff.experts.{expert_id}.{weight_suffix}"
                )
                candidate_weight = f"module.layers.{layer_id}.ff.{weight_suffix}"
                moe[tensor_name] = (
                    candidates[expert_id][candidate_weight].clone().detach()
                )

    print("Update position embeddings and image end of input token") 
    num_image_tokens = moe["module.pos_embs"].shape[0]
    for key in ["module.pos_embs", "module.image_eoi"]:
        moe[key] = candidates[0][key][:num_image_tokens, ...].clone().detach()

    print(f"Discarded tensors: {discarded_tensors}")


def initialize_perceiver_model(
    moe_checkpoint_path: str,
    candidate_checkpoints_path: list[str],
    output_checkpoint_path: str,
    num_experts: int = 4,
    num_layers: int = 22,
) -> None:
    """Initialize the Perceiver model."""

    print(f"Loading MoE template: {moe_checkpoint_path}")
    moe = torch.load(moe_checkpoint_path)
    
    candidates = []
    for i, candidate_checkpoint in enumerate(candidate_checkpoints_path):
        print(f"Loading candidate for expert #{i}")
        candidate = torch.load(candidate_checkpoint)
        candidates.append(candidate)

    print("Initializing MoE")
    initialize_moe(moe, candidates, num_layers=num_layers, num_experts=num_experts)
    
    print(f"Saving initialized MoE to {output_checkpoint_path}")
    initialization_log = {
        "moe_checkpoint_path": moe_checkpoint_path,
        "candidate_checkpoints_path": candidate_checkpoints_path,
        "num_experts": num_experts,
        "num_layers": num_layers
    }
    
    try:
        os.mkdir(output_checkpoint_path)
    except FileExistsError:
        raise FileExistsError(f"Directory '{output_checkpoint_path}' already exists.")
    
    torch.save(moe, f"{output_checkpoint_path}/perception_tokenizer.pt")
    
    with open(f"{output_checkpoint_path}/initialization_log.json", "w") as f:
        json.dump(initialization_log, f, indent=4)
    
    print("Done")
    
if __name__ == "__main__":
    stage1_checkpoint="/fsx_0/checkpoints/mm10.1/MM10.1_Stage1_70B/MH22final_70B_ViTH_336px_R1_idl/checkpoint-17800/perception_tokenizer.pt"
    
    num_layers = 22
    num_experts = 8
    initialize_perceiver_model(
        moe_checkpoint_path=f"/fsx_3/bucket/tranx/moe/templates/perception_tokenizer_moe_22x{num_experts}x2.pt",
        candidate_checkpoints_path=[stage1_checkpoint]*num_experts,
        output_checkpoint_path=f"/fsx_3/bucket/tranx/moe/initialized/perception_tokenizer_mm10.1_17800_22x{num_experts}",
        num_experts=num_experts,
        num_layers=num_layers
    )