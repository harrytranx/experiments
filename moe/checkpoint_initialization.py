# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

import argparse

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
        f"Number of parameters in MoE: {moe_num_params}, in candidate: {candidate_num_params}"
    )

    for key in expert.keys():
        if key in moe.keys() and expert[key].shape == moe[key].shape:
            print("matched keys:", key, expert[key].shape)
            moe[key] = expert[key].clone().detach()
        else:
            discarded_tensors.add(key)

    # Update weights for each layer and expert
    for layer_id in range(num_layers):
        for expert_id in range(num_experts):
            for weight_suffix in ["w12.weight", "w3.weight"]:
                tensor_name = (
                    f"module.layers.{layer_id}.ff.experts.{expert_id}.{weight_suffix}"
                )
                candidate_weight = f"module.layers.{layer_id}.ff.{weight_suffix}"
                moe[tensor_name] = (
                    candidates[expert_id][candidate_weight].clone().detach()
                )

    # Update position embeddings and image end of input token
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

    moe = torch.load(moe_checkpoint_path)
    candidates = []
    for candidate_checkpoint in candidate_checkpoints_path:
        candidate = torch.load(candidate_checkpoint)
        candidates.append(candidate)

    initialize_moe(moe, candidates, num_layers=num_layers, num_experts=num_experts)
    torch.save(moe, output_checkpoint_path)


def invoke_main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--moe_checkpoint_path",
        default=None,
        type=str,
        help="MoE checkpoint path",
    )
    parser.add_argument(
        "--candidate_checkpoints_path",
        default=None,
        type=str,
        action="append",
        help="candidate checkpoints path",
    )
    parser.add_argument(
        "--output_checkpoint_path",
        default=None,
        type=str,
        help="output checkpoint path",
    )
    parser.add_argument(
        "--num_experts",
        default=None,
        type=int,
        help="Number of experts in mixture",
    )
    parser.add_argument(
        "--num_layers",
        default=None,
        type=int,
        help="Number of transformer layers",
    )
    args = parser.parse_args()

    initialize_perceiver_model(
        moe_checkpoint_path=args.moe_checkpoint_path,
        candidate_checkpoints_path=args.candidate_checkpoints_path,
        output_checkpoint_path=args.output_checkpoint_path,
        num_experts=args.num_experts,
        num_layers=args.num_layers,
    )


if __name__ == "__main__":
    invoke_main()  # pragma: no cover
