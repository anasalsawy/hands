"""The mouth that can change and still bill per token.

She speaks from Qwen/Qwen3.8-27B on CoreWeave serverless. A request names
that model, or a LoRA file trained on it. AL and AR are two of those
files, rank at most 16. A new file replaces the old one. Speech is billed
per token, the same rate as the base model. Storage of the file is separate.
There is no GPU-hour deployment on this path.
"""
from __future__ import annotations

import os

BASE_MODEL = "Qwen/Qwen3.8-27B"
COREWEAVE = "https://api.inference.wandb.ai/v1"
MAX_RANK = 16

# How far the fast copy moves back toward the untouched base on one home
# reply. 0.002 of the gap that is still left. About 350 home replies to
# halve a change, about 1,150 until a tenth is left. The slow copy does
# not use this. It does not decay.
COOL = 0.002
ALPHA = 0.01


def is_coreweave(base_url: str) -> bool:
    return "inference.wandb.ai" in (base_url or "").lower()


def artifact_ref(entity: str, project: str, name: str, version: str = "latest") -> str:
    """The model id a chat request uses for one uploaded adapter."""
    return f"wandb-artifact:///{entity}/{project}/{name}:{version}"


def publish_adapter(directory: str, name: str, *, entity: str, project: str,
                    base_model: str, api_key: str) -> str:
    """Upload one PEFT adapter and return the id the next reply names."""
    import wandb

    os.environ["WANDB_API_KEY"] = api_key
    run = wandb.init(entity=entity, project=project, job_type="mouth", reinit="finish_previous")
    try:
        artifact = wandb.Artifact(
            name, type="lora", metadata={"wandb.base_model": base_model},
        )
        artifact.add_dir(directory)
        run.log_artifact(artifact)
    finally:
        run.finish()
    return artifact_ref(entity, project, name)
