"""Safe-weight reconstruction of the pinned ScaleBFM-M deployment network."""
import json
from pathlib import Path
from types import SimpleNamespace
import torch
from safetensors.torch import load_file
import native_network
import tensor_native
from motion_challenge.validation import sha256_file


def load(directory, device="cpu", expected_source=None):
    directory=Path(directory)
    receipt=json.loads((directory/"conversion.json").read_text())
    if expected_source and receipt["source_sha256"]!=expected_source:raise ValueError("Source checkpoint mismatch")
    for name,digest in receipt["files"].items():
        if sha256_file(directory/name)!=digest:raise ValueError("Converted file checksum mismatch: "+name)
    metadata=json.loads((directory/"metadata.json").read_text());cfg=metadata["policy_architecture"]
    actor=native_network.HumanoidTransformer(cfg['prop_obs_dim'],cfg['action_dim'],cfg['output_dim'],cfg['embedding_dim'],cfg['num_heads'],cfg['ff_dim'],cfg['num_layers']).eval()
    embed=native_network.TaskEmbedder(cfg['task_obs_dim'],cfg['embedding_dim'],cfg['reduced_task_dim'],cfg['task_embedder_hidden_dims']).eval()
    state=load_file(str(directory/"policy.safetensors"))
    policy=tensor_native.HumanoidTransformerPolicyWrapperWithMode(
        SimpleNamespace(actor=actor,actor_task_embedder=embed),state['mode_mappings'],state['mode_vectors'],
        state['default_dof_pos'],state['action_scale'],metadata['history_buffer_size'],len(metadata['future_idx']),
        state['local_translation'],state['local_rotation'],state['parent_indices'],state['joint_axis'],
        state['selected_link_indices'],state['lab_to_xml_joint_indices']).eval()
    policy.load_state_dict(state,strict=True,assign=True)
    return policy.to(device),metadata
