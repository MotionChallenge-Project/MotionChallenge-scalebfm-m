"""Load hash-pinned author network and export definitions without IsaacLab startup."""
import ast
import importlib.util
import json
import sys
from types import ModuleType, SimpleNamespace
import xml.etree.ElementTree as ET

import numpy as np
import torch


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def load_export(path):
    names = {'quat_apply', 'quat_apply_inverse', 'quat_mul',
             'quat_mul_inverse_left', 'quat_mul_inverse_right',
             'HumanoidTransformerPolicyWrapperWithMode', 'build_mode_mappings', 'parse_xml'}
    nodes = [n for n in ast.parse(path.read_text()).body
             if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in names]
    assert {n.name for n in nodes} == names
    module = ModuleType('motionchallenge_scale_author_export')
    module.__file__ = str(path)
    module.__dict__.update(torch=torch, nn=torch.nn, np=np, ET=ET)
    sys.modules[module.__name__] = module
    # Function bodies and decorators remain unchanged; omit simulator startup.
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), 'exec'), module.__dict__)
    return module


def load_policy(paths, weights):
    metadata = json.loads(paths['metadata'].read_text())
    cfg = metadata['policy_architecture']
    module = load_module('motionchallenge_scale_author_network', paths['network'])
    native = load_export(paths['export'])
    actor = module.HumanoidTransformer(
        cfg['prop_obs_dim'], cfg['action_dim'], cfg['output_dim'], cfg['embedding_dim'],
        cfg['num_heads'], cfg['ff_dim'], cfg['num_layers']).eval()
    embed = module.TaskEmbedder(cfg['task_obs_dim'], cfg['embedding_dim'],
                               cfg['reduced_task_dim'], cfg['task_embedder_hidden_dims']).eval()
    state = torch.load(weights, map_location='cpu', weights_only=True)['model_state_dict']
    for prefix, target in [('actor.', actor), ('actor_task_embedder.', embed)]:
        target.load_state_dict({k[len(prefix):]: v for k, v in state.items()
                                if k.startswith(prefix)}, strict=True)
    bodies, joints, parents, axes, positions, rotations = native.parse_xml(str(paths['robot']))
    rotations = rotations / rotations.norm(dim=-1, keepdim=True)
    modes = torch.load(paths['mode_table'], map_location='cpu', weights_only=True)
    mappings = native.build_mode_mappings(modes, metadata['mode_feature_dims'],
                                         with_time=metadata['mode_mapping_with_time'])
    policy = native.HumanoidTransformerPolicyWrapperWithMode(
        SimpleNamespace(actor=actor, actor_task_embedder=embed), mappings, modes,
        torch.tensor([metadata['default_dof_pos']], dtype=torch.float32),
        torch.tensor([metadata['action_scale']], dtype=torch.float32),
        metadata['history_buffer_size'], len(metadata['future_idx']), positions, rotations,
        parents, axes, torch.tensor([bodies.index(n) for n in metadata['selected_body_names']]),
        torch.tensor([metadata['joint_names'].index(n) for n in joints])).eval()
    return policy, metadata, native
