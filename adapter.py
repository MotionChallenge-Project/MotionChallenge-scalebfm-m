"""ScaleBFM M: original CPU network, author whole-body mode and input history."""
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation
import torch

from motion_challenge.sdk import ControlCommand
from motion_challenge.validation import sha256_file
from native_runtime import load_policy


class ScaleBFM:
    def __init__(self, root, runtime):
        root = Path(root)
        if sha256_file(root / 'policy.json') != runtime['inference_config_sha256']:
            raise ValueError('ScaleBFM configuration hash mismatch')
        self.config = json.loads((root / 'policy.json').read_text())
        if not self.config['reference_forcing']:
            raise ValueError('ScaleBFM requires reference-only root forcing')
        weights = root / runtime['weights']
        if sha256_file(weights) != runtime['weights_sha256']:
            raise ValueError('ScaleBFM weight hash mismatch')
        paths = {}
        for key, item in runtime['artifacts'].items():
            paths[key] = root / item['path']
            if sha256_file(paths[key]) != item['sha256']:
                raise ValueError(f'ScaleBFM artifact hash mismatch: {key}')
        torch.set_num_threads(1)
        self.policy, self.metadata, self.native = load_policy(paths, weights)
        self.names = self.metadata['joint_names']
        self.mode = torch.tensor([self.config['control_mode']], dtype=torch.long)
        self.offsets = torch.tensor(self.metadata['future_idx'], dtype=torch.long)[None, :, None]

    def sample(self, state, reference_root):
        rotation = Rotation.from_quat(state.root_orientation_xyzw)
        return {
            'root_pos_buffer': np.asarray(reference_root),
            'root_quat_wxyz_buffer': np.asarray(state.root_orientation_xyzw)[[3, 0, 1, 2]],
            'base_ang_vel_buffer': rotation.inv().apply(state.root_angular_velocity_world_rad_s),
            'dof_pos_buffer': np.asarray(state.joint_position_rad)[self.order],
            'dof_vel_buffer': np.asarray(state.joint_velocity_rad_s)[self.order],
        }

    def reset(self, state, reference, seed):
        self.order = [state.joint_names.index(n) for n in self.names]
        self.inverse = [self.names.index(n) for n in state.joint_names]
        count = self.metadata['history_buffer_size']
        self.buffers = {k: torch.tensor(v, dtype=torch.float32)[None, None].repeat(1, count, 1)
                        for k, v in self.sample(state, reference[0].root_position_world_m).items()}
        self.buffers['action_buffer'] = torch.zeros(1, count, 29)
        self.last = torch.zeros(1, 29)
        # Reference-only initialization: extend author reference forcing to reset.
        # No robot position is observed or integrated.
        self.first = True
        self.reference_cache = {}

    def reference_bodies(self, frame):
        key = frame.source_time_s
        if key not in self.reference_cache:
            # Preserve the author's float32 quaternion FK operation order.
            model = self.policy
            q = torch.tensor(np.asarray(frame.joint_position_rad)[self.order], dtype=torch.float32)
            half = q[model.lab_to_xml_joint_indices, None] / 2
            joint_rotation = torch.cat([torch.cos(half), model.joint_axis * torch.sin(half)], -1)
            count = len(model.parent_indices)
            positions = torch.zeros(count, 3)
            rotations = torch.zeros(count, 4)
            rotations[:, 0] = 1
            positions[0] = torch.tensor(frame.root_position_world_m)
            rotations[0] = torch.tensor(np.asarray(frame.root_orientation_xyzw)[[3, 0, 1, 2]])
            for index in range(1, count):
                parent = model.parent_indices[index]
                parent_rotation = rotations[parent:parent + 1]
                positions[index] = positions[parent] + self.native.quat_apply(
                    parent_rotation, model.local_translation[index - 1:index])[0]
                local = self.native.quat_mul(model.local_rotation[index - 1:index],
                                            joint_rotation[index - 1:index])
                rotations[index] = self.native.quat_mul(parent_rotation, local)[0]
            selected = model.selected_link_indices
            self.reference_cache[key] = (positions[selected].numpy(), rotations[selected].numpy())
        return self.reference_cache[key]

    def observations(self, state, reference):
        if not self.first:
            for key, value in self.sample(state, reference[0].root_position_world_m).items():
                self.buffers[key] = torch.roll(self.buffers[key], -1, 1)
                self.buffers[key][:, -1] = torch.tensor(value, dtype=torch.float32)
            self.buffers['action_buffer'] = torch.roll(self.buffers['action_buffer'], -1, 1)
            self.buffers['action_buffer'][:, -1] = self.last
        body = [self.reference_bodies(frame) for frame in reference]
        positions = torch.tensor(np.stack([x[0] for x in body]), dtype=torch.float32)[None]
        quaternions = torch.tensor(np.stack([x[1] for x in body]), dtype=torch.float32)[None]
        if self.config['reference_forcing'] and not self.first:
            self.buffers['root_pos_buffer'][:, -1] = positions[:, 0, 0]
        self.first = False
        root = self.buffers['root_pos_buffer'][:, -1][:, None, None]
        quat = self.buffers['root_quat_wxyz_buffer'][:, -1][:, None, None].expand_as(quaternions)
        target_position = self.native.quat_apply_inverse(quat, positions - root)
        target_rotation = self.native.quat_mul_inverse_left(quat, quaternions)
        self.reference_cache = {k: v for k, v in self.reference_cache.items()
                                if k >= reference[0].source_time_s - 1e-9}
        return (self.buffers['root_quat_wxyz_buffer'], self.buffers['base_ang_vel_buffer'],
                self.buffers['dof_pos_buffer'], self.buffers['dof_vel_buffer'],
                self.buffers['action_buffer'], target_position, target_rotation, self.mode, self.offsets)

    def act(self, state, reference):
        with torch.inference_mode():
            target, action = self.policy(*self.observations(state, reference))
        if not torch.isfinite(target).all() or not torch.isfinite(action).all():
            raise ValueError('Nonfinite ScaleBFM output')
        self.last = action.clone()
        def reorder(value):
            return tuple(float(x) for x in np.asarray(value)[self.inverse])
        return ControlCommand(reorder(target[0].numpy()), (0.,) * 29,
                              reorder(self.metadata['stiffness']),
                              reorder(self.metadata['damping']), (0.,) * 29)

    def close(self):
        self.policy = None


def create(root, runtime):
    return ScaleBFM(root, runtime)
