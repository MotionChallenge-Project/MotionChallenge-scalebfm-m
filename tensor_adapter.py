"""ScaleBFM-M batched history and adapter-owned reference kinematics."""
import json
from pathlib import Path
import torch
import tensor_native as native
from tensor_policy import load
from motion_challenge.simulation.tensor_math import normalize,conjugate,rotate


class TensorScaleBFM:
    def __init__(self,root,runtime,joint_names,slots,device,converted_weights):
        self.device=torch.device(device)
        self.cfg=json.loads((Path(root)/"policy.json").read_text())
        if not self.cfg['reference_forcing']:raise ValueError("Root forcing must use reference information")
        self.policy,self.metadata=load(converted_weights,device,runtime['weights_sha256'])
        self.names=self.metadata['joint_names'];self.order=[joint_names.index(n) for n in self.names]
        self.inverse=[self.names.index(n) for n in joint_names]
        count=self.metadata['history_buffer_size']
        self.buffers={k:torch.zeros((slots,count,n),device=device) for k,n in [('root_pos_buffer',3),('root_quat_wxyz_buffer',4),('base_ang_vel_buffer',3),('dof_pos_buffer',29),('dof_vel_buffer',29),('action_buffer',29)]}
        self.last=torch.zeros((slots,29),device=device)
        self.fresh=torch.ones(slots,dtype=torch.bool,device=device)
        self.mode=torch.full((slots,),self.cfg['control_mode'],dtype=torch.long,device=device)
        self.offsets=torch.tensor(self.metadata['future_idx'],dtype=torch.long,device=device)[None,:,None].expand(slots,-1,-1)
        self.parents=self.policy.parent_indices.cpu().tolist()
        self.kp=torch.tensor(self.metadata['stiffness'],device=device,dtype=torch.float32)[self.inverse]
        self.kd=torch.tensor(self.metadata['damping'],device=device,dtype=torch.float32)[self.inverse]

    def reset(self,slots):
        self.last[slots]=0;self.buffers['action_buffer'][slots]=0;self.fresh[slots]=True

    def reference_bodies(self,reference):
        shape=reference.joint_position_rad.shape[:2];model=self.policy
        q=reference.joint_position_rad[:,:,self.order].float().reshape(-1,29)
        half=q[:,model.lab_to_xml_joint_indices,None]/2
        joint_rotation=torch.cat((torch.cos(half),model.joint_axis[None]*torch.sin(half)),dim=-1)
        count=len(self.parents);n=q.shape[0]
        positions=torch.zeros((n,count,3),device=self.device);rotations=torch.zeros((n,count,4),device=self.device);rotations[...,0]=1
        positions[:,0]=reference.root_position_world_m.float().reshape(n,3)
        rotations[:,0]=reference.root_orientation_xyzw[..., [3,0,1,2]].float().reshape(n,4)
        for index in range(1,count):
            parent=self.parents[index];parent_rotation=rotations[:,parent]
            positions[:,index]=positions[:,parent]+native.quat_apply(parent_rotation,model.local_translation[index-1:index].expand(n,-1))
            local=native.quat_mul(model.local_rotation[index-1:index].expand(n,-1),joint_rotation[:,index-1])
            rotations[:,index]=native.quat_mul(parent_rotation,local)
        selected=model.selected_link_indices
        return positions[:,selected].reshape(*shape,-1,3),rotations[:,selected].reshape(*shape,-1,4)

    def observations(self,state,reference):
        inv=conjugate(normalize(state.root_orientation_xyzw.double()))
        sample={
          'root_pos_buffer':reference.root_position_world_m[:,0].float(),
          'root_quat_wxyz_buffer':state.root_orientation_xyzw[:,[3,0,1,2]].float(),
          'base_ang_vel_buffer':rotate(inv,state.root_angular_velocity_world_rad_s.double()).float(),
          'dof_pos_buffer':state.joint_position_rad[:,self.order].float(),
          'dof_vel_buffer':state.joint_velocity_rad_s[:,self.order].float(),
        }
        for key,value in sample.items():
            buf=self.buffers[key];shifted=torch.cat((buf[:,1:],value[:,None]),dim=1)
            buf.copy_(torch.where(self.fresh[:,None,None],value[:,None],shifted))
        a=self.buffers['action_buffer'];shifted=torch.cat((a[:,1:],self.last[:,None]),dim=1)
        a.copy_(torch.where(self.fresh[:,None,None],torch.zeros_like(a),shifted))
        positions,rotations=self.reference_bodies(reference)
        self.buffers['root_pos_buffer'][:,-1]=torch.where(self.fresh[:,None],self.buffers['root_pos_buffer'][:,-1],positions[:,0,0])
        root=self.buffers['root_pos_buffer'][:,-1][:,None,None]
        quat=self.buffers['root_quat_wxyz_buffer'][:,-1][:,None,None].expand_as(rotations)
        target_position=native.quat_apply_inverse(quat,positions-root)
        target_rotation=native.quat_mul_inverse_left(quat,rotations)
        self.fresh.zero_()
        return (self.buffers['root_quat_wxyz_buffer'],self.buffers['base_ang_vel_buffer'],self.buffers['dof_pos_buffer'],self.buffers['dof_vel_buffer'],self.buffers['action_buffer'],target_position,target_rotation,self.mode,self.offsets)

    @torch.inference_mode()
    def act(self,state,reference):
        target,action=self.policy(*self.observations(state,reference))
        target=target[:,self.inverse];self.last.copy_(action)
        zeros=torch.zeros_like(target)
        return torch.stack((target,zeros,self.kp.expand_as(target),self.kd.expand_as(target),zeros),dim=1)
