"""Full 4096-environment inference and independent seven-slot consistency check."""
import argparse,dataclasses,importlib,json,time
from pathlib import Path
import numpy as np,torch
from motion_challenge.resources import resource_root
from motion_challenge.simulation.physics import World
from motion_challenge.simulation.tensor_reference import TensorMotionBank,TensorReference
from motion_challenge.sdk import DeployableStateBatch
p=argparse.ArgumentParser();p.add_argument('--prepared',type=Path,required=True);p.add_argument('--weights',type=Path,required=True);p.add_argument('--device',default='cuda:0');p.add_argument('--slots',type=int,default=4096);a=p.parse_args();root=Path(__file__).resolve().parent
runtime=json.loads((root/'runtime.mjwarp.json').read_text());mod,name=runtime['tensor_entrypoint'].split(':');cls=getattr(importlib.import_module(mod),name);torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False
world=World(resource_root()/'robots/g1-29dof-mjwarp-v1.json');meta=json.loads((a.prepared/'dataset.json').read_text());poses=np.load(a.prepared/'poses.npy',mmap_mode='r');poses=np.array(poses[meta['starts'][0]:meta['starts'][0]+100]);bank=TensorMotionBank(poses,[0],[len(poses)],[meta['fps']],['capacity-check'],a.device);reference=TensorReference(world,bank,a.slots*len(runtime['reference_offsets_s']),a.device)
ids=torch.zeros(a.slots,dtype=torch.long,device=a.device);sample_ids=torch.tensor([0,1,32,128,1024,2048,a.slots-1],device=a.device);offsets=torch.tensor(runtime['reference_offsets_s'],device=a.device);times=torch.linspace(.4,1.2,a.slots,device=a.device,dtype=torch.float64)
full=cls(root,runtime,world.joint_names,a.slots,a.device,a.weights);small=cls(root,runtime,world.joint_names,7,a.device,a.weights)
network=next(getattr(full,k) for k in ('actor','policy','network') if hasattr(full,k));sizes=[];handle=network.register_forward_pre_hook(lambda module,inputs:sizes.append(int(inputs[0].shape[0])))
def subset(packet):return dataclasses.replace(packet,**{f.name:getattr(packet,f.name)[sample_ids] for f in dataclasses.fields(packet) if isinstance(getattr(packet,f.name),torch.Tensor)})
maximum=0.;seconds=[];torch.cuda.reset_peak_memory_stats(a.device)
with torch.inference_mode():
 for step in range(4):
  refs=reference.motion(ids,times[:,None]+step*.02+offsets)
  state=DeployableStateBatch(world.joint_names,times+step*.02,refs.joint_position_rad[:,0],refs.joint_velocity_rad_s[:,0],refs.root_orientation_xyzw[:,0],refs.root_angular_velocity_world_rad_s[:,0])
  if step==2:full.reset(sample_ids[[1,6]]);small.reset(torch.tensor([1,6],device=a.device))
  torch.cuda.synchronize(a.device);begin=time.perf_counter();actual=full.act(state,refs);torch.cuda.synchronize(a.device);seconds.append(time.perf_counter()-begin)
  expected=small.act(subset(state),subset(refs));difference=float((actual[sample_ids]-expected).abs().max());maximum=max(maximum,difference)
  assert actual.shape==(a.slots,5,29) and torch.isfinite(actual).all();torch.testing.assert_close(actual[sample_ids],expected,atol=1e-4,rtol=1e-4)
handle.remove();assert sizes==[a.slots]*4,sizes
report={'status':'passed','slots':a.slots,'forward_batch_sizes':sizes,'steps':4,'partial_reset_slots':[1,a.slots-1],'seven_slot_consistency_max_abs_error':maximum,'forward_seconds':seconds,'peak_torch_allocated_bytes':torch.cuda.max_memory_allocated(a.device),'peak_torch_reserved_bytes':torch.cuda.max_memory_reserved(a.device),'device':torch.cuda.get_device_name(a.device)}
(root/'validation/full-batch-capacity.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)
