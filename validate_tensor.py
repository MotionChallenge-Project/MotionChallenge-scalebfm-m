
import sys,types,json,argparse
from pathlib import Path
import numpy as np,torch
root=Path(__file__).resolve().parent
parser=argparse.ArgumentParser();parser.add_argument('--prepared',required=True,type=Path);parser.add_argument('--weights',required=True,type=Path);parser.add_argument('--artifact-map',required=True,type=Path);parser.add_argument('--source',required=True,type=Path);parser.add_argument('--device',default='cuda:0')
a=parser.parse_args();sys.path.insert(0,str(root))
from motion_challenge.resources import resource_root
from motion_challenge.simulation.physics import World
from motion_challenge.data.reference import Motion
from motion_challenge.simulation.tensor_reference import TensorMotionBank,TensorReference
from motion_challenge.sdk import DeployableStateBatch,DeployableStatePacket
from tensor_adapter import TensorScaleBFM
from native_runtime import load_policy
from adapter import ScaleBFM
torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.set_num_threads(1)
runtime=json.loads((root/'runtime.mjwarp.json').read_text())
source_map=json.loads(a.artifact_map.read_text())['artifacts'];paths={k:Path(v['path']) for k,v in source_map.items()}
policy,metadata,native=load_policy(paths,a.source)
cfg=json.loads((root/'policy.json').read_text())
meta=json.loads((a.prepared/'dataset.json').read_text());poses=np.load(a.prepared/'poses.npy',mmap_mode='r')
worlds=[World(resource_root()/'robots/g1-29dof-mjwarp-v1.json') for _ in range(7)];names=worlds[0].joint_names
for clip,w in enumerate(worlds):
 p=poses[meta['starts'][clip]:meta['starts'][clip]+meta['lengths'][clip]]
 w.reset(Motion(names,tuple(np.arange(len(p))/meta['fps']),p[:,7:],p[:,:3],p[:,3:7]))
bank=TensorMotionBank(np.array(poses),meta['starts'],meta['lengths'],np.full(len(meta['names']),meta['fps']),meta['names'],a.device)
refs=TensorReference(worlds[0],bank,42,a.device)
adapter=TensorScaleBFM(root,runtime,names,7,a.device,a.weights);original=[]
for _ in range(7):
 old=ScaleBFM.__new__(ScaleBFM);old.config=cfg;old.policy=policy;old.metadata=metadata;old.native=native;old.names=metadata['joint_names'];old.mode=torch.tensor([cfg['control_mode']]);old.offsets=torch.tensor(metadata['future_idx'])[None,:,None];original.append(old)
rng=np.random.default_rng(2027);observation_error=0.;action_error=0.;target_error=0.
for step in range(15):
 times=np.arange(7)[:,None]*.03+step*.02+np.array(runtime['reference_offsets_s'])
 reference=refs.motion(torch.arange(7,device=a.device),torch.tensor(times,device=a.device))
 q=rng.normal(0,.2,(7,29));qd=rng.normal(0,.4,(7,29));rotation=rng.normal(size=(7,4));rotation/=np.linalg.norm(rotation,axis=1,keepdims=True);omega=rng.normal(0,.1,(7,3))
 state=DeployableStateBatch(names,torch.full((7,),step*.02,device=a.device),*[torch.tensor(x,device=a.device) for x in (q,qd,rotation,omega)])
 if step==8:adapter.reset(torch.tensor([1,5],device=a.device))
 actual=adapter.observations(state,reference);expected=[[] for _ in actual]
 expected_output=[]
 for row,(world,old) in enumerate(zip(worlds,original)):
  packet=DeployableStatePacket(step*.02,names,tuple(q[row]),tuple(qd[row]),tuple(rotation[row]),tuple(omega[row]))
  frames=world.references(times[row])
  if step==0 or (step==8 and row in (1,5)):old.reset(packet,frames,0)
  inputs=old.observations(packet,frames)
  for j,x in enumerate(inputs):expected[j].append(x)
  with torch.inference_mode():expected_output.append(old.policy(*inputs))
 for j in range(len(actual)):
  e=torch.cat(expected[j]).numpy();x=actual[j].cpu().numpy()
  error=float(np.max(abs(x-e)));observation_error=max(observation_error,error)
  np.testing.assert_allclose(x,e,atol=2e-5,rtol=2e-5,err_msg=str(j))
 with torch.inference_mode():target,action=adapter.policy(*actual)
 et=torch.cat([x[0] for x in expected_output]);ea=torch.cat([x[1] for x in expected_output])
 action_error=max(action_error,float((action.cpu()-ea).abs().max()));target_error=max(target_error,float((target.cpu()-et).abs().max()))
 np.testing.assert_allclose(action.cpu().numpy(),ea.numpy(),atol=3e-5,rtol=2e-5)
 np.testing.assert_allclose(target.cpu().numpy(),et.numpy(),atol=2e-5,rtol=2e-5)
 adapter.last.copy_(ea.to(a.device))
 for row,old in enumerate(original):old.last=ea[row:row+1].clone()
report={'status':'passed','steps':15,'batch':7,'partial_reset_slots':[1,5],'observation_max_abs_error':observation_error,'action_max_abs_error':action_error,'pd_target_max_abs_error_rad':target_error,'reference':'Unmodified author reference FK, history and policy; matched action history'}
(root/'validation/tensor-observation-parity.json').write_text(json.dumps(report,indent=2)+'\n')
runtime['tensor_entrypoint']='tensor_adapter:TensorScaleBFM';(root/'runtime.mjwarp.json').write_text(json.dumps(runtime,indent=2)+'\n')
print(json.dumps(report),flush=True)

